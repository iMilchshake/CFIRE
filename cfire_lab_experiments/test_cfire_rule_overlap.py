from pathlib import Path
from typing import Callable, Dict, List, Sequence, Set, Tuple, TypedDict, Optional
import copy
import logging
from collections import Counter
import json

import numpy as np
import torch

from lxg.datasets import RandomSeed
import lxg.datasets as datasets
from lxg.models import make_ff
from lxg.util import restore_checkpoint
from cfire.cfire_module import CFIRE
from cfire.util import __preprocess_explanations_ext
from cfire_lab_experiments.util import loader_to_tensor
from .test_cfire import ks_fn_cached, pprint_dnf_rules

# PRUNE_WINS_THRESHOLDS = [0, 1, 2, 3, 4, 5]
PRUNE_WINS_THRESHOLDS = list(range(0, 25 + 1))
MODEL_CKPT = Path("./models/tmp.ckpt")
EXPLANATIONS_PT = Path("./models/explanations.pt")
SUMMARY_OUT = Path("./cfire_lab_experiments/rule-overlap-analysis/summary.json")

# ensure parent folder(s) exists
SUMMARY_OUT.parent.mkdir(exist_ok=True)


# --- type helpers ---
ClauseKey = Tuple[int, int]  # (class, clause-id)


class PerfDict(TypedDict):
    accuracy: float  # i could also add f1 here, but i dont really use it


# --- generic helpers ---
def canon(rule: Sequence) -> Tuple:
    return tuple(rule) if isinstance(rule, list) else rule


def build_perf_by_key(cf: CFIRE) -> Dict[ClauseKey, PerfDict]:
    return {
        (cls, cid): cf.dnf.rule_performances[cls][canon(rule)]
        for cls, rules in enumerate(cf.dnf.rules)
        for cid, rule in enumerate(rules)
    }


def prune_rules(rule_tree: Sequence[Sequence], to_remove: Set[ClauseKey]):
    return [
        [r for cid, r in enumerate(rules) if (cls, cid) not in to_remove]
        for cls, rules in enumerate(rule_tree)
    ]


# --- cfire ---
def load_data():
    loaders = datasets.get_abalone()
    _, test_loader, val_loader, n_dim, n_classes = loaders
    X_val, _ = loader_to_tensor(val_loader)
    X_test, _ = loader_to_tensor(test_loader)
    return X_val, X_test, n_dim, n_classes


def build_model(n_dim: int, n_classes: int):
    model = make_ff([n_dim, 128, 128, n_classes], torch.nn.ReLU).to("cpu")
    restore_checkpoint(MODEL_CKPT, model, train=False)
    return model


def fit_cfire(model, X_val: torch.Tensor):
    expl_bin: Callable = lambda x: __preprocess_explanations_ext(x, threshold=0.01) > 0
    with RandomSeed(42):
        cfire = CFIRE(
            localexplainer_fn=ks_fn_cached(EXPLANATIONS_PT),
            inference_fn=model.predict_batch_softmax,
            expl_binarization_fn=expl_bin,
        )
        cfire.fit(X_val.numpy(), model.predict_batch(X_val).numpy())
    return cfire


# --- analysis ---
def collect_match_info(cfire: CFIRE, X_val: torch.Tensor):
    cfire_out = cfire(X_val, explain=True)
    clause_keys: list[ClauseKey] = []
    match_per_sample: list[list[ClauseKey]] = []
    for _, matches in cfire_out:
        keys = [k for k, _ in matches]
        match_per_sample.append(keys)
        for k in keys:
            if k not in clause_keys:
                clause_keys.append(k)
    return clause_keys, match_per_sample


def compute_winners(
    match_per_sample: List[List[ClauseKey]],
    clause_keys: List[ClauseKey],
    perf_by_key: Dict[ClauseKey, PerfDict],
):
    key2col = {k: i for i, k in enumerate(clause_keys)}

    def best_key(keys: List[ClauseKey]) -> ClauseKey:
        return max(keys, key=lambda k: perf_by_key[k]["accuracy"])

    n_samples, n_clauses = len(match_per_sample), len(clause_keys)
    winner_idx = np.full(n_samples, -1, int)
    wins = np.zeros(n_clauses, int)
    loss = np.zeros(n_clauses, int)

    for s, keys in enumerate(match_per_sample):
        if not keys:
            continue
        w = best_key(keys)
        w_col = key2col[w]
        winner_idx[s] = w_col
        for k in keys:
            (wins if k == w else loss)[key2col[k]] += 1

    winner_key_per_sample: list[Optional[ClauseKey]] = [
        clause_keys[idx] if idx != -1 else None for idx in winner_idx
    ]
    return wins, loss, winner_key_per_sample


def extra_stats(
    clause_keys: List[ClauseKey],
    match_per_sample: List[List[ClauseKey]],
    wins: np.ndarray,
    loss: np.ndarray,
    winner_key_per_sample: List[Optional[ClauseKey]],
    perf_by_key: Dict[ClauseKey, PerfDict],
):
    n_samples, n_clauses = len(match_per_sample), len(clause_keys)
    key2col = {k: i for i, k in enumerate(clause_keys)}

    M = np.zeros((n_samples, n_clauses), bool)
    coverage = np.zeros(n_clauses, int)
    match_counts = np.zeros(n_samples, int)
    for s, keys in enumerate(match_per_sample):
        match_counts[s] = len(keys)
        for k in keys:
            j = key2col[k]
            M[s, j] = True
            coverage[j] += 1

    logging.info("\nCoverage & overlap statistics")
    logging.info(
        "share of samples with ≥2 matches: %.2f%%", (match_counts >= 2).mean() * 100
    )
    logging.info("histogram of #-matches per sample:")
    for k in np.unique(match_counts):
        logging.info("  %d: %d", k, (match_counts == k).sum())

    overlap_matrix = M.astype(int).T @ M.astype(int)
    class_of = np.array([cls for cls, _ in clause_keys])
    intra = np.zeros(n_clauses, int)
    inter = np.zeros(n_clauses, int)

    for j in range(n_clauses):
        for k in range(j + 1, n_clauses):
            if overlap_matrix[j, k] == 0:
                continue
            if class_of[j] == class_of[k]:
                intra[j] += 1
                intra[k] += 1
            else:
                inter[j] += 1
                inter[k] += 1

    win_rate = np.divide(
        wins, coverage, out=np.zeros_like(wins, float), where=coverage > 0
    )

    logging.info("\ntop clauses by tie-breaker looses")
    for idx in np.argsort(-loss):
        class_id, clause_id = clause_keys[idx]
        logging.info(
            "cl %d/term %d | loss %4d | wins %4d | acc %.2f | winrate %.2f",
            class_id,
            clause_id,
            loss[idx],
            wins[idx],
            perf_by_key[(class_id, clause_id)]["accuracy"],
            win_rate[idx],
        )

    logging.info("\ntop clauses by tie-breaker wins")
    for idx in np.argsort(-wins):
        class_id, clause_id = clause_keys[idx]
        logging.info(
            "cl %d/term %d | loss %4d | wins %4d | acc %.2f | winrate %.2f",
            class_id,
            clause_id,
            loss[idx],
            wins[idx],
            perf_by_key[(class_id, clause_id)]["accuracy"],
            win_rate[idx],
        )

    logging.info("\ntop clauses by tie-breaker wins (least wins)")
    for idx in np.argsort(wins):
        class_id, clause_id = clause_keys[idx]
        logging.info(
            "cl %d/term %d | loss %4d | wins %4d | acc %.2f | winrate %.2f",
            class_id,
            clause_id,
            loss[idx],
            wins[idx],
            perf_by_key[(class_id, clause_id)]["accuracy"],
            win_rate[idx],
        )

    # calculate inter vs intra collision count
    intra_res = inter_res = 0
    for keys, w in zip(match_per_sample, winner_key_per_sample):
        if len(keys) <= 1 or w is None:
            continue
        other_cls = {k[0] for k in keys} - {w[0]}
        if other_cls:
            inter_res += 1
        else:
            intra_res += 1
    total_res = intra_res + inter_res
    logging.info(
        "\nRule collisions: %.2f%% intra-class | %.2f%% inter-class",
        intra_res / total_res * 100,
        inter_res / total_res * 100,
    )

    # return stats for dumping
    return dict(
        match_hist=dict(Counter(match_counts)),
        share_multi=float((match_counts >= 2).mean() * 100),
        collision_ratio=dict(
            intra=float(intra_res / total_res * 100) if total_res else 0.0,
            inter=float(inter_res / total_res * 100) if total_res else 0.0,
        ),
    )


def loser_winner_pairs(
    keys: Set[ClauseKey],
    match_per_sample: List[List[ClauseKey]],
    winner_key_per_sample: List[Optional[ClauseKey]],
):
    out: list[tuple[ClauseKey, ClauseKey]] = []
    for k in keys:
        for s, mks in enumerate(match_per_sample):
            w = winner_key_per_sample[s]
            if k in mks and w and w != k:
                out.append((k, w))
                break
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # build model & CFIRE
    X_val, X_test, n_dim, n_classes = load_data()
    model = build_model(n_dim, n_classes)
    cfire = fit_cfire(model, X_val)

    y_val = model.predict_batch(X_val).numpy()
    y_test = model.predict_batch(X_test).numpy()
    base_val_acc = (cfire(X_val) == y_val).mean()
    base_test_acc = (cfire(X_test) == y_test).mean()
    logging.info(
        "\nORIGINAL CFIRE acc vs model -> val %.3f | test %.3f",
        base_val_acc,
        base_test_acc,
    )
    pprint_dnf_rules(cfire.dnf.rules)

    clause_keys, match_per_sample = collect_match_info(cfire, X_val)
    perf_by_key = build_perf_by_key(cfire)
    wins, loss, winner_key_per_sample = compute_winners(
        match_per_sample, clause_keys, perf_by_key
    )

    # verbose stats
    extra = extra_stats(
        clause_keys,
        match_per_sample,
        wins,
        loss,
        winner_key_per_sample,
        perf_by_key,
    )

    # clause-level table for dump
    clause_stats = [
        dict(
            cls=k[0],
            term=k[1],
            wins=int(w),
            loss=int(l),
            acc=float(perf_by_key[k]["accuracy"]),
        )
        for k, w, l in zip(clause_keys, wins, loss)
    ]

    # pruning evaluation
    original_rules = copy.deepcopy(cfire.dnf.rules)
    total_rules = sum(len(r) for r in original_rules)
    pruning_results = []

    logging.info("\nPruning evaluation (absolute wins threshold)")
    logging.info(
        f"{'th':<3} {'kept/r':<7} {'val_acc (Δ,%)':<22}"
        f" {'test_acc (Δ,%)':<24} {'intra class coll%':<15}"
    )
    logging.info("-" * 70)

    for thr in PRUNE_WINS_THRESHOLDS:
        remove: Set[ClauseKey] = {
            clause_keys[i] for i, w in enumerate(wins) if w <= thr
        }
        new_rules = prune_rules(original_rules, remove)

        # pprint_dnf_rules(new_rules)
        saved = cfire.dnf.rules
        cfire.dnf.rules = new_rules
        val_acc = (cfire(X_val) == y_val).mean()
        test_acc = (cfire(X_test) == y_test).mean()
        cfire.dnf.rules = saved

        drop_val_pct = (val_acc - base_val_acc) / base_val_acc * 100
        drop_test_pct = (test_acc - base_test_acc) / base_test_acc * 100

        pairs = loser_winner_pairs(remove, match_per_sample, winner_key_per_sample)
        intra_coll_ratio = (
            (sum(1 for l, w in pairs if l[0] == w[0]) / len(pairs) * 100)
            if pairs
            else 0.0
        )

        kept_cnt = sum(len(r) for r in new_rules)
        kept_col = f"{kept_cnt}/{total_rules}"
        val_str = f"{val_acc:.3f} ({val_acc-base_val_acc:+.3f},{drop_val_pct:6.2f}%)"
        test_str = (
            f"{test_acc:.3f} ({test_acc-base_test_acc:+.3f},{drop_test_pct:6.2f}%)"
        )
        logging.info(
            f"{thr:<3} {kept_col:<7} {val_str:<22} {test_str:<24} {intra_coll_ratio:5.1f}%"
        )

        pruning_results.append(
            dict(
                thr=thr,
                kept=kept_cnt,
                total=total_rules,
                val_acc=float(val_acc),
                test_acc=float(test_acc),
                intra_coll_ratio=intra_coll_ratio,
            )
        )

    logging.info(
        f"Original CFIRE accuracy – validation: {base_val_acc:.3f}, test: {base_test_acc:.3f}"
    )

    match_hist_py = {int(k): int(v) for k, v in extra["match_hist"].items()}
    summary = dict(
        clause_stats=clause_stats,
        match_hist=match_hist_py,
        share_multi_match=extra["share_multi"],
        collision_ratio=extra["collision_ratio"],
        pruning=pruning_results,
    )
    with open(SUMMARY_OUT, "w") as f:
        json.dump(summary, f, indent=2)
    logging.info("Saved compact stats -> cfire_stats/summary.json")


if __name__ == "__main__":
    main()
