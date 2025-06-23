
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Set, Tuple, TypedDict, Optional
import copy
import logging
from collections import Counter

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

PRUNE_WINS_THRESHOLDS = [0, 1, 2, 3, 4, 5]
MODEL_CKPT = Path("./models/tmp.ckpt")
EXPLANATIONS_PT = Path("./models/explanations.pt")


# ------------- utils ------------

# typing helpers
ClauseKey = Tuple[int, int]  # (class, clause-id)


class PerfDict(TypedDict):
    accuracy: float


def canon(rule: Sequence) -> Tuple:
    """Hashable representation of a clause rule."""
    return tuple(rule) if isinstance(rule, list) else rule


def build_perf_by_key(cf: CFIRE) -> Dict[ClauseKey, PerfDict]:
    """Map (cls, cid) → performance dict."""
    return {
        (cls, cid): cf.dnf.rule_performances[cls][canon(rule)]
        for cls, rules in enumerate(cf.dnf.rules)
        for cid, rule in enumerate(rules)
    }


def predict_fast(cf: CFIRE, X: torch.Tensor | np.ndarray) -> np.ndarray:
    """CFIRE prediction without explanation generation."""
    tensor = X if isinstance(X, torch.Tensor) else torch.as_tensor(X)
    return cf(tensor)


def prune_rules(rule_tree: Sequence[Sequence], to_remove: Set[ClauseKey]):
    """Return deep-copy of *rule_tree* with specified clauses removed."""
    return [
        [r for cid, r in enumerate(rules) if (cls, cid) not in to_remove]
        for cls, rules in enumerate(rule_tree)
    ]


# ------------- model / cfire utils ------------


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


# ------------- analysis utils ------------


def collect_match_info(cfire: CFIRE, X_val: torch.Tensor):
    """Return per-sample match keys and unique clause key list."""
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
    """Return wins, losses, and winner key per sample."""
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
        winner = best_key(keys)
        w_col = key2col[winner]
        winner_idx[s] = w_col
        for k in keys:
            (wins if k == winner else loss)[key2col[k]] += 1

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

    # build match matrix M (bool)
    M = np.zeros((n_samples, n_clauses), bool)
    coverage = np.zeros(n_clauses, int)
    match_counts = np.zeros(n_samples, int)
    for s, keys in enumerate(match_per_sample):
        match_counts[s] = len(keys)
        for k in keys:
            j = key2col[k]
            M[s, j] = True
            coverage[j] += 1

    # coverage / overlap stats
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

    # tie-breaker win/loss analysis
    win_rate = np.divide(
        wins, coverage, out=np.zeros_like(wins, float), where=coverage > 0
    )
    ignored_ratio = np.divide(
        loss, coverage, out=np.zeros_like(loss, float), where=coverage > 0
    )

    logging.info("\ntop clauses by tie-breaker looses")
    for idx in np.argsort(-loss):
        cls, cid = clause_keys[idx]
        acc = perf_by_key[(cls, cid)]["accuracy"]
        logging.info(
            "cl %d/term %d | loss %4d (%.1f%% of %3d) | wins %4d | acc %.3f",
            cls,
            cid,
            loss[idx],
            ignored_ratio[idx] * 100,
            coverage[idx],
            wins[idx],
            acc,
        )

    logging.info("\ntop clauses by tie-breaker wins")
    for idx in np.argsort(-wins):
        cls, cid = clause_keys[idx]
        acc = perf_by_key[(cls, cid)]["accuracy"]
        logging.info(
            "cl %d/term %d | wins %4d | loss %4d | acc %.3f",
            cls,
            cid,
            wins[idx],
            loss[idx],
            acc,
        )

    logging.info("\ntop clauses by tie-breaker wins (least wins)")
    for idx in np.argsort(wins):
        cls, cid = clause_keys[idx]
        acc = perf_by_key[(cls, cid)]["accuracy"]
        logging.info(
            "cl %d/term %d | wins %4d | loss %4d | acc %.3f",
            cls,
            cid,
            wins[idx],
            loss[idx],
            acc,
        )

    # collision type breakdown
    intra_res = inter_res = 0
    for keys, w in zip(match_per_sample, winner_key_per_sample):
        if len(keys) <= 1 or w is None:
            continue
        other_cls = {k[0] for k in keys} - {w[0]}
        if other_cls:
            inter_res += 1
        else:
            intra_res += 1
    tot = intra_res + inter_res
    if tot:
        logging.info(
            "\nRule collisions: %.2f%% intra-class | %.2f%% inter-class",
            intra_res / tot * 100,
            inter_res / tot * 100,
        )


# ------------- reporting utils ------------
def loser_winner_pairs(
    keys: Set[ClauseKey],
    match_per_sample: List[List[ClauseKey]],
    winner_key_per_sample: List[Optional[ClauseKey]],
):
    """Return one (loser, winner) pair per loser that lost at least once."""
    out: list[tuple[ClauseKey, ClauseKey]] = []
    for k in keys:
        for s, mks in enumerate(match_per_sample):
            w = winner_key_per_sample[s]
            if k in mks and w and w != k:
                out.append((k, w))
                break
    return out


# def print_removed_detail(
#     remove: Set[ClauseKey],
#     clause_keys: List[ClauseKey],
#     match_per_sample: List[List[ClauseKey]],
#     winner_key_per_sample: List[Optional[ClauseKey]],
#     perf_by_key: Dict[ClauseKey, PerfDict],
# ):
#     """Verbose per-clause outcome after pruning."""
#     for key in sorted(remove):
#         cls, cid = key
#         acc_loser = perf_by_key[key]["accuracy"]
#         samples = [i for i, m in enumerate(match_per_sample) if key in m]
#         winners = [
#             winner_key_per_sample[i]
#             for i in samples
#             if winner_key_per_sample[i] and winner_key_per_sample[i] != key
#         ]
#         if not winners:
#             print(f"cl {cls}/term {cid} -> (no_tie_loss) | acc {acc_loser:.3f}")
#             continue
#         win_key, _ = Counter(winners).most_common(1)[0]
#         acc_w = perf_by_key[win_key]["accuracy"]
#         print(
#             f"cl {cls}/term {cid} -> cl {win_key[0]}/term {win_key[1]} | Δacc {acc_w - acc_loser:+.3f} | "
#             f"same_cls {win_key[0] == cls}"
#         )


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # fit CFIRE
    X_val, X_test, n_dim, n_classes = load_data()
    model = build_model(n_dim, n_classes)
    cfire = fit_cfire(model, X_val)

    y_val = model.predict_batch(X_val).numpy()
    y_test = model.predict_batch(X_test).numpy()
    base_val_acc = (predict_fast(cfire, X_val) == y_val).mean()
    base_test_acc = (predict_fast(cfire, X_test) == y_test).mean()
    logging.info(
        "\nORIGINAL CFIRE acc vs model → val %.3f | test %.3f",
        base_val_acc,
        base_test_acc,
    )
    pprint_dnf_rules(cfire.dnf.rules)

    # get statistics
    clause_keys, match_per_sample = collect_match_info(cfire, X_val)
    perf_by_key = build_perf_by_key(cfire)
    wins, loss, winner_key_per_sample = compute_winners(
        match_per_sample, clause_keys, perf_by_key
    )
    extra_stats(
        clause_keys,
        match_per_sample,
        wins,
        loss,
        winner_key_per_sample,
        perf_by_key,
    )

    # perform pruning: drop clauses with less wins than the threshold
    original_rules = copy.deepcopy(cfire.dnf.rules)
    total_rules = sum(len(r) for r in original_rules)

    logging.info("\nPruning evaluation (absolute wins threshold)")
    logging.info(
        f"{'th':<3} {'kept/r':<7} {'val_acc (Δ,%)':<22}"
        f" {'test_acc (Δ,%)':<24} {'same class%':<10}"
    )
    logging.info("-" * 70)

    for thr in PRUNE_WINS_THRESHOLDS:
        remove: Set[ClauseKey] = {
            clause_keys[i] for i, w in enumerate(wins) if w <= thr
        }
        new_rules = prune_rules(original_rules, remove)

        saved = cfire.dnf.rules
        cfire.dnf.rules = new_rules
        val_acc = (predict_fast(cfire, X_val) == y_val).mean()
        test_acc = (predict_fast(cfire, X_test) == y_test).mean()
        cfire.dnf.rules = saved

        drop_val_pct = (val_acc - base_val_acc) / base_val_acc * 100
        drop_test_pct = (test_acc - base_test_acc) / base_test_acc * 100

        pairs = loser_winner_pairs(remove, match_per_sample, winner_key_per_sample)
        same_ratio = (
            (sum(1 for l, w in pairs if l[0] == w[0]) / len(pairs) * 100)
            if pairs
            else 0
        )

        kept_col = f"{sum(len(r) for r in new_rules)}/{total_rules}"
        val_str = f"{val_acc:.3f} ({val_acc-base_val_acc:+.3f},{drop_val_pct:6.2f}%)"
        test_str = (
            f"{test_acc:.3f} ({test_acc-base_test_acc:+.3f},{drop_test_pct:6.2f}%)"
        )
        logging.info(
            f"{thr:<3} "
            f"{kept_col:<7} "
            f"{val_str:<22} "
            f"{test_str:<24} "
            f"{same_ratio:5.1f}%"
        )

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

    match_hist = dict(Counter(match_counts))
    share_multi_match = float((match_counts >= 2).mean() * 100)
    collision_ratio = dict(
        intra=float(intra_res / tot * 100), inter=float(inter_res / tot * 100)
    )

    prune_tbl = [
        dict(
            thr=thr,
            kept=sum(
                len(r)
                for r in prune_rules(
                    original_rules,
                    {clause_keys[i] for i, w in enumerate(wins) if w <= thr},
                )
            ),
            total=total_rules,
            val_acc=float(v),
            test_acc=float(t),
        )
        for thr, v, t in zip(
            PRUNE_WINS_THRESHOLDS,  # use vals already computed
            [val_accs_here],  # store as you loop
            [test_accs_here],
        )  # idem
    ]

    summary = dict(
        clause_stats=clause_stats,
        match_hist=match_hist,
        share_multi_match=share_multi_match,
        collision_ratio=collision_ratio,
        pruning=prune_tbl,
    )

    Path("./cfire_stats").mkdir(exist_ok=True)
    with open("cfire_stats/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logging.info(
        f"Original CFIRE accuracy – validation: {base_val_acc:.3f}, test: {base_test_acc:.3f}"
    )


if __name__ == "__main__":
    main()
