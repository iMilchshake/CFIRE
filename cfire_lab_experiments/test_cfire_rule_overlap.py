# test_cfire_rule_overlap.py
# run `test_train.py` beforehand!

from pathlib import Path
from typing import Dict, Tuple, Any, List, Sequence
import copy
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


# ------------------------------------------------------------------ #
# helpers
# ------------------------------------------------------------------ #
def canon(rule):
    """tuple-ify rules (CFIRE may store lists – unhashable)."""
    return tuple(rule) if isinstance(rule, list) else rule


def build_perf_by_key(cf: CFIRE) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """(cls,clause) → metrics dict."""
    out = {}
    for cls, rule_list in enumerate(cf.dnf.rules):
        for cid, rule in enumerate(rule_list):
            out[(cls, cid)] = cf.dnf.rule_performances[cls][canon(rule)]
    return out


def predict_fast(cf: CFIRE, X) -> np.ndarray:
    """CFIRE prediction w/o explanations (numpy or torch ok)."""
    return cf(X if isinstance(X, torch.Tensor) else torch.as_tensor(X))


def prune_rules(
    rule_tree: Sequence[Sequence], to_remove: set[Tuple[int, int]]
) -> List[List]:
    """Return deep-copied rule list with (cls,cid) removed."""
    new_tree = []
    for cls, rl in enumerate(rule_tree):
        new_tree.append(
            [rule for cid, rule in enumerate(rl) if (cls, cid) not in to_remove]
        )
    return new_tree


# ------------------------------------------------------------------ #


def main():
    # ----------------------------- data / model -----------------------------
    loaders = datasets.get_abalone()
    train_loader, test_loader, val_loader, n_dim, n_classes = loaders
    X_val, _ = loader_to_tensor(val_loader)
    X_test, _ = loader_to_tensor(test_loader)

    model = make_ff([n_dim, 128, 128, n_classes], torch.nn.ReLU).to("cpu")
    restore_checkpoint(Path("./models/tmp.ckpt"), model, train=False)

    # ----------------------------- CFIRE fit -----------------------------
    def expl_bin(x):
        # return __preprocess_explanations_ext(x, top_k=2) > 0
        return __preprocess_explanations_ext(x, threshold=0.01) > 0

    with RandomSeed(42):
        cfire = CFIRE(
            localexplainer_fn=ks_fn_cached(Path("./models/explanations.pt")),
            inference_fn=model.predict_batch_softmax,
            expl_binarization_fn=expl_bin,
        )
        cfire.fit(X_val.numpy(), model.predict_batch(X_val).numpy())

    # ---------- baseline accuracy (val & test) ----------------------------
    y_val_model = model.predict_batch(X_val).numpy()
    y_test_model = model.predict_batch(X_test).numpy()
    base_val_acc = (predict_fast(cfire, X_val) == y_val_model).mean()
    base_test_acc = (predict_fast(cfire, X_test) == y_test_model).mean()
    print(
        f"\nORIGINAL CFIRE acc vs model → val {base_val_acc:.3f} | "
        f"test {base_test_acc:.3f}"
    )
    pprint_dnf_rules(cfire.dnf.rules)

    # ------------------------------------------------------------------ #
    # full explanation pass on validation set
    # ------------------------------------------------------------------ #
    cfire_out = cfire(X_val, explain=True)

    clause_keys: List[Tuple[int, int]] = []
    for _, matches in cfire_out:
        for key, _ in matches:
            if key not in clause_keys:
                clause_keys.append(key)

    key2col = {k: i for i, k in enumerate(clause_keys)}
    n_clauses = len(clause_keys)
    n_samples = len(cfire_out)

    M = np.zeros((n_samples, n_clauses), bool)
    for s, (_, matches) in enumerate(cfire_out):
        for key, _ in matches:
            M[s, key2col[key]] = True

    coverage = M.sum(axis=0)
    match_counts = M.sum(axis=1)
    overlap_matrix = M.astype(int).T @ M.astype(int)
    class_of = np.array([cls for cls, _ in clause_keys])

    print(f"\nshare of samples covered by ≥2 clauses: {(match_counts>1).mean():.2%}")
    print("histogram of #-matches per sample:")
    for k in np.unique(match_counts):
        print(f"  {k}: {(match_counts==k).sum()}")

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

    perf_by_key = build_perf_by_key(cfire)

    print("\nTop 10 clauses by inter-class competition:")
    for idx in np.argsort(-inter)[:10]:
        cls, cid = clause_keys[idx]
        acc = perf_by_key[(cls, cid)]["accuracy"]
        print(
            f"cl {cls}/term {cid} | covers {coverage[idx]:4d} | "
            f"intra {intra[idx]:3d} | inter {inter[idx]:3d} | acc {acc:.3f}"
        )

    print("\nTop 10 clauses by intra-class competition:")
    for idx in np.argsort(-intra)[:10]:
        cls, cid = clause_keys[idx]
        acc = perf_by_key[(cls, cid)]["accuracy"]
        print(
            f"cl {cls}/term {cid} | covers {coverage[idx]:4d} | "
            f"intra {intra[idx]:3d} | inter {inter[idx]:3d} | acc {acc:.3f}"
        )

    pairs = [
        (overlap_matrix[j, k], j, k)
        for j in range(n_clauses)
        for k in range(j + 1, n_clauses)
        if overlap_matrix[j, k]
    ]
    pairs.sort(reverse=True)
    print("\nTop 20 clause-pair overlaps:")
    for cnt, j, k in pairs[:20]:
        same = class_of[j] == class_of[k]
        a_j = perf_by_key[clause_keys[j]]["accuracy"]
        a_k = perf_by_key[clause_keys[k]]["accuracy"]
        print(
            f"{clause_keys[j]} ↔ {clause_keys[k]} : {cnt:4d} "
            f"({'same' if same else 'diff'} class) | acc {a_j:.3f}/{a_k:.3f}"
        )

    # tie-breaker wins / losses
    def best_key(match_list):
        return max(match_list, key=lambda m: perf_by_key[m[0]]["accuracy"])[0]

    wins = np.zeros(n_clauses, int)
    loss = np.zeros(n_clauses, int)
    winner_idx = np.full(n_samples, -1, int)
    for s, (_, matches) in enumerate(cfire_out):
        if not matches:
            continue
        wkey = best_key(matches)
        wcol = key2col[wkey]
        winner_idx[s] = wcol
        for key, _ in matches:
            col = key2col[key]
            if col == wcol:
                wins[col] += 1
            else:
                loss[col] += 1

    win_rate = np.divide(
        wins, coverage, out=np.zeros_like(wins, float), where=coverage > 0
    )

    ignored_ratio = np.divide(
        loss, coverage, out=np.zeros_like(loss, float), where=coverage > 0
    )

    print("\nTop 10 clauses by ignored samples (loss after tie-break):")
    for idx in np.argsort(-loss)[:10]:
        cls, cid = clause_keys[idx]
        acc = perf_by_key[(cls, cid)]["accuracy"]
        print(
            f"cl {cls}/term {cid} | loss {loss[idx]:4d} "
            f"({ignored_ratio[idx]*100:5.1f}% of {coverage[idx]:3d}) | "
            f"wins {wins[idx]:4d} | acc {acc:.3f}"
        )

    print("\nTop 10 clauses by wins after tie-breaker:")
    for idx in np.argsort(-wins)[:10]:
        cls, cid = clause_keys[idx]
        acc = perf_by_key[(cls, cid)]["accuracy"]
        print(
            f"cl {cls}/term {cid} | wins {wins[idx]:4d} | "
            f"loss {loss[idx]:4d} | acc {acc:.3f}"
        )

    acc_arr = np.array([perf_by_key[k]["accuracy"] for k in clause_keys])
    if (coverage > 0).sum() >= 2:
        print(
            f"\nPearson corr (accuracy ↔ win-rate): "
            f"{np.corrcoef(acc_arr[coverage>0], win_rate[coverage>0])[0,1]:.3f}"
        )
    print(
        f"Pearson corr (coverage ↔ accuracy): "
        f"{np.corrcoef(coverage, acc_arr)[0,1]:.3f}"
    )

    intra_res = inter_res = 0
    for s, (_, matches) in enumerate(cfire_out):
        if len(matches) <= 1:
            continue
        w_cls = clause_keys[winner_idx[s]][0]
        other_cls = {k[0] for k, _ in matches} - {w_cls}
        if other_cls:
            inter_res += 1
        else:
            intra_res += 1
    tot = intra_res + inter_res
    if tot:
        print(
            f"\nTie-breaker resolved collisions: "
            f"{intra_res/tot:.2%} intra-class | {inter_res/tot:.2%} inter-class"
        )

    # default pruning suggestion list
    suggest = [
        i
        for i in range(n_clauses)
        if wins[i] <= 5 or (win_rate[i] < 0.10 and coverage[i] > 0)
    ]
    print("\nSuggested for pruning (≤5 wins OR win-rate <10%):")
    for i in sorted(suggest, key=lambda j: wins[j]):
        cls, cid = clause_keys[i]
        print(
            f"  cl {cls}/term {cid} | wins {wins[i]:3d} | "
            f"wr {win_rate[i]*100:5.1f}% | acc {acc_arr[i]:.3f}"
        )

    # ------------------------------------------------------------------ #
    # pruning scenarios without manual name strings
    # ------------------------------------------------------------------ #
    levels = [
        (0, 0.00),
        (1, 0.00),
        (2, 0.00),
        (3, 0.00),
        (4, 0.00),
        (5, 0.00),
        (5, 0.10),
        (5, 0.20),
    ]
    original_rules = copy.deepcopy(cfire.dnf.rules)

    print("\nPruning evaluation")
    print("cfg_id   kept/r   val_acc    Δ     drop%   test_acc    Δ     drop%")
    print("--------------------------------------------------------------------")

    total_rules = sum(len(r) for r in original_rules)

    for max_w, wr_th in levels:
        cfg_id = f"({max_w}, {wr_th})"

        remove = {
            clause_keys[i]
            for i in range(n_clauses)
            if wins[i] <= max_w or (win_rate[i] < wr_th and coverage[i] > 0)
        }

        cfire.dnf.rules = prune_rules(original_rules, remove)

        val_acc = (predict_fast(cfire, X_val) == y_val_model).mean()
        test_acc = (predict_fast(cfire, X_test) == y_test_model).mean()

        d_val, d_test = val_acc - base_val_acc, test_acc - base_test_acc
        drop_val_pct = -d_val / base_val_acc * 100
        drop_test_pct = -d_test / base_test_acc * 100
        kept = sum(len(r) for r in cfire.dnf.rules)

        # ── aligned printing ───────────────────────────────────────────
        print(
            f"{cfg_id:<8} "
            f"{kept:>3d}/{total_rules:<3d}  "
            f"{val_acc:7.3f} {d_val:+6.3f} {drop_val_pct:6.2f}%  "
            f"{test_acc:7.3f} {d_test:+6.3f} {drop_test_pct:6.2f}%"
        )

    cfire.dnf.rules = original_rules

    # -> generally ALOT of overlap (inter and intra), 99%> points have more than 1 matching bounding box???
    # 1. it would be interesting to see how much of the overlap disappears by applying tie breaker
    #       -> can rules be heavily pruned by considering this?
    #       -> can i re-arrange rules, so that overlapping regions in which one rule wins anyway,
    #       we just make the loosing rule smaller? (we'd need to re-calculate the performance metrics tho)
    # 2. maybe also check the actual overlap of rules? compare areas? (how to do that for rules of different number of dimensions?)

    # idea: just prune inter-class collisions?
    # idea: modify existing rules instead of prune?
    #    - Hmm at this point i feel like it makes more sense to modify rule generation..
    #    - one approach would be to not fully remove a clause, but just get rid of overlapping features? (easier than modifying ranges xd)


if __name__ == "__main__":
    main()
