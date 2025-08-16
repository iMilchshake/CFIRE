import logging
from typing import TypeAlias, Tuple, Set, Dict

import numpy as np
from sklearn.metrics import pairwise_distances

from cfire.cfire_module import ItemsetNodeCollection

log = logging.getLogger(__name__)

Literal:    TypeAlias = Tuple[int, Tuple[float, float]]     # (dimension, (low, high)) interval test
Clause:     TypeAlias = list[Literal]                       # Conjunction (AND) of literals
ClassRules: TypeAlias = list[Clause]                        # Disjunction (OR) of clauses for one class label
Rules:      TypeAlias = list[ClassRules]                    # List of ClassRules, one entry per class in the data set

Box: TypeAlias = list[Tuple[float, float]]

def get_rule_size(rules: Rules) -> int:
    return sum(len(class_rules) for class_rules in rules)

def get_literal_count(rules: Rules)-> int:
    return sum(
        len(conjunction)
        for class_rule in rules
        for clause in class_rule
        for conjunction in clause
    )

def get_unique_literal_count(rules: Rules) -> int:
    unique_literals: Set[Literal] = set()
    for class_rules in rules:
        for clause in class_rules:
            unique_literals.update(clause)
    return len(unique_literals)

def _global_bounds(rules: Rules) -> Dict[int, Tuple[float, float]]:
    """Infer per‑dimension min/max across all Literals."""
    bounds: Dict[int, Tuple[float, float]] = {}
    for class_rules in rules:
        for clause in class_rules:
            for dim, (lo, hi) in clause:
                # skip dummy/invalid dims
                if dim < 0 or not np.isfinite(lo) or not np.isfinite(hi):
                    continue
                if hi < lo:  # maybe high/low could be flipped?
                    lo, hi = hi, lo
                if dim not in bounds:
                    bounds[dim] = (lo, hi)
                else:
                    cur_lo, cur_hi = bounds[dim]
                    bounds[dim] = (min(cur_lo, lo), max(cur_hi, hi))
    return bounds

def _sanitize_clause(clause: Clause, bounds: Dict[int, Tuple[float, float]]) -> Clause:
    """
    Keep only literals with real dims and finite endpoints; fix flipped (hi < lo).
    Return the cleaned clause. If it comes back empty, treat it as invalid/dummy.
    """
    cleaned: Clause = []
    for dim, (lo, hi) in clause:
        if dim not in bounds:
            continue
        if not (np.isfinite(lo) and np.isfinite(hi)):
            continue
        if hi < lo:
            lo, hi = hi, lo
        cleaned.append((dim, (float(lo), float(hi))))
    return cleaned


def _expand_clause(clause: Clause, bounds: Dict[int, Tuple[float, float]]) -> Box:
    """Blow up a sparse clause into a full‑dimensional axis‑aligned box."""
    dim_to_interval = {d: interval for d, interval in clause}
    return [dim_to_interval.get(dim, bounds[dim]) for dim in sorted(bounds)]


def _clause_has_valid_dim(clause: Clause, bounds: Dict[int, Tuple[float, float]]) -> bool:
    """Sanity method to check if a clause has at least one valid dimension."""
    return any(dim in bounds and np.isfinite(lo) and np.isfinite(hi) for dim, (lo,hi) in clause)


def get_class_iou_matrix_mc(
        rules: Rules,
        n_samples: int = 500_000,
        batch_size: int = 50_000,
        seed: int | None = 0,
) -> list[list[float]]:
    """sample points uniformly inside directly inside boxes (as opposed to uniform in the gloabl grid) defined by the
     rules and compute IoU matrix for classes."""
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    # build boxes per class and flat across all boxes
    bounds = _global_bounds(rules)

    if not bounds:
        # nothing to measure
        log.warning("[IoU/MC] Empty global bounds → returning all-zeros IoU.")
        C = len(rules)
        return np.zeros((C, C), dtype=float).tolist()


    dims = sorted(bounds)
    D = len(dims)
    boxes_by_class: list[list[Box]] = [[] for _ in range(len(rules))]
    all_boxes: list[tuple[int, Box]] = []  # (class_id, box)

    # for logging/debugging
    kept = np.zeros(len(rules), dtype=int)
    skipped_invalid = np.zeros(len(rules), dtype=int)
    skipped_zero = np.zeros(len(rules), dtype=int)

    for c, class_rules in enumerate(rules):
        for clause in class_rules:
            cleaned = _sanitize_clause(clause, bounds)
            if not cleaned: # would expand to whole hull, so skip
                skipped_invalid[c] += 1
                continue
            b = _expand_clause(cleaned, bounds)

            widths = np.array([hi - lo for (lo, hi) in b], dtype=float)
            if not np.isfinite(widths).all():
                skipped_invalid[c] += 1
                continue
            if np.any(widths < 0):
                skipped_invalid[c] += 1
                continue # shouldn’t happen after swap; guard anyway
            vol = float(np.prod(np.maximum(widths, 0.0)))
            if vol <= 0.0:
                skipped_zero[c] += 1
                continue

            boxes_by_class[c].append(b)
            all_boxes.append((c, b))
            kept[c] += 1


    C = len(rules)
    if not all_boxes:
        log.warning("[IoU/MC] No valid boxes after cleaning (kept=%s, invalid=%s, zero=%s) → returning all-zeros IoU.",
                    kept.tolist(), skipped_invalid.tolist(), skipped_zero.tolist())
        return np.zeros((C, C), dtype=float).tolist()

    for c in range(C):
        if skipped_invalid[c] or skipped_zero[c]:
            log.info("[IoU/MC] class %d: kept=%d, skipped_invalid=%d, skipped_zero=%d",
                     c, kept[c], skipped_invalid[c], skipped_zero[c])


    vols = np.array([np.prod([hi - lo for (lo, hi) in b], dtype=float) for _, b in all_boxes], dtype=float)
    # sanity clippings so that we cant have NaNs or negative volumes
    vols[~np.isfinite(vols)] = 0.0
    vols[vols < 0] = 0.0
    total = float(vols.sum())
    if np.isfinite(total) and total > 0.0:
        probs = vols / total
    else:
        log.warning("This should never happen. [IoU/MC] Volume sum invalid (total=%g) — falling back to uniform over %d boxes", total, len(all_boxes))
        probs = np.full(len(all_boxes), 1.0 / len(all_boxes), dtype=float)


    n_w = np.zeros(C, dtype=np.float64)
    I_w = np.zeros((C, C), dtype=np.float64)

    drawn = 0
    while drawn < n_samples:
        n_this = min(batch_size, n_samples - drawn)

        idx = rng.choice(len(all_boxes), size=n_this, p=probs)
        chosen = [all_boxes[i][1] for i in idx]

        # sample uniformly inside each chosen box
        lo = np.stack([[l for (l, h) in b] for b in chosen], axis=0)
        hi = np.stack([[h for (l, h) in b] for b in chosen], axis=0)
        X = lo + rng.random(size=(n_this, D)) * (hi - lo)

        # membership vs every box (for weights) and vs every class (for IoU)
        # weight denominator m_all(x): number of boxes containing x (across all classes)
        m_all = np.zeros(n_this, dtype=np.int32)
        S_batch = np.zeros((n_this, C), dtype=bool)

        # count overlaps and build per-class membership
        for c, boxes in enumerate(boxes_by_class):
            in_any_c = np.zeros(n_this, dtype=bool)
            for b in boxes:
                blo = np.array([l for (l, h) in b])
                bhi = np.array([h for (l, h) in b])
                in_b = np.all((X >= blo) & (X <= bhi), axis=1)
                in_any_c |= in_b
                m_all += in_b.astype(np.int32)
            S_batch[:, c] = in_any_c

        # weights
        w = np.where(m_all > 0, 1.0 / m_all, 0.0)

        # accumulate weighted counts
        n_w += (w[:, None] * S_batch).sum(axis=0)
        I_w += (w[:, None, None] * (S_batch[:, :, None] & S_batch[:, None, :])).sum(axis=0)

        drawn += n_this

    U_w = n_w[:, None] + n_w[None, :] - I_w
    IoU = np.divide(I_w, U_w, out=np.zeros_like(I_w), where=U_w > 0)


    # Diagonal consistency, union and intersection with itself should be equal to the number of samples in each class
    if not np.allclose(np.diag(I_w), n_w):
        log.warning("[IoU/MC] diag(I) != n (got %s vs %s).", np.diag(I_w).tolist(), n_w.tolist())
    if not np.allclose(np.diag(U_w), n_w):
        log.warning("[IoU/MC] diag(U) != n (got %s vs %s).", np.diag(U_w).tolist(), n_w.tolist())


    for c in range(C):
        IoU[c, c] = 1.0 if n_w[c] > 0 else 0.0

    if (IoU < -1e-12).any() or (IoU > 1 + 1e-12).any():
        log.warning("[IoU/MC] IoU outside [0,1] before clipping. Clipping to valid range.")

    IoU = np.clip(IoU, 0.0, 1.0)
    if not np.allclose(IoU, IoU.T, atol=1e-8):
        log.warning("[IoU/MC] IoU not symmetric (max |A-A^T| = %.3e).", float(np.max(np.abs(IoU - IoU.T))))

    return IoU.tolist()

# actual metrics
def mean_offdiag_iou(iou: list[list[float]]) -> float:
    arr = np.asarray(iou)
    n = arr.shape[0]
    return arr[np.triu_indices(n, k=1)].mean() if n > 1 else 0.0


def max_offdiag_iou(iou: list[list[float]]) -> float:
    arr = np.asarray(iou)
    n = arr.shape[0]
    return arr[np.triu_indices(n, k=1)].max() if n > 1 else 0.0

## possible DNF -> minimized normal form
# problem with probably all approaches: normally those are boolean, so we would apply on each class predictor. but we would also need to carry accuracy of the boxes from CFIRE and work them into this step so we can tie-break later one with accuracy of boxes
# --> therefore we can probably use these minimizations for metrics, not for prediction

# Quine MCluskey algorithm https://en.wikipedia.org/wiki/Quine%E2%80%93McCluskey_algorithm
    # problem NP-complete

# Espresso heuristic logic minimizer https://en.wikipedia.org/wiki/Espresso_heuristic_logic_minimizer
    # works with booleans, so we would need to convert the rules to boolean form
    # problem: has a reduce step that might throw away overlapping/competing rules, that we would normally tie-break via accuracy rule.


def build_coverage_matrices(frequent_nodes: list[ItemsetNodeCollection]) -> list[np.ndarray]:
    """
    coverage_mats[c][i, j] == True <=> sample i is in support of node j of class c.
    """
    coverage_mats: list[np.ndarray] = []
    for col in frequent_nodes:
        supports = col.class_support
        if not supports:
            coverage_mats.append(np.empty((0, 0), dtype=bool))
            continue
        sample_universe = {i for s in supports for i in s}
        n_samples = (max(sample_universe) + 1) if sample_universe else 0
        assert n_samples > 0, "Class has nodes but no samples"
        mat = np.zeros((n_samples, len(supports)), dtype=bool)
        for j, s in enumerate(supports):
            if s:
                mat[list(s), j] = True
        coverage_mats.append(mat)
    return coverage_mats


def mean_coverage_ratio(coverage_mats: list[np.ndarray]) -> float:
    """
    Average (over classes with nodes) of the fraction of samples covered by at least 1 node.
    """
    ratios = []
    for cov_mat in coverage_mats:
        n_samples, n_nodes = cov_mat.shape
        if n_nodes == 0:
            continue
        ratios.append(float(cov_mat.any(axis=1).mean()))
    return float(np.mean(ratios)) if ratios else 0.0


def mean_single_coverage_ratio(coverage_mats: list[np.ndarray]) -> float:
    """
    Average (over classes with nodes) of the fraction of samples covered by exactly 1 node.
    """
    ratios = []
    for cov_mat in coverage_mats:
        n_samples, n_nodes = cov_mat.shape
        if n_nodes == 0:
            continue
        depth_per_sample = cov_mat.sum(axis=1)
        ratios.append(float((depth_per_sample == 1).mean()))
    return float(np.mean(ratios)) if ratios else 0.0


def mean_nodes_per_sample(coverage_mats: list[np.ndarray]) -> float:
    """
    Average (over classes with nodes) of the mean number of covering nodes per sample.
    """
    depths = []
    for cov_mat in coverage_mats:
        n_samples, n_nodes = cov_mat.shape
        if n_nodes == 0:
            continue
        depths.append(float(cov_mat.sum(axis=1).mean()))
    return float(np.mean(depths)) if depths else 0.0


def mean_duplicate_nodes_ratio(coverage_mats: list[np.ndarray]) -> float:
    """
    Average (over classes with nodes) of the fraction of duplicate nodes.
    """
    def count_unique_columns(mat: np.ndarray) -> int:
        seen = {mat[:, j].tobytes() for j in range(mat.shape[1])}
        return len(seen)

    ratios = []
    for cov_mat in coverage_mats:
        n_samples, n_nodes = cov_mat.shape
        if n_nodes == 0:
            continue
        n_unique = count_unique_columns(cov_mat)
        ratios.append(1.0 - (n_unique / n_nodes))
    return float(np.mean(ratios)) if ratios else 0.0

# --- Metrics on normalized attributions ---

def normalize_explanations(a: np.ndarray) -> np.ndarray:
    e = a.astype(float, copy=True)
    max_abs = np.max(np.abs(e), axis=1, keepdims=True)   # (n_samples, 1)
    max_abs[max_abs == 0.0] = 1.0                        # avoid div-by-zero
    e /= max_abs                                         # broadcast divide
    np.maximum(e, 0.0, out=e)                            # clamp negatives
    return e

def mean_absolute_attribution(e: np.ndarray) -> float:
    return float(np.mean(e))

def attribution_variance(e: np.ndarray) -> float:
    return float(np.var(e))

def sparsity(e: np.ndarray, eps: float = 1e-6) -> float:
    return float((e <= eps).mean())

def class_separation_in_attribution_space(e: np.ndarray, y: np.ndarray, metric: str = "cosine") -> float:
    classes = np.unique(y)
    if classes.size <= 1:
        return 0.0
    centroids = [e[y == c].mean(axis=0) for c in classes if np.any(y == c)]
    if len(centroids) <= 1:
        return 0.0
    C = np.stack(centroids, axis=0)
    D = pairwise_distances(C, metric=("euclidean" if metric == "euclidean" else "cosine"))
    iu = np.triu_indices(D.shape[0], k=1)
    return float(D[iu].mean()) if iu[0].size else 0.0


# --- Metrics on binarized masks ---

def mean_active_features_per_sample(binarized: np.ndarray) -> float:
    return float(binarized.sum(axis=1).mean())

def mean_active_features_ratio(binarized: np.ndarray) -> float:
    return float((binarized.sum(axis=1) / binarized.shape[1]).mean())

def mean_feature_activation_ratio(binarized: np.ndarray) -> float:
    return float(binarized.mean(axis=0).mean())

def features_inactive_ratio(binarized: np.ndarray) -> float:
    return float((~binarized.any(axis=0)).mean())

def all_features_active_ratio(binarized: np.ndarray) -> float:
    return float(np.all(binarized, axis=1).mean())

def all_features_inactive_ratio(binarized: np.ndarray) -> float:
    return float((~np.any(binarized, axis=1)).mean())

def mean_feature_class_specificity(m: np.ndarray, y: np.ndarray) -> float:
    """
    For each feature j: max_c P(m_ij=1 | y_i=c). Then average over features.
    """
    if m.ndim != 2 or m.size == 0:
        return 0.0
    n, d = m.shape
    classes, y_idx = np.unique(y, return_inverse=True)
    k = len(classes)
    if k == 0:
        return 0.0
    class_counts = np.bincount(y_idx, minlength=k).astype(float)  # (k,)
    class_counts[class_counts == 0.0] = np.nan
    A = np.zeros((k, d), dtype=np.float64)
    for c in range(k):
        A[c] = m[y_idx == c].sum(axis=0)
    P = A / class_counts[:, None]  # (k, d)
    per_feature = np.nanmax(P, axis=0)
    per_feature = np.nan_to_num(per_feature, nan=0.0)
    return float(per_feature.mean())

def _mean_jaccard_within_class(m: np.ndarray) -> float:
    if m.shape[0] <= 1 or m.shape[1] == 0:
        return 0.0
    # drop all-zero rows (Jaccard undefined if both rows are zero; we exclude such pairs)
    mask = m.any(axis=1)
    m = m[mask]
    if m.shape[0] <= 1:
        return 0.0
    Dj = pairwise_distances(m, metric="jaccard")
    iu = np.triu_indices(Dj.shape[0], k=1)
    return float((1.0 - Dj[iu]).mean()) if iu[0].size else 0.0

def mean_within_class_jaccard(m: np.ndarray, y: np.ndarray) -> float:
    vals = []
    for c in np.unique(y):
        Mc = m[y == c]
        if Mc.shape[0] >= 2:
            vals.append(_mean_jaccard_within_class(Mc))
    return float(np.mean(vals)) if vals else 0.0

def mean_across_class_jaccard(m: np.ndarray, y: np.ndarray) -> float:
    classes = np.unique(y)
    sims_sum, count = 0.0, 0
    for i, ci in enumerate(classes):
        Mi = m[y == ci]
        Mi = Mi[Mi.any(axis=1)]
        if Mi.size == 0:
            continue
        for cj in classes[i+1:]:
            Mj = m[y == cj]
            Mj = Mj[Mj.any(axis=1)]
            if Mj.size == 0:
                continue
            Dj = pairwise_distances(Mi, Mj, metric="jaccard")
            sim = 1.0 - Dj
            sims_sum += float(sim.sum())
            count += sim.size
    return float(sims_sum / count) if count > 0 else 0.0

def class_separation_score(m: np.ndarray, y: np.ndarray) -> float:
    return float(mean_within_class_jaccard(m, y) - mean_across_class_jaccard(m, y))
