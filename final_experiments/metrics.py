import logging
from typing import TypeAlias, Tuple, Set, Dict, List
import numpy as np

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
