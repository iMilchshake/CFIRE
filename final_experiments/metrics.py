from typing import TypeAlias, Tuple, Set, Dict, List
import numpy as np

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
                if dim not in bounds:
                    bounds[dim] = (lo, hi)
                else:
                    cur_lo, cur_hi = bounds[dim]
                    bounds[dim] = (min(cur_lo, lo), max(cur_hi, hi))
    return bounds

def _expand_clause(clause: Clause, bounds: Dict[int, Tuple[float, float]]) -> Box:
    """Blow up a sparse clause into a full‑dimensional axis‑aligned box."""
    dim_to_interval = {d: interval for d, interval in clause}
    return [dim_to_interval.get(dim, bounds[dim]) for dim in sorted(bounds)]

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
    dims = sorted(bounds)
    D = len(dims)
    boxes_by_class: list[list[Box]] = [[] for _ in range(len(rules))]
    all_boxes: list[tuple[int, Box]] = []  # (class_id, box)

    for c, class_rules in enumerate(rules):
        for clause in class_rules:
            b = _expand_clause(clause, bounds)
            boxes_by_class[c].append(b)
            all_boxes.append((c, b))

    # volumes and proposal probs
    vols = np.array([np.prod([hi - lo for (lo, hi) in b]) for _, b in all_boxes])
    probs = vols / vols.sum()

    C = len(rules)
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
    np.fill_diagonal(IoU, 1.0)
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
