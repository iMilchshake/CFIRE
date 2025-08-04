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


def _box_volume(box: Box) -> float:
    vol = 1.0
    for lo, hi in box:
        edge = hi - lo
        if edge <= 0:
            return 0.0
        vol *= edge
    return vol


def _intersection_volume(box_a: Box, box_b: Box) -> float:
    inter = 1.0
    for (lo_a, hi_a), (lo_b, hi_b) in zip(box_a, box_b):
        lo = max(lo_a, lo_b)
        hi = min(hi_a, hi_b)
        if hi <= lo:
            return 0.0
        inter *= (hi - lo)
    return inter


def get_class_iou_matrix(rules: Rules) -> list[list[float]]:
    """Return an N×N symmetric matrix of class‑wise IoU values."""
    n_classes = len(rules)
    bounds = _global_bounds(rules)

    boxes: List[Box] = []
    cls_ids: List[int] = []
    for cls_id, class_rules in enumerate(rules):
        for clause in class_rules:
            boxes.append(_expand_clause(clause, bounds))
            cls_ids.append(cls_id)

    n_boxes = len(boxes)
    box_volumes = [_box_volume(b) for b in boxes]

    inter_mat = [[0.0] * n_boxes for _ in range(n_boxes)]
    for i in range(n_boxes):
        for j in range(i, n_boxes):
            v = _intersection_volume(boxes[i], boxes[j])
            inter_mat[i][j] = inter_mat[j][i] = v

    vol_per_class = [0.0] * n_classes
    for v, cls in zip(box_volumes, cls_ids):
        vol_per_class[cls] += v

    cls_to_indices: List[List[int]] = [[] for _ in range(n_classes)]
    for idx, cls in enumerate(cls_ids):
        cls_to_indices[cls].append(idx)

    matrix = [[0.0] * n_classes for _ in range(n_classes)]
    for i in range(n_classes):
        idx_i = cls_to_indices[i]
        vol_i = vol_per_class[i]
        for j in range(i + 1, n_classes):
            idx_j = cls_to_indices[j]
            vol_j = vol_per_class[j]
            inter = sum(inter_mat[a][b] for a in idx_i for b in idx_j)
            union = vol_i + vol_j - inter
            matrix[i][j] = matrix[j][i] = inter / union if union > 0 else 0.0

    return matrix


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
