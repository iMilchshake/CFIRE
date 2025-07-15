from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Union
from cfire.cfire_module import DNFClassifier


import numpy as np

Literal = Tuple[int, Tuple[float, float]]
Clause = List[Literal]
Rules = List[List[Clause]]


# ---------------------------------------------------------------------------
# Data‑domain helper
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class DomainDescriptor:
    """Holds finite per‑feature bounds *after* optional preprocessing.

    Attributes
    ----------
    bounds : np.ndarray of shape (d, 2)
        Each row ``[lo, hi]`` is the *closed* interval for that feature.
    """

    bounds: np.ndarray  # shape (d, 2)

    def __post_init__(self):  # type: ignore[override]
        b = self.bounds
        if b.ndim != 2 or b.shape[1] != 2:
            raise ValueError("`bounds` must be a (d, 2) array of [lo, hi] pairs.")
        if not np.all(b[:, 0] < b[:, 1]):
            raise ValueError("Lower bounds must be strictly < upper bounds for each feature.")

    @property
    def n_features(self) -> int:  # noqa: D401 – simple property
        """Total number of features (length of the domain vector)."""
        return self.bounds.shape[0]

    #fast broadcasting in volume calculations
    @property
    def lo(self) -> np.ndarray:
        return self.bounds[:, 0]

    @property
    def hi(self) -> np.ndarray:
        return self.bounds[:, 1]


# ---------------------------------------------------------------------------
# Public helper to compute bounds from data
# ---------------------------------------------------------------------------
def compute_domain_bounds(X: np.ndarray) -> DomainDescriptor:
    """Return *closed* min/max bounds for each column of X."""
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError("X must be 2‑D [n_samples, n_features].")

    if not np.issubdtype(X.dtype, np.number):
        raise TypeError("X must be numeric")

    lo = np.min(X, axis=0)
    hi = np.max(X, axis=0)
    return DomainDescriptor(np.stack([lo, hi], axis=1))


# ---------------------------------------------------------------------------
# Private geometry utilities
# ---------------------------------------------------------------------------

def _normalize_clause(
        clause: Clause, *, n_features: int, domain: DomainDescriptor
) -> np.ndarray:
    """Return a (d, 2) array [lo, hi] for this clause.

    Missing dimensions are filled with the corresponding domain bounds.
    """
    box = domain.bounds.copy()
    for feat_id, (lo, hi) in clause:
        if not 0 <= feat_id < n_features:
            raise IndexError(f"Feature id {feat_id} out of range (0..{n_features - 1})")
        box[feat_id, 0] = lo
        box[feat_id, 1] = hi

    return box


def _volume(box: np.ndarray) -> float:
    edges = box[:, 1] - box[:, 0]
    if np.any(edges <= 0):
        return 0.0
    return float(np.prod(edges))


def _intersection_volume(box_a: np.ndarray, box_b: np.ndarray) -> float:
    low = np.maximum(box_a[:, 0], box_b[:, 0])
    high = np.minimum(box_a[:, 1], box_b[:, 1])
    if np.any(low >= high):
        return 0.0
    return float(np.prod(high - low))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_box_overlap_matrix(
        classifier: "DNFClassifier",  # type: ignore[name‑defined]
        domain: DomainDescriptor,
        *,
        return_boxes: bool = False,
):
    rules: Rules = classifier.rules  # type: ignore[attr‑defined]
    d = domain.n_features

    flat_boxes: List[np.ndarray] = []
    class_ids: List[int] = []
    for cls_id, clauses in enumerate(rules):
        for clause in clauses:
            flat_boxes.append(_normalize_clause(clause, n_features=d, domain=domain))
            class_ids.append(cls_id)

    n_boxes = len(flat_boxes)
    overlaps = np.zeros((n_boxes, n_boxes), dtype=float)

    for i in range(n_boxes):
        box_i = flat_boxes[i]
        for j in range(i, n_boxes):
            vol_ij = _intersection_volume(box_i, flat_boxes[j])
            overlaps[i, j] = overlaps[j, i] = vol_ij

    if return_boxes:
        return overlaps, flat_boxes, class_ids  # type: ignore[return‑value]
    return overlaps


def compute_overlap_matrix(
        classifier: "DNFClassifier",  # type: ignore[name‑defined]
        domain: DomainDescriptor,
        *,
        metric: str = "intersection",  # or "iou"
):
    if metric not in {"intersection", "iou"}:
        raise ValueError("metric must be 'intersection' or 'iou'.")

    rules: Rules = classifier.rules
    n_classes = len(rules)

    overlaps, boxes, cls_ids = compute_box_overlap_matrix(
        classifier, domain, return_boxes=True
    )
    vol_cache_raw = np.array([_volume(b) for b in boxes])

    class_to_indices: List[List[int]] = [[] for _ in range(n_classes)]
    for idx, cls in enumerate(cls_ids):
        class_to_indices[cls].append(idx)

    agg = np.ones((n_classes, n_classes), dtype=float)
    for i in range(n_classes):
        idx_i = class_to_indices[i]
        vol_i_raw = vol_cache_raw[idx_i].sum()
        for j in range(i + 1, n_classes):
            idx_j = class_to_indices[j]
            vol_j_raw = vol_cache_raw[idx_j].sum()

            inter_raw = overlaps[np.ix_(idx_i, idx_j)].sum()

            if metric == "intersection":
                agg[i, j] = agg[j, i] = inter_raw
            else:  # IoU
                union_raw = vol_i_raw + vol_j_raw - inter_raw
                if union_raw <= 0:
                    agg[i, j] = agg[j, i] = 0.0
                else:
                    agg[i, j] = agg[j, i] = inter_raw / union_raw
    return agg

def print_class_overlap(iou: np.ndarray):
    """Pretty-print symmetric class-wise overlap matrix (IoU-style)."""
    n = iou.shape[0]
    col_labels = [f"C{j}" for j in range(n)]
    print("\nClass-wise IoU:")
    header = "          " + "  ".join(f"{label:>6}" for label in col_labels)
    print(header)
    for i in range(n):
        row_vals = "  ".join(f"{iou[i, j]:6.3f}" for j in range(n))
        print(f"Class {i:<2}  {row_vals}")

# ---------------------------------------------------------------------------
#  NEW: per-class root-normalised “length volume”  ---------------------------
# ---------------------------------------------------------------------------

def compute_class_root_volumes(
        classifier: "DNFClassifier",   # type: ignore[name-defined]
        domain: DomainDescriptor,
) -> Tuple[np.ndarray, float, float]:
    """
    Return:

    * ``class_root_vols`` – 1-D array, length = n_classes, each entry is the
      **sum of d_eff-root volumes** of that class’s boxes, where d_eff is the
      number of dimensions actually constrained in the clause.
    * ``total_root_vol``      – sum over all classes
    * ``domain_root_volume``  – n_features-root of the full domain hyper-volume

    """
    rules: Rules = classifier.rules  # type: ignore[attr-defined]
    n_classes = len(rules)
    class_root_vols = np.zeros(n_classes, dtype=float)

    for cls_id, clauses in enumerate(rules):
        acc = 0.0
        for clause in clauses:
            edges = [hi - lo for _, (lo, hi) in clause]
            if any(e <= 0 for e in edges):
                continue  # skip degenerate slice
            vol = float(np.prod(edges))
            d_eff = len(edges)
            acc += vol ** (1.0 / d_eff) # normalize with root here for normalization of different dimentiosns
        class_root_vols[cls_id] = acc

    total_root_vol = class_root_vols.sum()

    # full-domain root-volume
    domain_edges = domain.hi - domain.lo
    domain_volume = float(np.prod(domain_edges))
    d_full = domain.n_features
    domain_root_volume = domain_volume ** (1.0 / d_full)

    return class_root_vols, total_root_vol, domain_root_volume




__all__ = [
    "DomainDescriptor",
    "compute_domain_bounds",
    "compute_box_overlap_matrix",
    "compute_overlap_matrix",
    "print_class_overlap"
]


if __name__ == "__main__":
    class DummyDNF:
        def __init__(self, rules):
            self.rules = rules

    domain = DomainDescriptor(np.array([
        [-1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 1.0],
    ]))

    rules = [
        [[(0, (-0.5, 0.5)), (1, (-0.5, 0.5))]],
        [[(0, (0.0, 0.8)), (1, (-0.2, 0.2))]],
        [[(2, (-0.2, 0.2)), (1, (-0.6, -0.3))]],
    ]

    clf = DummyDNF(rules)

    print("Box overlap matrix:")
    overlaps, boxes, cls_ids = compute_box_overlap_matrix(clf, domain, return_boxes=True)
    print(overlaps)

    print("\nClass-wise raw volume overlap:")
    print(compute_overlap_matrix(clf, domain, metric="intersection"))

    print("\nClass-wise IoU:")
    print(compute_overlap_matrix(clf, domain, metric="iou"))