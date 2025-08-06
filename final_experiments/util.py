from typing import Tuple, Dict

import torch

from final_experiments.types import Rules


def loader_to_tensor(loader):
    """collect all batches of dataloader into one tensor"""
    xs, ys = [], []
    for xb, yb in loader:
        xs.append(xb)
        ys.append(yb)
    X = torch.cat(xs)
    y = torch.cat(ys)
    return X, y


def global_bounds(rules: Rules) -> Dict[int, Tuple[float, float]]:
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
