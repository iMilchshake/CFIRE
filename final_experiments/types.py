from typing import NamedTuple, TypeAlias, Tuple

from torch.utils.data import DataLoader


Literal:    TypeAlias = Tuple[int, Tuple[float, float]]     # (dimension, (low, high)) interval test
Clause:     TypeAlias = list[Literal]                       # Conjunction (AND) of literals
ClassRules: TypeAlias = list[Clause]                        # Disjunction (OR) of clauses for one class label
Rules:      TypeAlias = list[ClassRules]                    # List of ClassRules, one entry per class in the data set


class CFIREDataset(NamedTuple):
    """NamedTuple wrapper for cfire dataset tuple"""
    train_loader: DataLoader
    test_loader: DataLoader
    val_loader: DataLoader
    n_dim: int
    n_classes: int


