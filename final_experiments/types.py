from typing import NamedTuple

from torch.utils.data import DataLoader


class CFIREDataset(NamedTuple):
    """NamedTuple wrapper for cfire dataset tuple"""
    train_loader: DataLoader
    test_loader: DataLoader
    val_loader: DataLoader
    n_dim: int
    n_classes: int


