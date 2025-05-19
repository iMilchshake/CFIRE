import torch


def loader_to_tensor(loader):
    """collect all batches of dataloader into one tensor"""
    xs, ys = [], []
    for xb, yb in loader:
        xs.append(xb)
        ys.append(yb)
    X = torch.cat(xs)
    y = torch.cat(ys)
    return X, y
