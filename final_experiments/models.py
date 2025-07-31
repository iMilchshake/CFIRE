import logging
from glob import glob
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight

from cfire_lab_experiments.util import loader_to_tensor
from final_experiments.types import CFIREDataset
from lxg.attribution import kernelshap
from lxg.datasets import RandomSeed, dataset_callables
from lxg.models import make_ff
from lxg.util import create_checkpoint

PATIENCE = 90  # early stop patience
WEIGHT_DECAY = 1e-4
LEARNING_RATE = 1e-3
MAX_EPOCHS = 1000
MAX_RETRIES = 5
EPS = 1e-7


class ModelFiles(NamedTuple):
    model_idx: int
    model_path: Path
    expl_path: Path


def load_or_train_models(
    dataset: CFIREDataset,
    n_models: int,
    save_dir: Path,
    seed: int = 42,
) -> list[ModelFiles]:
    """If there are already enough completed models+explanations return their paths, otherwise re-train"""
    save_dir.mkdir(parents=True, exist_ok=True)

    found_models: list[ModelFiles] = []
    for m in map(Path, glob(str(save_dir / "*_model.ckpt"))):
        idx_str = m.stem.split("_", 1)[0]
        if idx_str.isdigit():
            idx = int(idx_str)
            expl = save_dir / f"{idx}_expl.pt"
            if expl.exists():
                found_models.append(ModelFiles(idx, m, expl))

    n_available_models = len(found_models)
    if n_available_models >= n_models:
        found_models.sort(key=lambda model_files: model_files.model_idx)
        logging.info(f"re-using existing {n_models}/{n_available_models} models")
        return found_models[:n_models]

    # not enough complete models -> train new ones (overwrite existing files)
    return init_model_and_explanations(
        dataset=dataset,
        n_models=n_models,
        save_dir=save_dir,
        seed=seed,
    )


def init_model_and_explanations(
    dataset: CFIREDataset,
    n_models: int,
    save_dir: Path,
    seed: int = 42,
):
    # load dataset
    X_train, y_train = loader_to_tensor(dataset.train_loader)
    X_val, y_val = loader_to_tensor(dataset.val_loader)
    X_test, y_test = loader_to_tensor(dataset.test_loader)

    # class weights
    cls_weights_np = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(dataset.n_classes),
        y=y_train.numpy(),
    )
    class_weights = torch.tensor(cls_weights_np, dtype=torch.float32)

    model_stats = []
    paths: list[ModelFiles] = []
    for model_idx in range(n_models):
        for retry in range(1, MAX_RETRIES + 1):

            # TODO: other model dims?
            model = make_ff(
                [dataset.n_dim, 128, 128, dataset.n_classes], torch.nn.ReLU
            ).to("cpu")

            _train_torch_model(
                model,
                dataset.train_loader,
                X_val,
                y_val,
                criterion=nn.CrossEntropyLoss(weight=class_weights),
                optimizer=optim.Adam(
                    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
                ),
                num_epochs=MAX_EPOCHS,
            )

            y_val_pred = model.predict_batch(X_val)
            if len(torch.unique(y_val_pred)) == dataset.n_classes:
                logging.info(
                    f"model {model_idx} train success after {retry} attempt(s)"
                )
                break
            print(f"model {model_idx} retry {retry}: missing classes, re‑training…")
        else:
            raise RuntimeError(
                f"model {model_idx} failed to predict all {dataset.n_classes} classes after {MAX_RETRIES} retries"
            )

        # final evaluation
        y_train_pred = model.predict_batch(X_train)
        y_test_pred = model.predict_batch(X_test)
        train_acc = float(np.mean(y_train_pred.numpy() == y_train.numpy()))
        val_acc = float(np.mean(y_val_pred.numpy() == y_val.numpy()))
        test_acc = float(np.mean(y_test_pred.numpy() == y_test.numpy()))

        model_stats.append(
            {
                "model_idx": model_idx,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "test_acc": test_acc,
            }
        )

        model_path = save_dir / f"{model_idx}_model.ckpt"
        explanations_path = save_dir / f"{model_idx}_expl.pt"
        paths.append(ModelFiles(model_idx, model_path, explanations_path))
        create_checkpoint(model_path, model)

        # explanations
        with RandomSeed(seed):
            kernelshap_mask = torch.arange(0, dataset.n_dim)
            explanations = kernelshap(
                model=model,
                data=X_val,
                targets=y_val_pred,
                inference_fn=model.predict_batch_softmax,
                n_samples=300,
                masks=kernelshap_mask,
            )
            torch.save(explanations, explanations_path)

    # store model stats
    pd.DataFrame(model_stats).to_csv(save_dir / "model_stats.csv")

    return paths


def _train_torch_model(
    model,
    train_loader,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    criterion,
    optimizer: optim.Optimizer,
    num_epochs: int,
):
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_X.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            val_outputs = model(
                X_val
            )  # assumes that entire validation set fits into memory
            val_loss = criterion(val_outputs, y_val).item()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] train_loss: {epoch_loss:.4f} val_loss: {val_loss:.4f}",
                end="\r" if epoch + 1 < num_epochs else "\n",
            )

        if epochs_no_improve >= PATIENCE:
            print(
                f"Early stopping at epoch {epoch + 1} (no improvement for {PATIENCE} epochs)"
            )
            break


# example usage:
if __name__ == "__main__":
    dataset_name = "abalone"

    model_dir = Path(f"./experiments/models/{dataset_name}/")
    model_dir.mkdir(parents=True, exist_ok=True)
    dataset_fn = dataset_callables[dataset_name]
    init_model_and_explanations(
        dataset=dataset_fn(),
        n_models=2,
        save_dir=model_dir,
    )
