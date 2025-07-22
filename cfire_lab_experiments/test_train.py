from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight

import lxg.datasets as datasets
from cfire_lab_experiments.util import loader_to_tensor
from lxg.attribution import kernelshap
from lxg.datasets import RandomSeed
from lxg.models import make_ff
from lxg.util import create_checkpoint

# init model dir
model_dir = Path("./cfire_lab_experiments/models/")
model_dir.mkdir(parents=True, exist_ok=True)

N_MODELS = 10
PATIENCE = 90 # early stop patience
WEIGHT_DECAY = 1e-4
LEARNING_RATE = 1e-3
MAX_EPOCHS = 1000
MAX_RETRIES = 5
EPS = 1e-7


def main():
    dataset = datasets.get_abalone()
    train_loader, test_loader, val_loader, n_dim, n_classes = dataset

    print(
        f"n_samples\ntrain: {len(train_loader.dataset)}\n val: {len(val_loader.dataset)}\n test: {len(test_loader.dataset)}"
    )

    # tensors for statistics / evaluation / explanations
    X_val, y_val = loader_to_tensor(val_loader)
    X_test, y_test = loader_to_tensor(test_loader)
    X_train, y_train = loader_to_tensor(train_loader)

    # ------- standard scaling based on training set -------
    train_mean = X_train.mean(0, keepdim=True)
    train_std = X_train.std(0, keepdim=True) + EPS

    def scale(x: torch.Tensor) -> torch.Tensor:
        return (x - train_mean) / train_std

    X_train_s = scale(X_train)
    X_val_s = scale(X_val)
    X_test_s = scale(X_test)

    # ------- class weights (sklearn) -------
    cls_weights_np = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(n_classes),
        y=y_train.numpy(),
    )
    class_weights = torch.tensor(cls_weights_np, dtype=torch.float32)

    for i in range(N_MODELS):
        for retry in range(1, MAX_RETRIES + 1):
            model = make_ff([n_dim, 128, 128, n_classes], torch.nn.ReLU).to("cpu")

            train_model(
                model,
                train_loader,
                train_mean,
                train_std,
                X_val_s,
                y_val,
                criterion=nn.CrossEntropyLoss(weight=class_weights),
                optimizer=optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY),
                num_epochs=MAX_EPOCHS,
            )

            y_val_pred = model.predict_batch(X_val_s)
            if len(torch.unique(y_val_pred)) == n_classes:
                print(f"model {i} converged after {retry} attempt(s)")
                break
            print(f"model {i} retry {retry}: missing classes, re‑training…")
        else:
            raise RuntimeError(
                f"model {i} failed to predict all {n_classes} classes after {MAX_RETRIES} retries"
            )

        # final evaluation
        y_train_pred = model.predict_batch(X_train_s)
        y_test_pred = model.predict_batch(X_test_s)
        print(f"model {i} train accuracy: {np.mean(y_train_pred.numpy() == y_train.numpy())}")
        print(f"model {i} val accuracy:   {np.mean(y_val_pred.numpy() == y_val.numpy())}")
        print(f"model {i} test accuracy:  {np.mean(y_test_pred.numpy() == y_test.numpy())}")

        create_checkpoint(model_dir / f"tmp_{i}.ckpt", model)

        # explanations
        with RandomSeed(42):
            kernelshap_mask = torch.arange(0, n_dim)
            explanations = kernelshap(
                model=model,
                data=X_val_s,
                targets=y_val_pred,
                inference_fn=model.predict_batch_softmax,
                n_samples=300,
                masks=kernelshap_mask,
            )
            torch.save(explanations, model_dir / f"explanations_{i}.pt")
            assert torch.equal(explanations, torch.load(model_dir / f"explanations_{i}.pt"))


def train_model(
        model,
        train_loader,
        mean: torch.Tensor,
        std: torch.Tensor,
        X_val_s: torch.Tensor,
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
            batch_X = (batch_X - mean) / std
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_X.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_s)
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
            print(f"Early stopping at epoch {epoch + 1} (no improvement for {PATIENCE} epochs)")
            break


if __name__ == "__main__":
    main()
