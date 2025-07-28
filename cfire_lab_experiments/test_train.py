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

PATIENCE = 90  # early stop patience
WEIGHT_DECAY = 1e-4
LEARNING_RATE = 1e-3
MAX_EPOCHS = 1000
MAX_RETRIES = 5
EPS = 1e-7


def init_model_and_explanations(
    dataset_fn,
    n_models: int,
    save_dir: Path,
    seed: int=42,
):
    # load dataset
    train_loader, test_loader, val_loader, n_dim, n_classes = dataset_fn()
    X_train, y_train = loader_to_tensor(train_loader)
    X_val, y_val = loader_to_tensor(val_loader)
    X_test, y_test = loader_to_tensor(test_loader)

    # class weights
    cls_weights_np = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(n_classes),
        y=y_train.numpy(),
    )
    class_weights = torch.tensor(cls_weights_np, dtype=torch.float32)

    for i in range(n_models):
        for retry in range(1, MAX_RETRIES + 1):

            # TODO: other model dims?
            model = make_ff([n_dim, 128, 128, n_classes], torch.nn.ReLU).to("cpu")

            _train_torch_model(
                model,
                train_loader,
                X_val,
                y_val,
                criterion=nn.CrossEntropyLoss(weight=class_weights),
                optimizer=optim.Adam(
                    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
                ),
                num_epochs=MAX_EPOCHS,
            )

            y_val_pred = model.predict_batch(X_val)
            if len(torch.unique(y_val_pred)) == n_classes:
                print(f"model {i} converged after {retry} attempt(s)")
                break
            print(f"model {i} retry {retry}: missing classes, re‑training…")
        else:
            raise RuntimeError(
                f"model {i} failed to predict all {n_classes} classes after {MAX_RETRIES} retries"
            )

        # final evaluation
        y_train_pred = model.predict_batch(X_train)
        y_test_pred = model.predict_batch(X_test)
        train_acc = float(np.mean(y_train_pred.numpy() == y_train.numpy()))
        val_acc = float(np.mean(y_val_pred.numpy() == y_val.numpy()))
        test_acc = float(np.mean(y_test_pred.numpy() == y_test.numpy()))
        print(f"model {i} train accuracy: {train_acc}")
        print(f"model {i} val accuracy:   {val_acc}")
        print(f"model {i} test accuracy:  {test_acc}")
        # TODO: instead of prints, we want this kind of info in some csv/json i think

        create_checkpoint(save_dir / f"tmp_{i}.ckpt", model)

        # explanations
        with RandomSeed(seed):
            kernelshap_mask = torch.arange(0, n_dim)
            explanations = kernelshap(
                model=model,
                data=X_val,
                targets=y_val_pred,
                inference_fn=model.predict_batch_softmax,
                n_samples=300,
                masks=kernelshap_mask,
            )
            torch.save(explanations, save_dir / f"explanations_{i}.pt")
            assert torch.equal(
                explanations, torch.load(save_dir / f"explanations_{i}.pt")
            )


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


if __name__ == "__main__":
    model_dir = Path("./cfire_lab_experiments/models/")
    model_dir.mkdir(parents=True, exist_ok=True)

    init_model_and_explanations(
        dataset_fn=datasets.get_abalone(),
        n_models=10,
        save_dir=model_dir,
    )
