# this script trains a simple FF model and calculates explanations for validation split
# run `test_cfire.py` after!

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from lxg.datasets import NumpyRandomSeed, TorchRandomSeed
import lxg.datasets as datasets
from lxg.attribution import kernelshap
from lxg.models import make_ff
from lxg.util import create_checkpoint

from cfire_lab_experiments.util import loader_to_tensor

# init model dir
model_dir = Path("./models/")
model_dir.mkdir(parents=True, exist_ok=True)


def main():
    # load dataset 'abalone' (tabular, multi-class, N=4177)
    # `_return_dataset` already properly applies standard scaling
    # `batch_sizes` refers to (train, val/test batch size) TODO: why would we want different batch sizes?
    dataset = datasets.get_abalone()
    train_loader, test_loader, val_loader, n_dim, n_classes = dataset

    print(
        f"n_samples\ntrain: {len(train_loader.dataset)}\n val: {len(val_loader.dataset)}\n test: {len(test_loader.dataset)}"
    )

    # TODO: crappy workaround or ok?
    X_train, y_train = loader_to_tensor(train_loader)
    X_val, y_val = loader_to_tensor(val_loader)
    X_test, y_test = loader_to_tensor(test_loader)

    # define model + CFIRE parameters
    # TODO: where is model batch size defined? is it implicit?
    model = make_ff([n_dim, 128, 128, n_classes], torch.nn.ReLU).to("cpu")

    # train and evaluate black box model
    train_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optim.Adam(model.parameters()),
        num_epochs=300,
    )

    create_checkpoint(model_dir / "tmp.ckpt", model)

    y_train_model_pred = model.predict_batch(X_train)
    y_val_model_pred = model.predict_batch(X_val)
    y_test_model_pred = model.predict_batch(X_test)
    print(
        f"model train accuacy: {np.mean(y_train_model_pred.numpy() == y_train.numpy())}"
    )
    print(f"model val accuacy: {np.mean(y_val_model_pred.numpy() == y_val.numpy())}")
    print(f"model test accuacy: {np.mean(y_test_model_pred.numpy() == y_test.numpy())}")

    # we calculate explanations, as they dont change and are computational expensive
    with NumpyRandomSeed(42):
        with TorchRandomSeed(42):
            kernelshap_mask = torch.arange(0, n_dim)
            explanations = kernelshap(
                model=model,
                data=X_val,
                targets=y_val_model_pred,  #  target values are model outputs!!!
                inference_fn=model.predict_batch_softmax,
                n_samples=300,
                masks=kernelshap_mask,
            )
            torch.save(explanations, model_dir / "explanations.pt")
            sanity_check = torch.load(model_dir / "explanations.pt")
            assert torch.equal(explanations, sanity_check)
            print(explanations[0])
            print(sanity_check[0])


def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    criterion,
    optimizer: optim.Optimizer,
    num_epochs: int,
):
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            # TODO: add early stopping? or cant we use validation as we typically do,
            # as we use validation split for explanaions/cfire?

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}",
                end="\r" if epoch + 1 < num_epochs else "\n",
            )


if __name__ == "__main__":
    main()
