# trying to extend provided test script to use actual datasets


# TODO: what exactly is this for? we need to run the script from the project root. This way data such as datasets is stored to outside the project directory.. why?
# import os
# default_path = "../"
# os.chdir(default_path)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from lxg.datasets import NumpyRandomSeed, TorchRandomSeed
import lxg.datasets as datasets
from lxg.attribution import kernelshap
from lxg.models import make_ff, SimpleNet

from cfire.cfire_module import CFIRE
from cfire.util import __preprocess_explanations


def loader_to_tensor(loader):
    """collect all batches of dataloader into one tensor"""
    xs, ys = [], []
    for xb, yb in loader:
        xs.append(xb)
        ys.append(yb)
    X = torch.cat(xs)
    y = torch.cat(ys)
    return X, y


def main():
    # load dataset 'abalone' (tabular, multi-class, N=4177)
    # `_return_dataset` already properly applies standard scaling
    # `batch_sizes` refers to (train, val/test batch size) TODO: why would we want different batch sizes?
    dataset = datasets.get_abalone()
    print(dataset)
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
    kernelshap_mask = torch.arange(0, n_dim)
    _perturb_args = {"model": model, "n_samples": 300, "masks": kernelshap_mask}
    ks_fn = lambda inference_fn, data, targets: kernelshap(
        inference_fn=inference_fn,
        data=torch.from_numpy(data).float(),
        targets=torch.from_numpy(targets),
        **_perturb_args,
    )
    expl_binarization_fn = lambda x: __preprocess_explanations(x, filtering=0.01) > 0

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
    y_train_model_pred = model.predict_batch(X_train).numpy()
    y_val_model_pred = model.predict_batch(X_val).numpy()
    y_test_model_pred = model.predict_batch(X_test).numpy()
    print(f"model train accuacy: {np.mean(y_train_model_pred == y_train.numpy())}")
    print(f"model val accuacy: {np.mean(y_val_model_pred == y_val.numpy())}")
    print(f"model test accuacy: {np.mean(y_test_model_pred == y_test.numpy())}")

    with NumpyRandomSeed(42):
        with TorchRandomSeed(42):
            cfire = CFIRE(
                localexplainer_fn=ks_fn,
                inference_fn=model.predict_batch_softmax,
                expl_binarization_fn=expl_binarization_fn,
                frequency_threshold=0.01,
            )
            cfire.fit(X_val.numpy(), y_val_model_pred)
            y_test_cfire_pred = cfire(X_test)
            acc = np.mean(y_test_model_pred == y_test_cfire_pred)
            print(f"cfire acc: {acc}")
            print(f"cfire rules: {cfire.dnf.rules}")


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

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}",
                end="\r" if epoch + 1 < num_epochs else "\n",
            )


if __name__ == "__main__":
    main()
