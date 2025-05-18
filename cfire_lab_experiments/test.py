# trying to extend provided test script to use actual datasets

import os

from torch.optim.optimizer import Optimizer

default_path = "../"
os.chdir(default_path)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from lxg.datasets import NumpyRandomSeed, TorchRandomSeed
from lxg.attribution import kernelshap
from lxg.models import make_ff, SimpleNet

from cfire.cfire_module import CFIRE
from cfire.util import __preprocess_explanations


def main():
    # Load dataset
    dataset = load_iris()
    X, y = dataset["data"], dataset["target"]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.6, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    input_size = X.shape[1]
    output_size = len(np.unique(y))
    print(f"n_samples\ntrain: {len(X_train)}\n val: {len(X_val)}\n test: {len(X_test)}")

    # Convert to PyTorch tensors
    X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(
        y_train, dtype=torch.long
    )
    X_val, y_val = torch.tensor(X_val, dtype=torch.float32), torch.tensor(
        y_val, dtype=torch.long
    )
    X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(
        y_test, dtype=torch.long
    )

    # define model + CFIRE parameters
    model = make_ff([input_size, 128, 128, output_size], torch.nn.ReLU).to("cpu")
    kernelshap_mask = torch.arange(0, input_size)
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
    )
    y_val_model_pred = model.predict_batch(X_val).numpy()
    print(f"model val accuacy: {np.mean(y_val_model_pred == y_val.numpy())}")
    y_test_model_pred = model.predict_batch(X_test).numpy()
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
    optimizer: Optimizer,
    num_epochs=300,
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
