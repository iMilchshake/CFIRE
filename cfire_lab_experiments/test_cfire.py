# this script fits CFIRE based on a trained model and explanations
# run `test_train.py` before!

# TODO: what exactly is this for? we need to run the script from the project root. This way data such as datasets is stored to outside the project directory.. why?
# import os
# default_path = "../"
# os.chdir(default_path)

import numpy as np
import torch
from pathlib import Path

from lxg.datasets import NumpyRandomSeed, TorchRandomSeed
import lxg.datasets as datasets
from lxg.models import make_ff
from lxg.util import restore_checkpoint
from cfire.cfire_module import CFIRE
from cfire.util import __preprocess_explanations

from cfire_lab_experiments.util import loader_to_tensor

# init model dir
model_dir = Path("./models/")
model_dir.mkdir(parents=True, exist_ok=True)


def ks_fn_cached(path):
    return lambda _inference_fn, _data, _targets: torch.load(path)


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
    expl_binarization_fn = lambda x: __preprocess_explanations(x, filtering=0.01) > 0

    # restore training checkpoint
    model_path = model_dir / "tmp.ckpt"
    print(model_path)
    restore_checkpoint(model_path, model)

    print(type(model))

    y_train_model_pred = model.predict_batch(X_train).numpy()
    y_val_model_pred = model.predict_batch(X_val).numpy()
    y_test_model_pred = model.predict_batch(X_test).numpy()
    print(f"model train accuacy: {np.mean(y_train_model_pred == y_train.numpy())}")
    print(f"model val accuacy: {np.mean(y_val_model_pred == y_val.numpy())}")
    print(f"model test accuacy: {np.mean(y_test_model_pred == y_test.numpy())}")

    # run CFIRE
    with NumpyRandomSeed(42):
        with TorchRandomSeed(42):
            cfire = CFIRE(
                localexplainer_fn=ks_fn_cached(model_dir / "explanations.pt"),
                inference_fn=model.predict_batch_softmax,
                expl_binarization_fn=expl_binarization_fn,
                frequency_threshold=0.01,
            )
            cfire.fit(X_val.numpy(), y_val_model_pred)
            y_test_cfire_pred = cfire(X_test)
            acc = np.mean(y_test_model_pred == y_test_cfire_pred)
            print(f"cfire acc: {acc}")
            print(f"cfire rules: {cfire.dnf.rules}")


if __name__ == "__main__":
    main()
