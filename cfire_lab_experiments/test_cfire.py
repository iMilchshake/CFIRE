# this script fits CFIRE based on a trained model and explanations
# run `test_train.py` before!

from typing import List, Tuple
import numpy as np
import torch
from pathlib import Path
import random
import pickle

from lxg.datasets import NumpyRandomSeed, TorchRandomSeed
import lxg.datasets as datasets
from lxg.models import make_ff
from lxg.util import restore_checkpoint
from cfire.cfire_module import CFIRE
from cfire.util import __preprocess_explanations, __preprocess_explanations_ext

from cfire_lab_experiments.util import loader_to_tensor

# init data dirs
model_dir = Path("./models/")
experiment_dir = Path("./experiments/")
model_dir.mkdir(parents=True, exist_ok=True)
experiment_dir.mkdir(parents=True, exist_ok=True)


def ks_fn_cached(path):
    return lambda _inference_fn, _data, _targets: torch.load(path)


def pprint_dnf_rules(dnf_rules: List[List[List[Tuple[int, Tuple[float, float]]]]]):
    for class_idx, class_data in enumerate(dnf_rules):
        print(f"\nClass {class_idx}:\n")
        rules_str = []
        for term in class_data:
            conjuncts = [
                f"F{feature_dim} ∈ [{low:.2f}, {high:.2f}]"
                for feature_dim, (low, high) in term
            ]
            rules_str.append(" ∧ ".join(conjuncts))
        print(" ∨\n".join(f"  ({r})" for r in rules_str))


def rule_size(dnf_rules: List[List[List[Tuple[int, Tuple[float, float]]]]]) -> int:
    """Total number of (conjunctive) terms across all classes"""
    return sum(len(class_data) for class_data in dnf_rules)


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
    # expl_binarization_fn = lambda x: __preprocess_explanations(x, filtering=0.01) > 0
    expl_binarization_fn = lambda x: __preprocess_explanations_ext(x, top_k=2) > 0

    # restore training checkpoint
    model_path = model_dir / "tmp.ckpt"
    print(model_path)
    restore_checkpoint(model_path, model, train=False)

    y_train_model_pred = model.predict_batch(X_train).numpy()
    y_val_model_pred = model.predict_batch(X_val).numpy()
    y_test_model_pred = model.predict_batch(X_test).numpy()
    print(f"model train accuacy: {np.mean(y_train_model_pred == y_train.numpy())}")
    print(f"model val accuacy: {np.mean(y_val_model_pred == y_val.numpy())}")
    print(f"model test accuacy: {np.mean(y_test_model_pred == y_test.numpy())}")

    # config TODO: move somewhere nice
    SEED_COUNT = 3
    PRINT_RULES = False
    DUMP = False
    FREQ_THRESHOLDS = [0.01, 0.02, 0.05, 0.1, 0.2]

    # run CFIRE
    seeds = [random.randint(0, 2**32 - 1) for _ in range(SEED_COUNT)]
    for freq_threshold in FREQ_THRESHOLDS:
        print(f"### freq_threshold = {freq_threshold}")
        for idx, seed in enumerate(seeds):
            with NumpyRandomSeed(seed):
                with TorchRandomSeed(seed):
                    cfire = CFIRE(
                        localexplainer_fn=ks_fn_cached(model_dir / "explanations.pt"),
                        inference_fn=model.predict_batch_softmax,
                        expl_binarization_fn=expl_binarization_fn,
                        frequency_threshold=freq_threshold,
                    )
                    cfire._verbose = False
                    cfire.fit(X_val.numpy(), y_val_model_pred)
                    y_val_cfire_pred = cfire(X_val)
                    y_test_cfire_pred = cfire(X_test)
                    val_acc = np.mean(y_val_model_pred == y_val_cfire_pred)
                    test_acc = np.mean(y_test_model_pred == y_test_cfire_pred)
                    print(
                        f"[{idx}] seed={seed} cfire val_acc={val_acc:.3f}, test_acc={test_acc:.3f}, rule_size={rule_size(cfire.dnf.rules)}"
                    )

                    if PRINT_RULES:
                        pprint_dnf_rules(cfire.dnf.rules)

                    if DUMP:
                        torch.save(X_val, experiment_dir / "X_val.pt")
                        torch.save(y_val, experiment_dir / "y_val.pt")
                        np.save(
                            experiment_dir / "y_val_model_pred.npy", y_val_model_pred
                        )
                        np.save(
                            experiment_dir / "y_test_model_pred.npy", y_test_model_pred
                        )
                        dnf_path = experiment_dir / "dnf.pkl"
                        with dnf_path.open("wb") as dnf_file:
                            pickle.dump(cfire.dnf.rules, dnf_file)


if __name__ == "__main__":
    main()
