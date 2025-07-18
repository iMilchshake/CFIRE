import logging
from itertools import chain
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn

import lxg.datasets as datasets
from cfire.cfire_module import CFIRE
from cfire.gely import ItemsetNode
from cfire.nodeselection import weighted_greedy_cover
from cfire.util import __preprocess_explanations_ext
from cfire_lab_experiments.util import loader_to_tensor
from lxg.datasets import RandomSeed
from lxg.models import make_ff, DNFClassifier
from lxg.util import restore_checkpoint
from .test_cfire import ks_fn_cached, pprint_dnf_rules

PRUNE_WINS_THRESHOLDS = list(range(0, 25 + 1))
# MODEL_CKPT is unused for now
MODEL_CKPT = None
# EXPLANATIONS_PT is unused for now
EXPLANATIONS_PT = None


# helpers / metrics
def get_rule_size(rules):
    return sum(len(class_rules) for class_rules in rules)


def get_literal_count(rules):
    return sum(len(conjunction) for class_rule in rules for clause in class_rule for conjunction in clause)


# --- cfire ---
def load_data():
    loaders = datasets.get_abalone()
    _, test_loader, val_loader, n_dim, n_classes = loaders
    X_val, _ = loader_to_tensor(val_loader)
    X_test, _ = loader_to_tensor(test_loader)
    return X_val, X_test, n_dim, n_classes


def build_model(n_dim: int, n_classes: int, model_path: Path) -> nn.Module:
    model = make_ff([n_dim, 128, 128, n_classes], torch.nn.ReLU).to("cpu")
    restore_checkpoint(model_path, model, train=False)
    return model


def fit_cfire(model, X_val: torch.Tensor, explanation_path: Path, seed: int):
    expl_bin: Callable = lambda x: __preprocess_explanations_ext(x, threshold=0.01) > 0
    with RandomSeed(seed):
        cfire = CFIRE(localexplainer_fn=ks_fn_cached(explanation_path), inference_fn=model.predict_batch_softmax,
                      expl_binarization_fn=expl_bin, )
        cfire.fit(X_val.numpy(), model.predict_batch(X_val).numpy())
    return cfire


def recalculate_rule_composition(cfire: CFIRE,
                                 fn_cover: Callable[[set[int], list[tuple[set[int], ItemsetNode]]], list[ItemsetNode]]):

    """ manually run rule composition """
    n_classes = len(np.unique(cfire._labels))
    rules = []

    for class_idx in range(n_classes):
        class_support = cfire.frequent_nodes[class_idx].class_support
        nodes = cfire.frequent_nodes[class_idx].nodes

        sample_universe = set(chain.from_iterable(class_support))
        selected_nodes = fn_cover(sample_universe, list(zip(class_support, nodes)))

        rules.append([rule for node in selected_nodes for rule in node.dnf.rules[0]])

    # build final (multi class) dnf and recalculate performance
    final_dnf = DNFClassifier(rules, 'accuracy')
    final_dnf.compute_rule_performance(cfire._data, cfire._labels)

    return final_dnf


def evaluate_cfire(cfire: CFIRE, model: nn.Module, X_val: torch.Tensor, X_test: torch.Tensor) -> dict:
    # evaluate
    y_val = model.predict_batch(X_val).numpy()
    y_test = model.predict_batch(X_test).numpy()
    base_val_acc = (cfire(X_val) == y_val).mean()
    base_test_acc = (cfire(X_test) == y_test).mean()

    rule_size = get_rule_size(cfire.dnf.rules)
    literal_count = get_literal_count(cfire.dnf.rules)
    return {
        "val_acc": base_val_acc,
        "test_acc": base_test_acc,
        "rule_size": rule_size,
        "literal_count": literal_count
    }


import pandas as pd

def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # build model & CFIRE
    X_val, X_test, n_dim, n_classes = load_data()

    from glob import glob

    model_paths = sorted(glob("./models/tmp_*.ckpt"))
    explanation_paths = sorted(glob("./models/explanations_*.pt"))

    results = []

    for model_idx, model_path in enumerate(model_paths):
        for seed in [42, 43, 44, 45, 46]:
            model = build_model(n_dim, n_classes, Path(model_path))
            logging.info(f"FIT CFIRE SEED={seed}")
            cfire = fit_cfire(model, X_val, Path(explanation_paths[model_idx]), seed)

            eval_default = evaluate_cfire(cfire, model, X_val, X_test)
            results.append({"model_idx": model_idx, "seed": seed, "composition": "greedy_cover", **eval_default})

            cfire.dnf = recalculate_rule_composition(cfire, weighted_greedy_cover)
            eval_wgc = evaluate_cfire(cfire, model, X_val, X_test)
            results.append({"model_idx": model_idx, "seed": seed, "composition": "weighted_greedy_cover", **eval_wgc})


    df = pd.DataFrame(results)
    df.to_csv("cfire_eval_results.csv", index=False)

if __name__ == "__main__":
    main()
