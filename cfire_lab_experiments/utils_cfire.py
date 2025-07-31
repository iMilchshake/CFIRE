from itertools import chain
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn

import lxg.datasets as datasets
from cfire.cfire_module import CFIRE
from cfire.gely import ItemsetNode
from cfire.util import __preprocess_explanations_ext
from cfire_lab_experiments.util import loader_to_tensor
from lxg.datasets import RandomSeed
from lxg.models import make_ff, DNFClassifier
from lxg.util import restore_checkpoint
from .test_cfire import ks_fn_cached


# helpers / metrics
def get_rule_size(rules):
    return sum(len(class_rules) for class_rules in rules)


def get_literal_count(rules):
    return sum(
        len(conjunction)
        for class_rule in rules
        for clause in class_rule
        for conjunction in clause
    )


# --- cfire ---
def load_data():
    loaders = datasets.get_abalone()
    _, test_loader, val_loader, n_dim, n_classes = loaders
    X_val, _ = loader_to_tensor(val_loader)
    X_test, _ = loader_to_tensor(test_loader)
    return X_val, X_test, n_dim, n_classes


def load_model(n_dim: int, n_classes: int, model_path: Path) -> nn.Module:
    model = make_ff([n_dim, 128, 128, n_classes], torch.nn.ReLU).to("cpu")
    restore_checkpoint(model_path, model, train=False)
    return model


def fit_cfire(model, X_val: torch.Tensor, explanation_path: Path, seed: int):
    expl_bin: Callable = lambda x: __preprocess_explanations_ext(x, threshold=0.01) > 0
    with RandomSeed(seed):
        cfire = CFIRE(
            localexplainer_fn=ks_fn_cached(explanation_path),
            inference_fn=model.predict_batch_softmax,
            expl_binarization_fn=expl_bin,
        )
        cfire._verbose = False
        cfire.fit(X_val.numpy(), model.predict_batch(X_val).numpy())
    return cfire


def recalculate_rule_composition(
        cfire: CFIRE,
        fn_cover: Callable[
            [set[int], list[tuple[set[int], ItemsetNode]]], list[ItemsetNode]
        ],
        update_cfire: bool = True,
):
    """manually run rule composition, unless disabled will update cfire object"""
    n_classes = len(np.unique(cfire._labels))
    rules = []
    all_selected_nodes: list[list[ItemsetNode]] = []

    for class_idx in range(n_classes):
        class_support = cfire.frequent_nodes[class_idx].class_support
        class_nodes = cfire.frequent_nodes[class_idx].nodes

        sample_universe = set(chain.from_iterable(class_support))
        selected_nodes = fn_cover(sample_universe, list(zip(class_support, class_nodes)))

        all_selected_nodes.append(selected_nodes)
        rules.append([rule for node in selected_nodes for rule in node.dnf.rules[0]])

    # build final (multi class) dnf and recalculate performance
    final_dnf = DNFClassifier(rules, "accuracy")
    final_dnf.compute_rule_performance(cfire._data, cfire._labels)

    if update_cfire:
        cfire.dnf = final_dnf

    return final_dnf, all_selected_nodes


def evaluate_cfire(
        cfire: CFIRE, model: nn.Module, X_val: torch.Tensor, X_test: torch.Tensor
) -> dict:
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
        "literal_count": literal_count,
    }
