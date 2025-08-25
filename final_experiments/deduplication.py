from itertools import chain
from typing import Callable

import numpy as np

from cfire.cfire_module import CFIRE
from cfire.gely import ItemsetNode
from lxg.models import DNFClassifier


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
