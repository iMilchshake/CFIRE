import colorsys
import logging
import sys
from glob import glob
from itertools import chain
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm

from cfire.cfire_module import ItemsetNodeCollection
from cfire.nodeselection import deduplicate_rules
from cfire_lab_experiments.test_cfire_utils import load_data, build_model, fit_cfire


def build_coverage_matrices(
    frequent_nodes: List[ItemsetNodeCollection],
) -> list[np.ndarray]:
    """
    matrices[c][i, j] == True <=> sample i is in support of node j of class c.
    """
    matrices = []
    for col in frequent_nodes:
        supports = col.class_support
        if not supports:
            matrices.append(np.empty((0, 0), dtype=bool))
            continue
        sample_universe = set(chain.from_iterable(supports))
        n_samples_c = max(sample_universe) + 1
        mat = np.zeros((n_samples_c, len(supports)), dtype=bool)
        for j, s in enumerate(supports):
            mat[list(s), j] = True
        matrices.append(mat)
    return matrices


def group_identical_nodes(matrices: List[np.ndarray]) -> List[List[List[int]]]:
    groups_per_class = []
    for mat in matrices:  # (samples, nodes)
        mapping: dict[bytes, list[int]] = {}
        for j in range(mat.shape[1]):  # iterate nodes
            key = mat[:, j].tobytes()  # hash of column
            mapping.setdefault(key, []).append(j)
        groups_per_class.append(list(mapping.values()))
    return groups_per_class


def _unique_colors(
    n: int,
    false_color: str,
    saturation_range=(0.65, 0.69),
    brightness_range=(0.85, 0.86),
) -> ListedColormap:
    hues = np.random.permutation(np.linspace(0, 1, n, endpoint=False))
    palette = [false_color]
    for h in hues:
        s = np.random.uniform(*saturation_range)
        b = np.random.uniform(*brightness_range)
        palette.append((*colorsys.hsv_to_rgb(h, s, b), 1.0))
    return ListedColormap(palette)


def plot_coverage_matrices(
    matrices: List[np.ndarray],
    path: str | Path = "coverage_matrices.png",
    groups: Optional[List[List[List[int]]]] = None,
    cell_size: float = 0.15,
    false_color: str = "#f5f5f5",
) -> Path:
    n_classes = len(matrices)
    max_samples = max(m.shape[0] for m in matrices)
    total_rules = sum(m.shape[1] for m in matrices)
    fig = plt.figure(figsize=(cell_size * max_samples, cell_size * total_rules))
    axes = [fig.add_subplot(n_classes, 1, i + 1) for i in range(n_classes)]

    for c, (mat, ax) in enumerate(zip(matrices, axes)):
        mat = mat.T  # rows = rules, cols = samples
        if groups is not None:
            group_map = np.zeros(mat.shape[0], dtype=int)
            for gid, node_idxs in enumerate(groups[c], start=1):
                group_map[node_idxs] = gid
            colour_grid = (mat.T * group_map).T  # broadcast
            n_groups = group_map.max()
            cmap = _unique_colors(n_groups, false_color)
            norm = BoundaryNorm(range(n_groups + 2), cmap.N)
            data = colour_grid
        else:
            true_color = "#4c87d9"
            cmap = ListedColormap([false_color, true_color])
            norm = None
            data = mat.astype(int)

        ax.imshow(data, aspect="equal", interpolation="nearest", cmap=cmap, norm=norm)
        ax.set_title(f"class {c}")
        ax.set_xlabel("sample index")
        ax.set_ylabel("rule index")
        ax.invert_yaxis()

    plt.tight_layout()
    out_path = Path(path).with_suffix(".png")
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out_path

def analyze_groups(
    frequent_nodes: List[ItemsetNodeCollection],
    groups_per_class: List[List[List[int]]],
) -> None:
    for cls_idx, (collection, class_groups) in enumerate(
        zip(frequent_nodes, groups_per_class)
    ):
        nodes = collection.nodes
        print(f"\nclass {cls_idx}")
        for g_idx, node_idxs in enumerate(class_groups, start=0):
            print(f"\tgroup {g_idx}: {node_idxs}")
            for j in node_idxs:
                n = nodes[j]
                print(
                    f"\t node {j}: "
                    f"complexity_factor={n.complexity_factor:.3f}, "
                    f"completeness_factor={n.completeness_factor:.3f}, "
                    f"n_literals={n.dnf.n_literals}, "
                    f"n_rules={n.dnf.n_rules}"
                    f" >> {n.dnf.rules}"
                )


def deduplicate_coverage_matrices(
        coverage_matrices: list[np.ndarray],
        groups_per_class: list[list[list[int]]],
) -> list[np.ndarray]:
    dedup = []
    for mat, class_groups in zip(coverage_matrices, groups_per_class):
        n_samples = mat.shape[0]
        out = np.empty((n_samples, len(class_groups)), dtype=bool)
        for g, node_idxs in enumerate(class_groups):
            out[:, g] = mat[:, node_idxs[0]]   # identical across the group
        dedup.append(out)
    return dedup

def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    # build model & CFIRE
    X_val, X_test, n_dim, n_classes = load_data()

    model_paths = sorted(glob("./models/tmp_*.ckpt"))
    explanation_paths = sorted(glob("./models/explanations_*.pt"))
    results = []

    SEEDS = [42, 43]

    for model_idx, model_path in enumerate(model_paths):
        logging.info(f"MODEL_IDX = {model_idx}")
        for seed in SEEDS:
            model = build_model(n_dim, n_classes, Path(model_path))
            logging.info(f"\tFIT CFIRE SEED={seed}")
            cfire = fit_cfire(model, X_val, Path(explanation_paths[model_idx]), seed)

            coverage_matricies = build_coverage_matrices(cfire.frequent_nodes)
            groups_per_class = group_identical_nodes(coverage_matricies)
            analyze_groups(cfire.frequent_nodes, groups_per_class)
            plot_coverage_matrices(coverage_matricies, groups=groups_per_class, path=f"coverage_matricies_{model_idx}_{seed}.png")

            dedup_coverage_matricies = deduplicate_coverage_matrices(coverage_matricies, groups_per_class)
            plot_coverage_matrices(dedup_coverage_matricies, f"coverage_matricies_dedup_{model_idx}_{seed}.png")


if __name__ == "__main__":
    main()
