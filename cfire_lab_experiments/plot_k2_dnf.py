# assumes maximum of 2 literals per term!
from pathlib import Path
import torch
import numpy as np
import pickle

from cfire_lab_experiments.test_cfire import pprint_dnf_rules

# init data dirs
model_dir = Path("./models/")
experiment_dir = Path("./experiments/")
model_dir.mkdir(parents=True, exist_ok=True)
experiment_dir.mkdir(parents=True, exist_ok=True)

# load paths
X_val = torch.load(experiment_dir / "X_val.pt")
y_val = torch.load(experiment_dir / "y_val.pt")
y_val_model_pred = np.load(experiment_dir / "y_val_model_pred.npy")
y_test_model_pred = np.load(experiment_dir / "y_test_model_pred.npy")

# load rules back into cfire
dnf_path = experiment_dir / "dnf.pkl"
with dnf_path.open("rb") as f:
    dnf_rules = pickle.load(f)


print(dnf_rules)
pprint_dnf_rules(dnf_rules)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict

X_np = X_val.numpy()

# 1. compute global feature extents
feature_extents = defaultdict(lambda: [float("inf"), float("-inf")])
for class_terms in dnf_rules:
    for term in class_terms:
        for feat, (low, high) in term:
            feature_extents[feat][0] = min(feature_extents[feat][0], low)
            feature_extents[feat][1] = max(feature_extents[feat][1], high)

# 2. collect distinct feature‐pairs
pairs = {
    tuple(sorted((term[0][0], term[1][0])))
    for class_terms in dnf_rules
    for term in class_terms
    if len(term) == 2
}
pairs = sorted(pairs)

# 3. prepare subplots
n = len(pairs)
cols = int(np.ceil(np.sqrt(n)))
rows = int(np.ceil(n / cols))
fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 8 * rows), squeeze=False)
cmap = plt.get_cmap("Set1")

# 4. draw rectangles, scatter, legend
for idx_pair, (i, j) in enumerate(pairs):
    ax = axes[idx_pair // cols][idx_pair % cols]
    # plot rectangles
    for class_idx, class_terms in enumerate(dnf_rules):
        for term in class_terms:
            color = cmap(class_idx)
            if len(term) == 2 and {term[0][0], term[1][0]} == {i, j}:
                (f1, (l1, h1)), (f2, (l2, h2)) = sorted(term, key=lambda x: x[0])
                rect = patches.Rectangle(
                    (l1, l2),
                    h1 - l1,
                    h2 - l2,
                    alpha=0.3,
                    edgecolor=color,
                    facecolor=color,
                )
                ax.add_patch(rect)
            elif len(term) == 1 and term[0][0] in (i, j):
                feat, (low, high) = term[0]
                if feat == i:
                    x0, x1 = low, high
                    y0, y1 = feature_extents[j]
                else:
                    x0, x1 = feature_extents[i]
                    y0, y1 = low, high
                rect = patches.Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    alpha=0.3,
                    edgecolor=color,
                    facecolor=color,
                )
                ax.add_patch(rect)
    # scatter actual points per class
    for class_idx in range(len(dnf_rules)):
        mask = y_val_model_pred == class_idx
        ax.scatter(
            X_np[mask, i],
            X_np[mask, j],
            s=10,
            color=cmap(class_idx),
            label=f"Class {class_idx}",
        )
    ax.legend()
    ax.set_xlabel(f"Feature {i}")
    ax.set_ylabel(f"Feature {j}")
    ax.set_xlim(feature_extents[i])
    ax.set_ylim(feature_extents[j])
    ax.set_title(f"{i} vs {j}")

# hide unused axes
for idx in range(n, rows * cols):
    fig.delaxes(axes[idx // cols][idx % cols])

plt.tight_layout()
plt.savefig("plot1.png", dpi=300)
plt.show()


features = sorted(
    {feat for class_terms in dnf_rules for term in class_terms for feat, _ in term}
)
n = len(features)
fig, axes = plt.subplots(n, 1, figsize=(8, 2 * n), sharex=False)

for ax, f in zip(axes, features):
    # draw rule‐intervals as boxes with outlines
    for class_idx, class_terms in enumerate(dnf_rules):
        for term in class_terms:
            for feat, (low, high) in term:
                if feat == f:
                    rect = patches.Rectangle(
                        (low, class_idx - 0.4),  #
                        high - low,
                        0.8,
                        facecolor=cmap(class_idx),
                        edgecolor=cmap(class_idx),
                        linewidth=1.5,
                        alpha=0.3,
                    )
                    ax.add_patch(rect)

    # scatter projected data
    for class_idx in range(len(dnf_rules)):
        mask = y_val_model_pred == class_idx
        xs = X_np[mask, f]
        ys = (
            np.full_like(xs, class_idx, dtype=float)
            + (np.random.rand(len(xs)) - 0.5) * 0.1
        )
        ax.scatter(xs, ys, s=10, color=cmap(class_idx))

    ax.set_ylabel("Class")
    ax.set_title(f"Feature {f}")
    ax.set_yticks(range(len(dnf_rules)))

plt.tight_layout()
plt.savefig("plot2.png", dpi=300)
plt.show()
