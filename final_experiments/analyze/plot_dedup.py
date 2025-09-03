#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from final_experiments.analyze.utils import init_theme

init_theme()
# plt.rcParams["figure.constrained_layout.use"]=True


raw_path = Path("./experiments/2_grid/analysis/dedup/raw_pct.csv")
out_path = Path("./experiments/2_grid/analysis/dedup/scatter_pct.pdf")

df = pd.read_csv(raw_path)
df = df.rename(columns={"dataset": "Dataset", "expl_method": "Explainer"})

agg = df.groupby(["Dataset", "drop", "Explainer"], as_index=False).mean(
    numeric_only=True
)

# keep only drop=True
agg = agg[agg["drop"]]

# drop method "merged"
agg = agg[agg["Explainer"] != "Merged"]

# rename methods
agg["Explainer"] = (
    agg["Explainer"]
    .map(
        {
            "kernelshap": "KernelSHAP",
            "ig": "IG",
            "lime": "LIME",
        }
    )
    .fillna(agg["Explainer"])
)

# reshape for literal_count and rule_size
df_long = pd.melt(
    agg,
    id_vars=["Dataset", "drop", "Explainer", "pct_test_f1_weighted"],
    value_vars=["pct_literal_count", "pct_rule_size"],
    var_name="metric",
    value_name="xval",
)

g = sns.relplot(
    data=df_long,
    x="xval",
    y="pct_test_f1_weighted",
    hue="Dataset",
    style="Explainer",
    kind="scatter",
    col="metric",  # side-by-side
    height=3,
    aspect=1,
    legend="brief",
    facet_kws=dict(sharex=False, sharey=False),
    s=70,
)

xlabel_map = {
    "pct_literal_count": "Δ Literal Count (%)",
    "pct_rule_size": "Δ Rule Size (%)",
}
for ax, col_name in zip(g.axes[0], g.col_names):
    ax.set_xlabel(xlabel_map.get(col_name, col_name))
    ax.axvline(0, color="grey", lw=1, alpha=0.6)
    ax.axhline(0, color="grey", lw=1, alpha=0.6)

g.set_ylabels("Δ Test F1 (%)")
g.set_titles("")  # remove facet titles

out_path.parent.mkdir(parents=True, exist_ok=True)
# plt.tight_layout()
plt.savefig(out_path, bbox_inches="tight")
plt.close()

print(f"saved: {out_path}")
