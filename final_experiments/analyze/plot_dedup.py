#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from final_experiments.analyze.utils import init_theme

init_theme()


raw_path = Path("./experiments/2_grid/analysis/dedup/raw_pct.csv")
out_path = Path("./experiments/2_grid/analysis/dedup/scatter_pct.png")

df = pd.read_csv(raw_path)

agg = (
    df.groupby(["dataset", "drop", "expl_method"], as_index=False)
    .mean(numeric_only=True)
)

# keep only drop=True
agg = agg[agg["drop"]]

# reshape for literal_count and rule_size
df_long = pd.melt(
    agg,
    id_vars=["dataset", "drop", "expl_method", "pct_test_f1_weighted"],
    value_vars=["pct_literal_count", "pct_rule_size"],
    var_name="metric",
    value_name="xval",
)

g = sns.relplot(
    data=df_long,
    x="xval",
    y="pct_test_f1_weighted",
    hue="dataset",
    style="expl_method",
    kind="scatter",
    row="metric",       # only rows now
    height=3,
    aspect=1.4,
    legend="brief",
    palette=sns.color_palette("hls", n_colors=20),
    facet_kws=dict(sharex=False, sharey=False),
)

xlabel_map = {
    "pct_literal_count": "Δ Literal Count (%)",
    "pct_rule_size": "Δ Rule Size (%)",
}
for r, row_name in enumerate(g.row_names):
    for ax in g.axes[r]:
        ax.set_xlabel(xlabel_map.get(row_name, row_name))
        ax.axvline(0, color="grey", lw=1, alpha=0.6)
        ax.axhline(0, color="grey", lw=1, alpha=0.6)

g.set_ylabels("Δ Test F1 (%)")
g.set_titles(row_template="{row_name}")

out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"saved: {out_path}")
