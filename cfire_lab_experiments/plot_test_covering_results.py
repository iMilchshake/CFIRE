import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv("./cfire_eval_results.csv")
sns.set(style="whitegrid")

# boxplots
for y, out in [
    ("val_acc", "val_acc_by_composition.png"),
    ("test_acc", "test_acc_by_composition.png"),
    ("rule_size", "rule_size_by_composition.png"),
    ("literal_count", "literal_count_by_composition.png"),
]:
    plt.figure(figsize=(16, 8))
    sns.boxplot(data=df, x="composition", y=y)
    plt.title(f"{y.replace('_', ' ').title()} by Composition Method")
    plt.tight_layout()
    plt.savefig(out)

# pair wise comparison plots
df["group"] = df[["model_idx", "seed"]].astype(str).agg("-".join, axis=1)
pivot = df.pivot(
    index="group",
    columns="composition",
    values=["test_acc", "rule_size", "literal_count"],
)
meta = df.drop_duplicates("group")[["group", "model_idx", "seed"]]


def facet_delta(alt, base):
    deltas = (
        pd.DataFrame(
            {
                "delta_test_acc": 100
                * (pivot["test_acc"][alt] - pivot["test_acc"][base])
                / pivot["test_acc"][base],
                "delta_rule_size": pivot["rule_size"][alt] - pivot["rule_size"][base],
                "delta_literal_cnt": pivot["literal_count"][alt]
                - pivot["literal_count"][base],
            },
            index=pivot.index,
        )
        .reset_index()
        .merge(meta, on="group")
    )

    long = deltas.melt(
        id_vars=["group", "model_idx", "seed", "delta_test_acc"],
        value_vars=["delta_rule_size", "delta_literal_cnt"],
        var_name="metric",
        value_name="delta_size",
    ).replace({"delta_rule_size": "Rule Size", "delta_literal_cnt": "Literal Count"})

    style_order = sorted(df["seed"].unique())
    palette = sns.color_palette("tab10")
    g = sns.FacetGrid(
        long,
        col="metric",
        sharey=True,
        sharex=False,
        height=5,
        aspect=1.2,
        palette=palette,
    )
    g.set_titles("")
    g.map_dataframe(
        sns.scatterplot,
        x="delta_size",
        y="delta_test_acc",
        hue="model_idx",
        alpha=0.8,
        style="seed",
        style_order=style_order,
        legend="full",
        palette=palette,
    )

    for metric_name, ax in g.axes_dict.items():
        ax.axhline(0, c="k", ls="--", lw=1)
        ax.axvline(0, c="k", ls="--", lw=1)
        ax.set_xlabel(f"Δ {metric_name}")
        ax.set_ylabel("Δ Test Accuracy (%)")

    g.add_legend(title="model_idx")
    g.fig.suptitle(f"{alt} vs {base}", fontsize=14)
    g.fig.subplots_adjust(top=0.88)
    g.tight_layout()
    g.savefig(f"facet_{alt}_vs_{base}.png")


facet_delta("default_dedup", "default")
facet_delta("inv_freq_set_cover_a=1", "default")
facet_delta("inv_freq_set_cover_a=0.5", "default")
facet_delta("inv_freq_set_cover_a=1.5", "default")
facet_delta("inv_freq_set_cover_a=1.5", "inv_freq_set_cover_a=1")
facet_delta("inv_freq_set_cover_a=0.5", "inv_freq_set_cover_a=1")
facet_delta("best_val_acc", "default")

# --- relative boxplots (vs. “default”) ---------------------------------------

metrics_info = {
    "val_acc":  ("Δ Val Acc (%)",        True,  "rel_val_acc_by_composition.png"),
    "test_acc": ("Δ Test Acc (%)",       True,  "rel_test_acc_by_composition.png"),
    "rule_size": ("Δ Rule Size",         False, "rel_rule_size_by_composition.png"),
    "literal_count": ("Δ Literal Count", False, "rel_literal_count_by_composition.png"),
}

for metric, (ylabel, pct, outfile) in metrics_info.items():
    pivot = df.pivot(index="group", columns="composition", values=metric)
    delta = pivot.sub(pivot["default"], axis=0)
    if pct:                                   # % difference for accuracies
        delta = 100 * delta.div(pivot["default"], axis=0)
    delta = (
        delta.drop(columns=["default"])       # drop baseline (always 0)
             .melt(ignore_index=False, var_name="composition", value_name="delta")
             .reset_index()
    )

    plt.figure(figsize=(16, 8))
    sns.boxplot(data=delta, x="composition", y="delta")
    plt.axhline(0, c="k", ls="--", lw=1)
    plt.title(f"{ylabel} relative to Default by Composition Method")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()
