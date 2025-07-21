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
    plt.figure()
    sns.boxplot(data=df, x="composition", y=y)
    plt.title(f"{y.replace('_', ' ').title()} by Composition Method")
    plt.savefig(out)

# pair wise comparison plots
df["group"] = df[["model_idx", "seed"]].astype(str).agg("-".join, axis=1)
pivot = df.pivot(
    index="group",
    columns="composition",
    values=["test_acc", "rule_size", "literal_count"],
)
meta = df.drop_duplicates("group")[["group", "model_idx"]]


def facet_delta(alt, base):
    deltas = (
        pd.DataFrame(
            {
                "delta_test_acc": pivot["test_acc"][alt] - pivot["test_acc"][base],
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
        id_vars=["group", "model_idx", "delta_test_acc"],
        value_vars=["delta_rule_size", "delta_literal_cnt"],
        var_name="metric",
        value_name="delta_size",
    ).replace({"delta_rule_size": "Rule Size", "delta_literal_cnt": "Literal Count"})

    g = sns.FacetGrid(
        long,
        col="metric",
        hue="model_idx",
        sharey=True,
        sharex=False,
        height=5,
        aspect=1.2,
        palette="Set2",
    )
    g.set_titles("")
    g.map_dataframe(sns.scatterplot, x="delta_size", y="delta_test_acc", alpha=0.8)

    for metric_name, ax in g.axes_dict.items():
        ax.axhline(0, c="k", ls="--", lw=1)
        ax.axvline(0, c="k", ls="--", lw=1)
        ax.set_xlabel(f"Δ {metric_name}")  # proper x‑axis label
        ax.set_ylabel("Δ Test Accuracy")

    g.add_legend(title="model_idx")
    g.fig.suptitle(f"{alt} vs {base}", fontsize=14)
    g.fig.subplots_adjust(top=0.88)
    g.tight_layout()
    g.savefig(f"facet_{alt}_vs_{base}.png")


facet_delta("default_dedup", "default")
facet_delta("inv_freq_set_cover", "default_dedup")
facet_delta("inv_freq_set_cover", "default")
