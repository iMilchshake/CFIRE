# --- imports ---------------------------------------------------------------
import re
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1.  Parse one result file ---------------------------------------------
def parse_results(path, experiment_name=None):
    """
    Read a CFIRE frequency-sweep result file and return a DataFrame with
    freq_threshold, val_acc, test_acc, rule_size, seed and experiment columns.
    """
    rows = []
    freq = None                                                     # current threshold
    pat_header = re.compile(r"###\s*freq_threshold\s*=\s*([\d.]+)")
    pat_row    = re.compile(
        r"\[\d+\]\s*seed=(\d+)\s+cfire\s+val_acc=([\d.]+),\s*test_acc=([\d.]+),\s*rule_size=(\d+)"
    )

    with open(path) as fh:
        for line in fh:
            if m := pat_header.match(line):
                freq = float(m.group(1))
                continue
            if m := pat_row.match(line):
                rows.append(
                    dict(
                        experiment = experiment_name or Path(path).stem,
                        freq_threshold = freq,
                        seed      = int(m.group(1)),
                        val_acc   = float(m.group(2)),
                        test_acc  = float(m.group(3)),
                        rule_size = int(m.group(4)),
                    )
                )
    return pd.DataFrame(rows)


# --- 2.  Collect all experiments ------------------------------------------
files = [
    "topk=2_frequcy_experiments.txt",
    "binarise_0.01threshhold_frequqncy_experiment.txt",
]
df = pd.concat([parse_results(f) for f in files], ignore_index=True)

# --- 3.  Plot helper --------------------------------------------------------
def plot_threshold_sweep(
        data: pd.DataFrame,
        accuracy: str = "val",        # "val"  or "test"
        average: bool = False,        # True = plot per-threshold mean
        hue: str = "experiment",      # colour dimension
        style: str = "whitegrid",     # Seaborn style
        palette = "tab10",
):
    """
    Plot freq_threshold vs accuracy (left y) and rule_size (right y).

    Parameters
    ----------
    accuracy : {"val", "test"}
        Which accuracy column to plot.
    average : bool
        If True draw one point (mean) per threshold; else draw every seed.
    hue : str
        Column that distinguishes multiple experiments.
    """
    acc_col = f"{accuracy}_acc"
    sns.set(style=style)
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # prepare data (aggregate if requested)
    plot_df = (data
               if not average
               else data.groupby([hue, "freq_threshold"])
               .agg({acc_col: "mean", "rule_size": "mean"})
               .reset_index())

    # --- left axis: accuracy ------------------------------------------------
    if average:
        sns.lineplot(data=plot_df, x="freq_threshold", y=acc_col,
                     hue=hue, marker="o", ax=ax1, palette=palette)
    else:
        sns.scatterplot(data=plot_df, x="freq_threshold", y=acc_col,
                        hue=hue, ax=ax1, palette=palette)
    ax1.set_ylabel(f"{accuracy.capitalize()} accuracy")

    # --- right axis: rule size ---------------------------------------------
    ax2 = ax1.twinx()
    if average:
        sns.lineplot(data=plot_df, x="freq_threshold", y="rule_size",
                     hue=hue, marker="X", linestyle="--", legend=False,
                     ax=ax2, palette=palette)
    else:
        sns.scatterplot(data=plot_df, x="freq_threshold", y="rule_size",
                        hue=hue, marker="X", legend=False,
                        ax=ax2, palette=palette)
    ax2.set_ylabel("Rule size")
    ax1.set_xlabel("Frequency threshold")

    # --- cosmetics + combined legend ----------------------------------------
    from matplotlib.lines import Line2D

    ttl = f"Frequency sweep – {accuracy} accuracy, " + \
          ("mean of 5 seeds" if average else "all seeds")
    ax1.set_title(ttl)

    # Get default Seaborn legend handles (experiment-color mapping)
    handles1, labels1 = ax1.get_legend_handles_labels()

    # Custom legend entries for metric type
    custom_legend = [
        Line2D([0], [0], marker='o', color='w', label='Accuracy',
               markerfacecolor='gray', markersize=8),
        Line2D([0], [0], marker='X', color='w', label='Rule size',
               markerfacecolor='gray', markersize=8),
    ]

    # Combine both sets
    combined_handles = handles1 + custom_legend
    combined_labels = labels1 + ["Accuracy", "Rule size"]

    ax1.legend(handles=combined_handles, labels=combined_labels,
               title="Legend", loc="upper right", borderaxespad=0.5)

    fig.tight_layout()
    fig.savefig(f"threshold_sweep_{accuracy}_avg={average}.svg", format="svg")
    plt.show()



# --- 4.  Examples -----------------------------------------------------------
#plot_threshold_sweep(df, accuracy="val",  average=False)   # all points
plot_threshold_sweep(df, accuracy="test",  average=True)    # per-threshold mean
# plot_threshold_sweep(df, accuracy="test", average=True)  # test accuracy instead
