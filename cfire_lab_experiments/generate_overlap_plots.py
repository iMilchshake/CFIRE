from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────────────────────────────────────
# hard-coded console dump  (copy-pasted unedited)
# ─────────────────────────────────────────────────────────────────────────────

RAW_LOSSES = """
cl 1/term 0 | loss  405 (99.5% of 407) | wins    2 | acc 0.511
cl 0/term 8 | loss  136 (97.8% of 139) | wins    3 | acc 0.950
cl 1/term 3 | loss  124 (97.6% of 127) | wins    3 | acc 0.535
cl 1/term 12 | loss  112 (75.7% of 148) | wins   36 | acc 0.595
cl 1/term 5 | loss   95 (100.0% of  95) | wins    0 | acc 0.589
cl 2/term 43 | loss   90 (100.0% of  90) | wins    0 | acc 0.511
cl 2/term 38 | loss   80 (92.0% of  87) | wins    7 | acc 0.977
cl 2/term 16 | loss   79 (95.2% of  83) | wins    4 | acc 0.964
cl 2/term 25 | loss   67 (100.0% of  67) | wins    0 | acc 0.522
cl 2/term 14 | loss   62 (80.5% of  77) | wins   15 | acc 0.623
cl 0/term 9 | loss   60 (80.0% of  75) | wins   15 | acc 0.640
cl 2/term 11 | loss   58 (80.6% of  72) | wins   14 | acc 0.986
cl 2/term 21 | loss   53 (85.5% of  62) | wins    9 | acc 0.742
cl 1/term 11 | loss   46 (52.3% of  88) | wins   42 | acc 0.830
cl 2/term 48 | loss   40 (97.6% of  41) | wins    1 | acc 1.000
cl 2/term 28 | loss   33 (94.3% of  35) | wins    2 | acc 1.000
cl 2/term 31 | loss   22 (100.0% of  22) | wins    0 | acc 1.000
cl 1/term 10 | loss   20 (62.5% of  32) | wins   12 | acc 0.594
cl 2/term 15 | loss   17 (73.9% of  23) | wins    6 | acc 0.696
cl 2/term 27 | loss   17 (81.0% of  21) | wins    4 | acc 0.857
cl 2/term 34 | loss   16 (100.0% of  16) | wins    0 | acc 1.000
cl 2/term 10 | loss   15 (100.0% of  15) | wins    0 | acc 0.933
cl 2/term 29 | loss   14 (100.0% of  14) | wins    0 | acc 0.929
cl 0/term 3 | loss   13 (100.0% of  13) | wins    0 | acc 0.538
cl 2/term 30 | loss   11 (100.0% of  11) | wins    0 | acc 0.818
cl 2/term 44 | loss    9 (81.8% of  11) | wins    2 | acc 0.909
cl 2/term 42 | loss    9 (64.3% of  14) | wins    5 | acc 1.000
cl 2/term 39 | loss    8 (100.0% of   8) | wins    0 | acc 0.875
cl 2/term 46 | loss    8 (88.9% of   9) | wins    1 | acc 1.000
cl 2/term 49 | loss    7 (100.0% of   7) | wins    0 | acc 1.000
cl 2/term 33 | loss    7 (100.0% of   7) | wins    0 | acc 0.714
cl 2/term 45 | loss    7 (87.5% of   8) | wins    1 | acc 0.750
cl 2/term 24 | loss    7 (100.0% of   7) | wins    0 | acc 1.000
cl 2/term 20 | loss    6 (54.5% of  11) | wins    5 | acc 1.000
cl 2/term 37 | loss    5 (100.0% of   5) | wins    0 | acc 1.000
cl 2/term 22 | loss    5 (9.3% of  54) | wins   49 | acc 1.000
cl 1/term 2 | loss    5 (41.7% of  12) | wins    7 | acc 0.667
cl 2/term 36 | loss    5 (83.3% of   6) | wins    1 | acc 0.667
cl 1/term 9 | loss    4 (100.0% of   4) | wins    0 | acc 0.750
cl 1/term 4 | loss    4 (5.5% of  73) | wins   69 | acc 0.945
cl 2/term 41 | loss    4 (80.0% of   5) | wins    1 | acc 0.800
cl 0/term 10 | loss    3 (60.0% of   5) | wins    2 | acc 0.600
cl 2/term 26 | loss    3 (75.0% of   4) | wins    1 | acc 0.750
cl 2/term 47 | loss    3 (100.0% of   3) | wins    0 | acc 1.000
cl 2/term 35 | loss    2 (50.0% of   4) | wins    2 | acc 1.000
cl 2/term 9 | loss    2 (40.0% of   5) | wins    3 | acc 0.800
cl 0/term 1 | loss    2 (4.7% of  43) | wins   41 | acc 0.953
cl 2/term 1 | loss    1 (10.0% of  10) | wins    9 | acc 0.900
cl 1/term 14 | loss    1 (100.0% of   1) | wins    0 | acc 1.000
cl 2/term 40 | loss    1 (100.0% of   1) | wins    0 | acc 1.000
cl 0/term 7 | loss    1 (25.0% of   4) | wins    3 | acc 0.750
cl 2/term 17 | loss    1 (100.0% of   1) | wins    0 | acc 1.000
cl 2/term 13 | loss    0 (0.0% of   1) | wins    1 | acc 1.000
cl 2/term 8 | loss    0 (0.0% of   5) | wins    5 | acc 1.000
cl 2/term 4 | loss    0 (0.0% of   3) | wins    3 | acc 1.000
cl 1/term 13 | loss    0 (0.0% of   1) | wins    1 | acc 1.000
cl 1/term 8 | loss    0 (0.0% of   1) | wins    1 | acc 1.000
cl 2/term 18 | loss    0 (0.0% of   2) | wins    2 | acc 1.000
cl 2/term 5 | loss    0 (0.0% of   2) | wins    2 | acc 1.000
cl 2/term 32 | loss    0 (0.0% of   1) | wins    1 | acc 1.000
cl 1/term 6 | loss    0 (0.0% of   2) | wins    2 | acc 1.000
cl 0/term 5 | loss    0 (0.0% of  22) | wins   22 | acc 1.000
cl 2/term 12 | loss    0 (0.0% of   7) | wins    7 | acc 1.000
cl 0/term 2 | loss    0 (0.0% of  23) | wins   23 | acc 0.957
cl 2/term 19 | loss    0 (0.0% of   3) | wins    3 | acc 1.000
cl 0/term 6 | loss    0 (0.0% of   4) | wins    4 | acc 1.000
cl 0/term 0 | loss    0 (0.0% of  68) | wins   68 | acc 1.000
cl 2/term 23 | loss    0 (0.0% of  18) | wins   18 | acc 1.000
cl 2/term 6 | loss    0 (0.0% of  16) | wins   16 | acc 1.000
cl 0/term 4 | loss    0 (0.0% of  25) | wins   25 | acc 0.720
cl 2/term 0 | loss    0 (0.0% of  16) | wins   16 | acc 1.000
cl 2/term 2 | loss    0 (0.0% of  22) | wins   22 | acc 1.000
cl 1/term 7 | loss    0 (0.0% of   1) | wins    1 | acc 1.000
cl 2/term 7 | loss    0 (0.0% of   2) | wins    2 | acc 1.000
cl 2/term 3 | loss    0 (0.0% of   1) | wins    1 | acc 1.000
cl 1/term 1 | loss    0 (0.0% of   3) | wins    3 | acc 1.000
"""

RAW_WINS = """
cl 1/term 4 | wins   69 | loss    4 | acc 0.945
cl 0/term 0 | wins   68 | loss    0 | acc 1.000
cl 2/term 22 | wins   49 | loss    5 | acc 1.000
cl 1/term 11 | wins   42 | loss   46 | acc 0.830
cl 0/term 1 | wins   41 | loss    2 | acc 0.953
cl 1/term 12 | wins   36 | loss  112 | acc 0.595
cl 0/term 4 | wins   25 | loss    0 | acc 0.720
cl 0/term 2 | wins   23 | loss    0 | acc 0.957
cl 0/term 5 | wins   22 | loss    0 | acc 1.000
cl 2/term 2 | wins   22 | loss    0 | acc 1.000
cl 2/term 23 | wins   18 | loss    0 | acc 1.000
cl 2/term 0 | wins   16 | loss    0 | acc 1.000
cl 2/term 6 | wins   16 | loss    0 | acc 1.000
cl 2/term 14 | wins   15 | loss   62 | acc 0.623
cl 0/term 9 | wins   15 | loss   60 | acc 0.640
cl 2/term 11 | wins   14 | loss   58 | acc 0.986
cl 1/term 10 | wins   12 | loss   20 | acc 0.594
cl 2/term 21 | wins    9 | loss   53 | acc 0.742
cl 2/term 1 | wins    9 | loss    1 | acc 0.900
cl 1/term 2 | wins    7 | loss    5 | acc 0.667
cl 2/term 12 | wins    7 | loss    0 | acc 1.000
cl 2/term 38 | wins    7 | loss   80 | acc 0.977
cl 2/term 15 | wins    6 | loss   17 | acc 0.696
cl 2/term 42 | wins    5 | loss    9 | acc 1.000
cl 2/term 8 | wins    5 | loss    0 | acc 1.000
cl 2/term 20 | wins    5 | loss    6 | acc 1.000
cl 2/term 27 | wins    4 | loss   17 | acc 0.857
cl 2/term 16 | wins    4 | loss   79 | acc 0.964
cl 0/term 6 | wins    4 | loss    0 | acc 1.000
cl 1/term 3 | wins    3 | loss  124 | acc 0.535
cl 0/term 7 | wins    3 | loss    1 | acc 0.750
cl 2/term 19 | wins    3 | loss    0 | acc 1.000
cl 2/term 9 | wins    3 | loss    2 | acc 0.800
cl 2/term 4 | wins    3 | loss    0 | acc 1.000
cl 1/term 1 | wins    3 | loss    0 | acc 1.000
cl 0/term 8 | wins    3 | loss  136 | acc 0.950
cl 1/term 6 | wins    2 | loss    0 | acc 1.000
cl 2/term 44 | wins    2 | loss    9 | acc 0.909
cl 2/term 5 | wins    2 | loss    0 | acc 1.000
cl 2/term 18 | wins    2 | loss    0 | acc 1.000
cl 1/term 0 | wins    2 | loss  405 | acc 0.511
cl 0/term 10 | wins    2 | loss    3 | acc 0.600
cl 2/term 35 | wins    2 | loss    2 | acc 1.000
cl 2/term 7 | wins    2 | loss    0 | acc 1.000
cl 2/term 28 | wins    2 | loss   33 | acc 1.000
cl 2/term 41 | wins    1 | loss    4 | acc 0.800
cl 2/term 46 | wins    1 | loss    8 | acc 1.000
cl 2/term 36 | wins    1 | loss    5 | acc 0.667
cl 1/term 13 | wins    1 | loss    0 | acc 1.000
cl 2/term 48 | wins    1 | loss   40 | acc 1.000
cl 1/term 8 | wins    1 | loss    0 | acc 1.000
cl 2/term 3 | wins    1 | loss    0 | acc 1.000
cl 2/term 13 | wins    1 | loss    0 | acc 1.000
cl 2/term 45 | wins    1 | loss    7 | acc 0.750
cl 2/term 32 | wins    1 | loss    0 | acc 1.000
cl 1/term 7 | wins    1 | loss    0 | acc 1.000
cl 2/term 26 | wins    1 | loss    3 | acc 0.750
cl 2/term 17 | wins    0 | loss    1 | acc 1.000
cl 2/term 10 | wins    0 | loss   15 | acc 0.933
cl 2/term 31 | wins    0 | loss   22 | acc 1.000
cl 2/term 43 | wins    0 | loss   90 | acc 0.511
cl 2/term 40 | wins    0 | loss    1 | acc 1.000
cl 2/term 47 | wins    0 | loss    3 | acc 1.000
cl 2/term 49 | wins    0 | loss    7 | acc 1.000
cl 2/term 37 | wins    0 | loss    5 | acc 1.000
cl 2/term 25 | wins    0 | loss   67 | acc 0.522
cl 2/term 39 | wins    0 | loss    8 | acc 0.875
cl 2/term 24 | wins    0 | loss    7 | acc 1.000
cl 2/term 30 | wins    0 | loss   11 | acc 0.818
cl 0/term 3 | wins    0 | loss   13 | acc 0.538
cl 1/term 9 | wins    0 | loss    4 | acc 0.750
cl 1/term 5 | wins    0 | loss   95 | acc 0.589
cl 2/term 34 | wins    0 | loss   16 | acc 1.000
cl 2/term 29 | wins    0 | loss   14 | acc 0.929
cl 2/term 33 | wins    0 | loss    7 | acc 0.714
cl 1/term 14 | wins    0 | loss    1 | acc 1.000
"""

# histogram + pruning still small enough to keep literal
MATCH_HIST = {1: 5, 2: 149, 3: 165, 4: 87, 5: 76, 6: 108, 7: 31, 8: 5}
SHARE_MULTI_MATCH = 99.20  # %
COLLISION_RATIO = dict(intra=47.5, inter=52.5)

PRUNING = [
    dict(thr=0, kept=57, total=76, val_acc=0.909, test_acc=0.812),
    dict(thr=1, kept=45, total=76, val_acc=0.904, test_acc=0.818),
    dict(thr=2, kept=36, total=76, val_acc=0.896, test_acc=0.807),
    dict(thr=3, kept=29, total=76, val_acc=0.877, test_acc=0.793),
    dict(thr=4, kept=26, total=76, val_acc=0.863, test_acc=0.780),
    dict(thr=5, kept=23, total=76, val_acc=0.855, test_acc=0.769),
]

# ─────────────────────────────────────────────────────────────────────────────
# simple regex parsers
# ─────────────────────────────────────────────────────────────────────────────


def _parse(lines: str, want_losses: bool) -> pd.DataFrame:
    """Return DataFrame with cls, term, wins, loss, acc."""
    pat_loss = re.compile(
        r"cl (\d+)/term (\d+) \| loss +(\d+) .*?\| wins +(\d+) \| acc ([0-9.]+)"
    )
    pat_win = re.compile(
        r"cl (\d+)/term (\d+) \| wins +(\d+) \| loss +(\d+) \| acc ([0-9.]+)"
    )
    rows = []
    for m in pat_loss.finditer(lines) if want_losses else pat_win.finditer(lines):
        cls, term, x1, x2, acc = m.groups()
        loss, wins = (int(x1), int(x2)) if want_losses else (int(x2), int(x1))
        rows.append(
            dict(
                cls=int(cls),
                term=int(term),
                wins=wins,
                loss=loss,
                acc=float(acc),
            )
        )
    return pd.DataFrame(rows)


DF_LOSSES = _parse(RAW_LOSSES, True)
DF_WINS = _parse(RAW_WINS, False)
DF_ALL = (
    pd.concat([DF_LOSSES, DF_WINS])
    .drop_duplicates(subset=["cls", "term"])
    .reset_index(drop=True)
)
DF_ALL["label"] = DF_ALL.apply(lambda r: f"{r.cls}.{r.term}", axis=1)

# fewest winners – take 20 smallest wins
DF_FEWEST = DF_ALL.sort_values("wins").head(20)

# ─────────────────────────────────────────────────────────────────────────────
# plotting setup
# ─────────────────────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid")
OUTDIR = Path("./cfire_lab_experiments/rule-overlap-analysis")
OUTDIR.mkdir(parents=True, exist_ok=True)


def save(fig, name: str):
    fig.tight_layout()
    fig.savefig(OUTDIR / f"{name}.png", dpi=300)
    plt.close(fig)


# 1) histogram of match counts (unchanged)
fig, ax = plt.subplots()
ser = pd.Series(MATCH_HIST)
sns.barplot(x=ser.index, y=ser.values, ax=ax, color=sns.color_palette()[0])
ax.set_xlabel("# matched clauses")
ax.set_ylabel("samples")
ax.set_title("Histogram of clause matches per sample")
ax.text(
    0.97,
    0.97,
    f"≥2 matches: {SHARE_MULTI_MATCH:.2f}%",
    ha="right",
    va="top",
    transform=ax.transAxes,
    fontsize="small",
)
save(fig, "hist_matches")

# 2) wins vs losses (linear)
fig, ax = plt.subplots()
sns.scatterplot(data=DF_ALL, x="wins", y="loss", ax=ax)
ax.set_xlabel("wins")
ax.set_ylabel("losses")
ax.set_title("Wins vs losses (linear)")
save(fig, "wins_vs_losses")

# 3) wins vs losses (log/log)
fig, ax = plt.subplots()
sns.scatterplot(data=DF_ALL, x="wins", y="loss", ax=ax)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("wins (log)")
ax.set_ylabel("losses (log)")
ax.set_title("Wins vs losses (log–log scale)")
save(fig, "wins_vs_losses_log")

# 4) pruning – rules kept (same as v2, y-min=0)
df_prune = pd.DataFrame(PRUNING)
df_prune["rel_rules"] = df_prune["kept"] / df_prune["total"].iloc[0]

fig, ax = plt.subplots()
sns.lineplot(data=df_prune, x="thr", y="kept", marker="o", ax=ax)
ax.set_xlabel("wins threshold")
ax.set_ylabel("rules kept")
ax.set_ylim(bottom=0)
ax.set_title("Pruned rule count vs wins threshold")
save(fig, "prune_rules_kept")

# 5) pruning – accuracy with secondary axis for rel_rules
fig, ax1 = plt.subplots()
sns.lineplot(
    data=df_prune, x="thr", y="val_acc", marker="o", label="validation", ax=ax1
)
sns.lineplot(data=df_prune, x="thr", y="test_acc", marker="s", label="test", ax=ax1)
ax1.set_xlabel("wins threshold")
ax1.set_ylabel("accuracy")
ax1.set_title("Accuracy & remaining rules vs threshold")

ax2 = ax1.twinx()
sns.lineplot(
    data=df_prune,
    x="thr",
    y="rel_rules",
    marker="^",
    color="gray",
    linestyle="--",
    label="relative rules",
    ax=ax2,
)
ax2.set_ylabel("relative rules kept")

# Combine legends from both axes
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper right")
save(fig, "prune_accuracy_and_rules")

# 6) accuracy vs relative rule count (+ full CFIRE point at 1.0)
df_rel = df_prune[["rel_rules", "val_acc", "test_acc"]].copy()
df_rel = pd.concat(
    [
        df_rel,
        pd.DataFrame([{"rel_rules": 1.0, "val_acc": 0.909, "test_acc": 0.815}]),
    ],
    ignore_index=True,
).sort_values("rel_rules")

fig, ax = plt.subplots()
sns.lineplot(
    data=df_rel, x="rel_rules", y="val_acc", marker="o", label="validation", ax=ax
)
sns.lineplot(data=df_rel, x="rel_rules", y="test_acc", marker="s", label="test", ax=ax)
ax.set_xlabel("relative rules kept")
ax.set_ylabel("accuracy")
ax.set_xlim(0, 1)
ax.set_title("Accuracy vs relative rule count")
ax.legend()
save(fig, "accuracy_vs_rel_rules")

# 7) collision pie
fig, ax = plt.subplots()
ax.pie(
    [COLLISION_RATIO["intra"], COLLISION_RATIO["inter"]],
    labels=["intra-class", "inter-class"],
    autopct="%1.1f%%",
    colors=sns.color_palette("pastel"),
)
ax.set_title("Tie-breaker collision type")
save(fig, "collision_pie")

print(f"Plots written to {OUTDIR.resolve()}")
