from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SUMMARY_IN = Path("./cfire_lab_experiments/rule-overlap-analysis/summary.json")
PLOTS_OUT = Path("./cfire_lab_experiments/rule-overlap-analysis/")
PLOTS_OUT.mkdir(parents=True, exist_ok=True)
with open(SUMMARY_IN) as f:
    S = json.load(f)

df_all = pd.DataFrame(S["clause_stats"])
df_all["label"] = df_all.apply(lambda r: f"{r.cls}.{r.term}", axis=1)
match_hist = {int(k): v for k, v in S["match_hist"].items()}
share_multi_match = S["share_multi_match"]
collision_ratio = S["collision_ratio"]
pruning = S["pruning"]
df_fewest = df_all.sort_values("wins").head(20)

sns.set_theme(style="whitegrid")


def save(fig, name: str):
    fig.tight_layout()
    fig.savefig(PLOTS_OUT / f"{name}.png", dpi=300)
    plt.close(fig)


# 1) histogram of match counts
fig, ax = plt.subplots()
ser = pd.Series(match_hist).sort_index()
sns.barplot(x=ser.index, y=ser.values, ax=ax, color=sns.color_palette()[0])
ax.set_xlabel("# matched clauses")
ax.set_ylabel("samples")
ax.set_title("Histogram of clause matches per sample")
ax.text(
    0.97,
    0.97,
    f"≥2 matches: {share_multi_match:.2f}%",
    ha="right",
    va="top",
    transform=ax.transAxes,
    fontsize="small",
)
save(fig, "hist_matches")

# 2) wins vs losses (linear)
fig, ax = plt.subplots()
sns.scatterplot(data=df_all, x="wins", y="loss", ax=ax)
ax.set_xlabel("wins")
ax.set_ylabel("losses")
ax.set_title("Wins vs losses (linear)")
save(fig, "wins_vs_losses")

# 3) wins vs losses (log/log)
fig, ax = plt.subplots()
sns.scatterplot(data=df_all, x="wins", y="loss", ax=ax)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("wins (log)")
ax.set_ylabel("losses (log)")
ax.set_title("Wins vs losses (log–log scale)")
save(fig, "wins_vs_losses_log")

# 4) pruning – rules kept
df_prune = pd.DataFrame(pruning)
df_prune["rel_rules"] = df_prune["kept"] / df_prune["total"].iloc[0]

fig, ax = plt.subplots()
sns.lineplot(data=df_prune, x="thr", y="kept", marker="o", ax=ax)
ax.set_xlabel("wins threshold")
ax.set_ylabel("rules kept")
ax.set_ylim(bottom=0)
ax.set_title("Pruned rule count vs wins threshold")
save(fig, "prune_rules_kept")

# 5) pruning – accuracy (left) & remaining rules (right) side-by-side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

# left: accuracies
sns.lineplot(
    data=df_prune, x="thr", y="val_acc", marker="o", label="validation", ax=ax1
)
sns.lineplot(data=df_prune, x="thr", y="test_acc", marker="s", label="test", ax=ax1)
ax1.set_xlabel("wins threshold")
ax1.set_ylabel("accuracy")
ax1.set_title("Accuracy vs threshold")
ax1.legend()

# right: relative rules kept
sns.lineplot(
    data=df_prune,
    x="thr",
    y="rel_rules",
    marker="^",
    color="gray",
    linestyle="--",
    ax=ax2,
)
ax2.set_xlabel("wins threshold")
ax2.set_ylabel("relative rules kept")
ax2.set_title("Rules kept vs threshold")

save(fig, "prune_accuracy_and_rules")

# 6) accuracy vs relative rule count (+ full CFIRE at 1.0)
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
    [collision_ratio["intra"], collision_ratio["inter"]],
    labels=["intra-class", "inter-class"],
    autopct="%1.1f%%",
    colors=sns.color_palette("pastel"),
)
ax.set_title("Tie-breaker collision type")
save(fig, "collision_pie")

# 8) same-class collision % vs wins threshold
fig, ax = plt.subplots()
sns.lineplot(data=df_prune, x="thr", y="intra_coll_ratio", marker="o", ax=ax)
ax.set_xlabel("wins threshold")
ax.set_ylabel("intra class collision")
ax.set_title("Intra class collision vs wins threshold")
save(fig, "intra_class_collision_vs_threshold")

print(f"Plots written to {PLOTS_OUT.resolve()}")
