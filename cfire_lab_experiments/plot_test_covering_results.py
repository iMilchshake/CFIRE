import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load CSV
results = pd.read_csv("./cfire_eval_results.csv")

results["method"] = results.apply(
    lambda r: (f"α={r.alpha:.2f}|dup={'T' if r.dedup else 'F'}")
    if r["composition"] == "set_packing"
    else r["composition"],
    axis=1
)

# Set style
sns.set(style="whitegrid")

# Plot accuracy
plt.figure()
sns.boxplot(data=results, x="method", y="val_acc")
plt.xticks(rotation=45, ha="right")
plt.title("Validation Accuracy by Composition Method")
plt.tight_layout()
plt.savefig("val_acc_by_composition.png")

plt.figure()
sns.boxplot(data=results, x="method", y="test_acc")
plt.xticks(rotation=45, ha="right")
plt.title("Test Accuracy by Composition Method")
plt.tight_layout()
plt.savefig("test_acc_by_composition.png")

# Plot rule metrics
plt.figure()
sns.boxplot(data=results, x="method", y="rule_size")
plt.xticks(rotation=45, ha="right")
plt.title("Rule Size by Composition Method")
plt.tight_layout()
plt.savefig("rule_size_by_composition.png")

plt.figure()
sns.boxplot(data=results, x="method", y="literal_count")
plt.xticks(rotation=45, ha="right")
plt.title("Literal Count by Composition Method")
plt.tight_layout()
plt.savefig("literal_count_by_composition.png")
