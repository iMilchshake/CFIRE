import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load CSV
results = pd.read_csv("./cfire_eval_results.csv")

# Set style
sns.set(style="whitegrid")

# Plot accuracy
plt.figure()
sns.boxplot(data=results, x="composition", y="val_acc")
plt.title("Validation Accuracy by Composition Method")
plt.savefig("val_acc_by_composition.png")

plt.figure()
sns.boxplot(data=results, x="composition", y="test_acc")
plt.title("Test Accuracy by Composition Method")
plt.savefig("test_acc_by_composition.png")

# Plot rule metrics
plt.figure()
sns.boxplot(data=results, x="composition", y="rule_size")
plt.title("Rule Size by Composition Method")
plt.savefig("rule_size_by_composition.png")

plt.figure()
sns.boxplot(data=results, x="composition", y="literal_count")
plt.title("Literal Count by Composition Method")
plt.savefig("literal_count_by_composition.png")
