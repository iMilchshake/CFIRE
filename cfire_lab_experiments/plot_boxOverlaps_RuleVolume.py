import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load CSV data
df = pd.read_csv("./experiments/hparam_results.csv")

# Pivot for IoU heatmap
iou_pivot = df.pivot(
    index="binarization_threshold",
    columns="freq_threshold",
    values="mean_offdiag_iou"
)

plt.figure(figsize=(8, 6))
sns.heatmap(iou_pivot, annot=True, fmt=".3f", cmap="viridis")
plt.gca().invert_yaxis()
plt.title("Mean Off-Diagonal IoU Heatmap")
plt.xlabel("Frequency Threshold")
plt.ylabel("Binarization Threshold")
plt.tight_layout()
plt.savefig("./experiments/mean_offdiag_iou_heatmap.png", dpi=300)
plt.show()

# Pivot for summed root volume heatmap
volume_pivot = df.pivot(
    index="binarization_threshold",
    columns="freq_threshold",
    values="summed_root_volume"
)

plt.figure(figsize=(8, 6))
sns.heatmap(volume_pivot, annot=True, fmt=".1f", cmap="magma")
plt.gca().invert_yaxis()
plt.title("Summed Root Volume Heatmap")
plt.xlabel("Frequency Threshold")
plt.ylabel("Binarization Threshold")
plt.tight_layout()
plt.savefig("./experiments/summed_root_volume_heatmap.png", dpi=300)
plt.show()

# Pivot for max IoU heatmap
max_iou_pivot = df.pivot(
    index="binarization_threshold",
    columns="freq_threshold",
    values="max_offdiag_iou"
)

plt.figure(figsize=(8, 6))
sns.heatmap(max_iou_pivot, annot=True, fmt=".3f", cmap="plasma")
plt.gca().invert_yaxis()
plt.title("Max Off-Diagonal IoU Heatmap")
plt.xlabel("Frequency Threshold")
plt.ylabel("Binarization Threshold")
plt.tight_layout()
plt.savefig("./experiments/max_offdiag_iou_heatmap.png", dpi=300)
plt.show()
