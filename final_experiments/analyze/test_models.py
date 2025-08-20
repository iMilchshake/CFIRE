import pickle
from pathlib import Path
import numpy as np

tasks=[
    "abalone",
    "breastw",
    "spambase",
    # "heloc",
    "beans",
    "ionosphere",
    "breastcancer",
    "btsc",
    "spf",
    "wine",
    "diggle",
    "iris",
    "vehicle",
    "autouniv",
]

print("# Test accuracies")
for task in tasks:
    dir = Path("./data/cfire/") / task
    out_dir = dir / "outputs"

    accuracies = []
    for file in out_dir.iterdir():
        with file.open("rb") as f:
            obj = pickle.load(f)
            accuracies.append(obj["accuracy"])
    print(f"{task:<12} {np.mean(accuracies):.2f}±{np.std(accuracies):.2f}")
