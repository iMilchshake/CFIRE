from pathlib import Path
import random
import csv
import numpy as np
import torch

from cfire.cfire_module import CFIRE
from cfire.nodeselection import _comp_greedy_cover, _comp_ilp_optimal
from cfire.util import __preprocess_explanations
from cfire_lab_experiments.util import loader_to_tensor
from cfire_lab_experiments.boxOverlaps import (
    compute_domain_bounds,
    compute_overlap_matrix,
    compute_class_root_volumes,
)

from lxg.datasets import NumpyRandomSeed, TorchRandomSeed
import lxg.datasets as datasets
from lxg.models import make_ff
from lxg.util import restore_checkpoint

FREQ_THRESHOLDS = [0.005, 0.01, 0.02, 0.05, 0.1]
BINARIZATION_THRESHOLDS = [0.005, 0.01, 0.02, 0.05, 0.1]
MAX_DT_DEPTHS = [4, 5, 6, 7, 8, 9, 10]
SEED = 42


def format_iou(iou: np.ndarray) -> str:
    """Return a neatly aligned IoU matrix (string)."""
    n = iou.shape[0]
    header = "           " + "  ".join(f"C{j:>2}" for j in range(n))
    rows = [
        f"Class {i:<2}  " + "  ".join(f"{iou[i, j]:6.3f}" for j in range(n))
        for i in range(n)
    ]
    return "\n".join([header] + rows)


def main():
    train_loader, test_loader, val_loader, n_dim, n_classes = datasets.get_abalone()
    X_train, y_train = loader_to_tensor(train_loader)
    X_val, y_val = loader_to_tensor(val_loader)
    X_test, y_test = loader_to_tensor(test_loader)

    model = make_ff([n_dim, 128, 128, n_classes], torch.nn.ReLU)
    restore_checkpoint(Path("./models/tmp.ckpt"), model, train=False)

    y_val_model_pred = model.predict_batch(X_val).numpy()
    y_test_model_pred = model.predict_batch(X_test).numpy()

    bounds = compute_domain_bounds(X_train)

    exp_dir = Path("./experiments")
    exp_dir.mkdir(exist_ok=True)

    log_txt  = exp_dir / "hparam_results_ilp_vs_greedy.txt"
    log_csv  = exp_dir / "hparam_results_ilp_vs_greedy.csv"

    log_txt.write_text("")   # truncate
    with log_csv.open("w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "freq_threshold", "binarization_threshold", "max_dt_depth", "composition_strategy",
            "rule_size", "mean_offdiag_iou", "max_offdiag_iou",
            "summed_root_volume", "domain_root_volume",
            "class_root_volumes", "val_accuracy", "test_accuracy"
        ])

    random.seed(SEED)
    with log_txt.open("a") as ftxt, log_csv.open("a", newline="") as fcsv:
        writer = csv.writer(fcsv)

        for freq in FREQ_THRESHOLDS:
            for bin_thresh in BINARIZATION_THRESHOLDS:
                for max_dt_depth in MAX_DT_DEPTHS:
                    for strategy, strategy_name in [
                        (_comp_greedy_cover, "greedy_cover"),
                        (_comp_ilp_optimal, "ilp_optimal"),
                    ]:
                        ftxt.write(f"-- freq={freq} | binarization_threshold={bin_thresh} | max_dt_depth={max_dt_depth} | strategy={strategy_name} ------------------------------\n")

                        binarize = lambda x, thr=bin_thresh: (
                                __preprocess_explanations(x, filtering=thr) > 0
                        )

                        # fit CFIRE
                        with NumpyRandomSeed(SEED), TorchRandomSeed(SEED):
                            cfire = CFIRE(
                                localexplainer_fn=lambda *_: torch.load("./models/explanations.pt"),
                                inference_fn=model.predict_batch_softmax,
                                expl_binarization_fn=binarize,
                                frequency_threshold=freq,
                                composition_strategy=strategy,
                                max_dt_depth=max_dt_depth
                            )
                            cfire.fit(X_val.numpy(), y_val_model_pred)

                        # metrics
                        iou = compute_overlap_matrix(cfire.dnf, bounds, metric="iou")
                        off_diag = iou[np.triu_indices_from(iou, k=1)]
                        mean_iou = off_diag.mean() if off_diag.size else 0.0
                        max_iou = off_diag.max() if off_diag.size else 0.0

                        cls_vols, summed_root_vols, domain_len_vol = (
                            compute_class_root_volumes(cfire.dnf, bounds)
                        )

                        rule_sz = sum(len(c) for c in cfire.dnf.rules)

                        y_val_cfire_pred = cfire(X_val)
                        y_test_cfire_pred = cfire(X_test)
                        val_acc = np.mean(y_val_model_pred == y_val_cfire_pred)
                        test_acc = np.mean(y_test_model_pred == y_test_cfire_pred)

                        ftxt.write(f"rule_size              : {rule_sz}\n")
                        ftxt.write(f"mean_offdiag_iou       : {mean_iou:.3f}\n")
                        ftxt.write(f"max_offdiag_iou        : {max_iou:.3f}\n")
                        ftxt.write(f"summed_root_volume     : {summed_root_vols:.3f}\n")
                        ftxt.write(f"domain_root_volume     : {domain_len_vol:.3f}\n")
                        ftxt.write(f"class_root_volumes     : {np.round(cls_vols, 3)}\n")
                        ftxt.write(f"val_acc                : {val_acc:.3f}\n")
                        ftxt.write(f"test_acc               : {test_acc:.3f}\n\n")
                        ftxt.write("IoU matrix\n")
                        ftxt.write(format_iou(iou) + "\n\n")

                        # also put it in table for plottings
                        writer.writerow([
                            freq, bin_thresh, max_dt_depth, strategy_name, rule_sz,
                            f"{mean_iou:.6f}", f"{max_iou:.6f}",
                            f"{summed_root_vols:.6f}",
                            f"{domain_len_vol:.6f}",
                            ";".join(f"{v:.6f}" for v in cls_vols),
                            f"{val_acc:.6f}",
                            f"{test_acc:.6f}"
                        ])

    print(f"Text log  → {log_txt.resolve()}")
    print(f"CSV table → {log_csv.resolve()}")


if __name__ == "__main__":
    main()
