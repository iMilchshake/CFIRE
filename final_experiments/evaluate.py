import numpy as np

from cfire.cfire_module import CFIRE
from .metrics import get_rule_size, get_literal_count, get_unique_literal_count, max_offdiag_iou, mean_offdiag_iou, get_class_iou_matrix_mc
from sklearn.metrics import precision_recall_fscore_support

def evaluate_cfire(
    cfire: CFIRE,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_val_model_pred: np.ndarray,
    y_test_model_pred: np.ndarray,
) -> dict:

    y_val_cfire = cfire(X_val)
    y_test_cfire = cfire(X_test)

    val_acc = float((y_val_cfire == y_val_model_pred).mean())
    test_acc = float((y_test_cfire == y_test_model_pred).mean())

    val_prec, val_rec, val_f1, _ = precision_recall_fscore_support(
        y_val_model_pred,
        y_val_cfire,
        average="macro", # un‑weighted mean over classes
        zero_division=0, # 0 if undefined
    )

    test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(
        y_test_model_pred,
        y_test_cfire,
        average="macro",
        zero_division=0,
    )

    rule_size = get_rule_size(cfire.dnf.rules)
    literal_count = get_literal_count(cfire.dnf.rules)
    unique_literal_count =  get_unique_literal_count(cfire.dnf.rules)
    class_iou_matrix = get_class_iou_matrix_mc(cfire.dnf.rules)
    max_iou = max_offdiag_iou(class_iou_matrix)
    mean_iou = mean_offdiag_iou(class_iou_matrix)

    # TODO: add more metrics, e.g:
    #   - are there any more cfire metrics from papaer we might want to include?
    #   - pruning "metrics"

    return {
        "val_acc": val_acc,
        "test_acc": test_acc,
        "val_precision": val_prec,
        "val_recall": val_rec,
        "val_f1": val_f1,
        "test_precision": test_prec,
        "test_recall": test_rec,
        "test_f1": test_f1,

        "rule_size": rule_size,
        "literal_count": literal_count,
        "unique_literal_count": unique_literal_count,
        "max_iou": max_iou,
        "mean_iou": mean_iou,
    }
