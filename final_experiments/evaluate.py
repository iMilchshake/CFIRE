import numpy as np

from cfire.cfire_module import CFIRE
from .metrics import get_rule_size, get_literal_count, get_unique_literal_count, get_class_iou_matrix, mean_offdiag_iou, \
    max_offdiag_iou, espresso_reformulated_rules


def evaluate_cfire(
    cfire: CFIRE,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_val_model_pred: np.ndarray,
    y_test_model_pred: np.ndarray,
) -> dict:

    val_acc = (cfire(X_val) == y_val_model_pred).mean()
    test_acc = (cfire(X_test) == y_test_model_pred).mean()
    rule_size = get_rule_size(cfire.dnf.rules)
    literal_count = get_literal_count(cfire.dnf.rules)
    unique_literal_count = get_unique_literal_count(cfire.dnf.rules)
    IoU_matrix = get_class_iou_matrix(cfire.dnf.rules)
    mean_iou = mean_offdiag_iou(IoU_matrix)
    max_iou = max_offdiag_iou(IoU_matrix)
    espresso_rules = espresso_reformulated_rules(cfire.dnf.rules)
    literal_count = get_literal_count(espresso_rules)
    unique_literal_count = get_unique_literal_count(espresso_rules)


    # TODO: add more metrics, e.g:
    #   - various ruben metrics
    #   - as cfire paper shows f1/prec/recall we should also include this for general performance
    #   - pruning "metrics"

    return {
        "val_acc": val_acc,
        "test_acc": test_acc,
        "rule_size": rule_size,
        "literal_count": literal_count,
        "unique_literal_count": unique_literal_count,
        "mean_iou": mean_iou,
        "max_iou": max_iou,
    }
