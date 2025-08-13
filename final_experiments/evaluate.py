import numpy as np

from cfire.cfire_module import CFIRE
from .metrics import get_rule_size, get_literal_count, get_unique_literal_count, max_offdiag_iou, mean_offdiag_iou, get_class_iou_matrix_mc
from sklearn.metrics import precision_recall_fscore_support, f1_score, accuracy_score


def evaluate_cfire(
        cfire: CFIRE,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_val_model: np.ndarray,
        y_test_model: np.ndarray,
) -> dict:
    results = {}

    y_val_cfire = cfire(X_val)
    y_test_cfire = cfire(X_test)

    datasets = {
        "val": (y_val_model, y_val_cfire),
        "test": (y_test_model, y_test_cfire),
    }
    for prefix, (y_true, y_pred) in datasets.items():
        results[f"{prefix}_acc"] = accuracy_score(y_true, y_pred)
        for avg_type in ["macro", "weighted"]:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average=avg_type, zero_division=0
            )
            results[f"{prefix}_precision_{avg_type}"] = precision
            results[f"{prefix}_recall_{avg_type}"] = recall
            results[f"{prefix}_f1_{avg_type}"] = f1

    rules = cfire.dnf.rules
    results["rule_count"] = len(rules)
    results["rule_size"] = get_rule_size(rules)
    results["literal_count"] = get_literal_count(rules)
    results["unique_literal_count"] = get_unique_literal_count(rules)

    class_iou_matrix = get_class_iou_matrix_mc(rules)
    results["max_iou"] = max_offdiag_iou(class_iou_matrix)
    results["mean_iou"] = mean_offdiag_iou(class_iou_matrix)

    results["t_explanations"] = cfire._compute_times['_calc_explanations']+cfire._compute_times['expl_binarization_fn']
    results["t_rule_candidates"] = cfire._compute_times["_calculate_rule_candidates"]
    results["t_compose_rules"] = cfire._compute_times['_compose_rule_model']

    # TODO: add more metrics, e.g:
    #   - are there any more cfire metrics from paper we might want to include?
    #   - pruning "metrics"

    return results
