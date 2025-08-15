import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from cfire.cfire_module import CFIRE
from .metrics import (
    get_rule_size,
    get_literal_count,
    get_unique_literal_count,
    max_offdiag_iou,
    mean_offdiag_iou,
    get_class_iou_matrix_mc,
    build_coverage_matrices,
    mean_coverage_ratio,
    mean_single_coverage_ratio,
    mean_nodes_per_sample,
    mean_duplicate_nodes_ratio,
)


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

    # quantify the lack of rules for an entire class / lack of predictions for individual samples
    results["missing_class_rules"] = sum(len(class_rules) == 0 for class_rules in cfire.dnf.rules)
    results["missing_pred_val"] = np.sum(y_val_cfire == -1) / len(y_val_cfire)
    results["missing_pred_test"] = np.sum(y_test_cfire != -1) / len(y_test_cfire)

    # analyze input to set covering algorithm
    coverage_matricies = build_coverage_matrices(cfire.frequent_nodes)
    results["mean_coverage_ratio"] = mean_coverage_ratio(coverage_matricies)
    results["mean_single_coverage_ratio"] = mean_single_coverage_ratio(coverage_matricies)
    results["mean_nodes_per_sample"] = mean_nodes_per_sample(coverage_matricies)
    results["mean_duplicate_nodes_ratio"] = mean_duplicate_nodes_ratio(coverage_matricies)
    results["total_frequent_node_count"] = sum(cov_mat.shape[1] for cov_mat in coverage_matricies)

    # TODO: add more metrics, e.g:
    #   - are there any more cfire metrics from paper we might want to include?
    #   - pruning "metrics"

    return results
