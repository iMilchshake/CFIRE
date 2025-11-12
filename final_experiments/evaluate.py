import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from cfire.cfire_module import CFIRE
from lxg.models import DNFClassifier
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
    normalize_explanations,
    attribution_variance,
    sparsity,
    class_separation_in_attribution_space,
    mean_active_features_per_sample,
    mean_active_features_ratio,
    mean_feature_activation_ratio,
    features_inactive_ratio,
    mean_feature_class_specificity,
    mean_within_class_jaccard,
    mean_across_class_jaccard,
    class_separation_score,
    all_features_active_ratio,
    all_features_inactive_ratio,
    mean_max_attribution,
    mean_min_attribution,
    max_max_attribution,
    min_min_attribution,
    mean_mean_attribution, get_rule_size_nonempty,
)


def get_dnf_rule_metrics(
    dnf: DNFClassifier,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_val_model: np.ndarray,
    y_test_model: np.ndarray,
):
    results = {}

    # get dnf predictions
    y_val_cfire = dnf(X_val)
    y_test_cfire = dnf(X_test)

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

    results["n_classes"] = len(dnf.rules)
    results["rule_size_nonempty"] = get_rule_size_nonempty(dnf.rules)
    results["rule_size"] = get_rule_size(dnf.rules)
    results["literal_count"] = get_literal_count(dnf.rules)
    results["unique_literal_count"] = get_unique_literal_count(dnf.rules)

    # quantify the lack of rules for an entire class / lack of predictions for individual samples
    results["missing_class_rules"] = sum(len(class_rules) == 0 for class_rules in dnf.rules)
    results["missing_pred_val"] = np.sum(y_val_cfire == -1) / len(y_val_cfire)
    results["missing_pred_test"] = np.sum(y_test_cfire == -1) / len(y_test_cfire)

    return results


def evaluate_cfire(
        cfire: CFIRE,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_val_model: np.ndarray,
        y_test_model: np.ndarray,
) -> dict:
    results = {}

    dnf_rule_metrics = get_dnf_rule_metrics(cfire.dnf, X_val, X_test, y_val_model, y_test_model)
    results.update(dnf_rule_metrics)

    # NOTE: Monte Carlo IoU computation is VERY slow (samples 500k points)
    # Commented out to speed up experiments. Uncomment if you need IoU metrics.
    # class_iou_matrix = get_class_iou_matrix_mc(cfire.dnf.rules)
    # results["max_iou"] = max_offdiag_iou(class_iou_matrix)
    # results["mean_iou"] = mean_offdiag_iou(class_iou_matrix)

    results["t_explanations"] = cfire._compute_times['_calc_explanations']+cfire._compute_times['expl_binarization_fn']
    results["t_rule_candidates"] = cfire._compute_times["_calculate_rule_candidates"]
    results["t_compose_rules"] = cfire._compute_times['_compose_rule_model']

    # analyze input to set covering algorithm
    coverage_matricies = build_coverage_matrices(cfire.frequent_nodes)
    results["mean_coverage_ratio"] = mean_coverage_ratio(coverage_matricies)
    results["mean_single_coverage_ratio"] = mean_single_coverage_ratio(coverage_matricies)
    results["mean_nodes_per_sample"] = mean_nodes_per_sample(coverage_matricies)
    results["mean_duplicate_nodes_ratio"] = mean_duplicate_nodes_ratio(coverage_matricies)
    results["total_frequent_node_count"] = sum(cov_mat.shape[1] for cov_mat in coverage_matricies)

    # metrics on normalized attributions
    E_norm = normalize_explanations(cfire._explanations)
    results["attr_mean_max"] = mean_max_attribution(E_norm)
    results["attr_mean_min"] = mean_min_attribution(E_norm)
    results["attr_max_max"] = max_max_attribution(E_norm)
    results["attr_min_min"] = min_min_attribution(E_norm)
    results["attr_mean_mean"] = mean_mean_attribution(E_norm)
    results["attr_variance"] = attribution_variance(E_norm)
    results["attr_sparsity"] = sparsity(E_norm)
    results["attr_class_separation"] = class_separation_in_attribution_space(E_norm, y_val_model)

    # metrics on binarized explanations
    binarized = cfire._binarized_explanations
    results["bin_mean_active_features_per_sample"] = mean_active_features_per_sample(binarized)
    results["bin_mean_active_features_ratio"] = mean_active_features_ratio(binarized)
    results["bin_mean_feature_activation_ratio"] = mean_feature_activation_ratio(binarized)
    results["bin_features_inactive_ratio"] = features_inactive_ratio(binarized)
    results["bin_mean_feature_class_specificity"] = mean_feature_class_specificity(binarized, y_val_model)
    results["bin_mean_within_class_jaccard"] = mean_within_class_jaccard(binarized, y_val_model)
    results["bin_mean_across_class_jaccard"] = mean_across_class_jaccard(binarized, y_val_model)
    results["bin_class_separation_score"] = class_separation_score(binarized, y_val_model)
    results["bin_all_features_active_ratio"] = all_features_active_ratio(binarized)
    results["bin_all_features_inactive_ratio"] = all_features_inactive_ratio(binarized)

    # TODO: add more metrics, e.g:
    #   - are there any more cfire metrics from paper we might want to include?
    #   - pruning "metrics"

    return results
