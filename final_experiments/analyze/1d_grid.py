from pathlib import Path

import pandas as pd

from final_experiments.analyze.utils import (
    ALL_PARAMS,
    filter_other_params_to_default,
    merge_local_explainers,
    get_metric_table,
    load_csv_files,
)


def analyze_dataset(name: str, base_df: pd.DataFrame, fix_others_to_default: bool = False) -> None:
    print(f"\n\n{'='*15} ANALYSIS FOR DATASET: '{name}' {'='*15}")

    for param in ALL_PARAMS:
        df = filter_other_params_to_default(base_df, param) if fix_others_to_default else base_df
        df_cfire = merge_local_explainers(df)
        df_IG = df[df["expl_method"] == "IG"]
        df_lime = df[df["expl_method"] == "lime"]
        df_kernelshap = df[df["expl_method"] == "kernelshap"]

        for expl_method, df_method in [
            ("IG", df_IG),
            ("lime", df_lime),
            ("kernelshap", df_kernelshap),
            ("merged", df_cfire)
        ]:
            print(f"\n ======= [{name}] param={param} | expl={expl_method} ======")
            print(get_metric_table(df_method, param, metric_selection).to_string())

if __name__ == "__main__":
    metric_selection = [
        "val_acc",
        "val_f1_weighted",
        "test_f1_weighted",
        "test_acc",
        "rule_size",
        "rule_count",
        "literal_count",
        "unique_literal_count",
        "max_iou",
        "mean_iou",
        "t_rule_candidates",
        "t_compose_rules",
        "missing_class_rules",
        "missing_pred_val",
        "missing_pred_test",
        "mean_coverage_ratio",
        "mean_single_coverage_ratio",
        "mean_nodes_per_sample",
        "mean_duplicate_nodes_ratio",
        "total_frequent_node_count",
        "attr_mean_absolute_attribution",
        "attr_attribution_variance",
        "attr_sparsity",
        "attr_class_separation",
        "bin_mean_active_features_per_sample",
        "bin_mean_active_features_ratio",
        "bin_mean_feature_activation_ratio",
        "bin_features_inactive_ratio",
        "bin_mean_feature_class_specificity",
        "bin_mean_within_class_jaccard",
        "bin_mean_across_class_jaccard",
        "bin_class_separation_score",
        "bin_all_features_active_ratio",
        "bin_all_features_inactive_ratio",
    ]
    results_dir = Path("./experiments/2_grid/results/")
    dataframes = load_csv_files(results_dir, csv_file_name="metrics.csv")
    for dataset, base_df in dataframes.items():
        analyze_dataset(dataset, base_df, fix_others_to_default=True)
