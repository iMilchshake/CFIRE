from pathlib import Path
import pandas as pd

from final_experiments.experiment import ThresholdBinarization

def safe_read_csv(p: Path) -> pd.DataFrame:
    """Fail-fast CSV loader."""
    return pd.read_csv(p)

def load_results(root: Path, name) -> dict[str, pd.DataFrame]:
    """Map dataset-name -> DataFrame (no ALL aggregation)."""
    out: dict[str, pd.DataFrame] = {}
    if not root.exists():
        return out
    for subdir in root.iterdir():
        if not subdir.is_dir():
            continue
        f = subdir / name
        if f.exists():
            out[subdir.name] = safe_read_csv(f)
    return out

def get_stat_str(df: pd.DataFrame, metric_column: str) -> str:
    vals = pd.to_numeric(df[metric_column], errors="raise").dropna() # coerce?
    return "—" if vals.empty else f"{vals.mean():.2f}±{vals.std():.2f}"

def apply_tie_breaker(df: pd.DataFrame, group_by: list[str] | None = None) -> pd.DataFrame:
    """Canonical selection: max val_acc per grouping keys."""
    if group_by is None:
        group_by = ["model_idx"] + PARAMS
    idx = df.groupby(group_by, dropna=False)["val_acc"].idxmax()
    return df.loc[idx]

def metric_table_simple(df_sel: pd.DataFrame, hyperparam: str, metrics: list[str]) -> pd.DataFrame:
    grouped = df_sel.groupby(df_sel[hyperparam], dropna=False)
    rows, index_vals = [], []
    for k, g in grouped:
        index_vals.append(k)
        rows.append({m: get_stat_str(g, m) for m in metrics})
    out = pd.DataFrame(rows, index=index_vals)
    out.index.name = hyperparam
    try:
        out = out.sort_index()
    except Exception:
        pass
    return out.T

PARAMS = ["freq_threshold", "max_dt_depth", "bin_config"]
METRICS = [
    "test_f1_weighted", "test_acc", "rule_size", "rule_count", "literal_count", "unique_literal_count",
    "max_iou", "mean_iou", "t_rule_candidates", "t_compose_rules", "missing_class_rules",
    "missing_pred_val", "missing_pred_test", "mean_coverage_ratio", "mean_single_coverage_ratio",
    "mean_nodes_per_sample", "mean_duplicate_nodes_ratio", "total_frequent_node_count",
    'attr_mean_absolute_attribution', 'attr_attribution_variance', 'attr_sparsity', 'attr_class_separation',
    'bin_mean_active_features_per_sample', 'bin_mean_active_features_ratio', 'bin_mean_feature_activation_ratio',
    'bin_features_inactive_ratio', 'bin_mean_feature_class_specificity', 'bin_mean_within_class_jaccard',
    'bin_mean_across_class_jaccard', 'bin_class_separation_score', 'bin_all_features_active_ratio', 'bin_all_features_inactive_ratio'
]

DEFAULT_PARAMS = {
    "freq_threshold": 0.01,
    "max_dt_depth": 7,
    "bin_config": ThresholdBinarization(threshold=0.01),
}

def filter_to_defaults(df: pd.DataFrame, target_param: str) -> pd.DataFrame:
    """ filter dataframe to fixed default parameters, but keep all values of target_param """
    mask = pd.Series(True, index=df.index)
    for p in PARAMS:
        if p == target_param:
            continue
        mask &= df[p].astype(str) == str(DEFAULT_PARAMS[p])
    return df.loc[mask]


def analyze_dataset(name: str, base_df: pd.DataFrame, fix_others_to_default: bool = False) -> None:
    print(f"\n\n{'='*15} ANALYSIS FOR DATASET: '{name}' {'='*15}")

    for param in PARAMS:

        if fix_others_to_default:
            df = filter_to_defaults(base_df, param)
            df_cfire = apply_tie_breaker(df, ["model_idx", param])
        else:
            df = base_df
            df_cfire = apply_tie_breaker(base_df)

        df_IG = df[df["expl_method"] == "IG"]  # 1
        df_lime = df[df["expl_method"] == "lime"]  # 1
        df_kernelshap = df[df["expl_method"] == "kernelshap"]  # 1

        for expl_method, df_method in [
            ("IG", df_IG),
            ("lime", df_lime),
            ("kernelshap", df_kernelshap),
            ("merged", df_cfire)
        ]:
            print(f"\n ======= [{name}] param={param} | expl={expl_method} ======")
            print(metric_table_simple(df_method, param, METRICS).to_string())

def analyze_results(root: Path, fix_others_to_default: bool = False) -> None:
    dataframes = load_results(root, name="metrics.csv")
    for dataset, df in dataframes.items():
        analyze_dataset(dataset, df, fix_others_to_default=fix_others_to_default)
    # print(df.columns.tolist())

if __name__ == "__main__":
    results_dir = Path("./experiments/2_grid/results/")
    analyze_results(results_dir, fix_others_to_default=True)
