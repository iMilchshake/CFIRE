# final_experiments/analyze/optimal_configurations.py
from pathlib import Path
import pandas as pd

from final_experiments.analyze.utils import (
    ALL_PARAMS,
    load_csv_files,
    merge_local_explainers,
    get_stat_str,
    filter_all_params_to_default,
)

METRICS = ["test_f1_weighted", "val_f1_weighted", "rule_size"]


def cfg_tuple_to_str(cfg_tuple) -> str:
    return ", ".join(f"{k}={v}" for k, v in zip(ALL_PARAMS, cfg_tuple))


def best_cfg_for_metric(df: pd.DataFrame, metric: str):
    grouped = df.groupby(ALL_PARAMS, dropna=False)[metric]
    stats = grouped.agg(["mean"])
    best_idx = stats["mean"].idxmax()
    # mask to slice all rows with that hyperparam config
    mask = pd.Series(True, index=df.index)
    for k, v in zip(ALL_PARAMS, best_idx):
        mask &= (df[k].astype(str) == str(v))
    return best_idx, df.loc[mask]


def method_views(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Replicate views used in other scripts, from a *raw* df."""
    df_cfire = merge_local_explainers(df)
    df_greedy = df.loc[df.groupby("model_idx")["test_f1_weighted"].idxmax()]
    df_ig = df[df["expl_method"] == "IG"]
    df_lime = df[df["expl_method"] == "lime"]
    df_ks = df[df["expl_method"] == "kernelshap"]
    return [
        ("CFIRE", df_cfire),
        ("Greedy", df_greedy),
        ("CFIRE-KS", df_ks),
        ("CFIRE-LI", df_lime),
        ("CFIRE-IG", df_ig),
    ]


def main(results_dir: Path) -> None:
    dataframes = load_csv_files(results_dir, csv_file_name="metrics.csv")

    for metric in METRICS:
        print(f"\n\n{'='*15} OPTIMAL CONFIGS — metric: {metric} {'='*15}")

        for dataset_name, df in dataframes.items():
            print(f"\n### DATASET = {dataset_name}")

            # Build method-specific views from raw df
            methods_best = method_views(df)

            # Build default-only baseline views from raw df filtered to defaults
            df_default_only = filter_all_params_to_default(df)
            methods_default = dict(method_views(df_default_only))

            colw_method, colw_cfg, colw_stat = 10, 90, 18
            header = [
                f"{'Method':<{colw_method}}",
                f"{'Best Config':<{colw_cfg}}",
                f"{'Best':<{colw_stat}}",
                f"{'Default':<{colw_stat}}",
            ]
            print("  ".join(header))
            print("  ".join([
                f"{'-'*colw_method:<{colw_method}}",
                f"{'-'*colw_cfg:<{colw_cfg}}",
                f"{'-'*colw_stat:<{colw_stat}}",
                f"{'-'*colw_stat:<{colw_stat}}",
            ]))

            for name, d in methods_best:
                # optimal (by mean of target metric) over hyperparameter configs
                cfg, d_best = best_cfg_for_metric(d, metric)
                best_stat = get_stat_str(d_best, metric)

                # default baseline: computed from default-only RAW df, then same method view
                d_default_view = methods_default.get(name, pd.DataFrame(columns=d.columns))
                default_stat = get_stat_str(d_default_view, metric)

                print("  ".join([
                    f"{name:<{colw_method}}",
                    f"{cfg_tuple_to_str(cfg):<{colw_cfg}}",
                    f"{best_stat:<{colw_stat}}",
                    f"{default_stat:<{colw_stat}}",
                ]))


if __name__ == "__main__":
    results_dir = Path("./experiments/2_grid/results/")
    main(results_dir)
