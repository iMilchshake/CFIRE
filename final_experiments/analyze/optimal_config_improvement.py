# final_experiments/analyze/optimal_configurations.py

# NOTE: GPT GENERATED VARIATION of optimal_configs (not double checked!)

from pathlib import Path
import pandas as pd
from tabulate import tabulate

from final_experiments.analyze.utils import (
    ALL_PARAMS,
    load_csv_files,
    merge_local_explainers,
    filter_all_params_to_default,
)

METRICS = ["test_f1_weighted", "val_f1_weighted", "rule_size"]
THRESHOLD_PCT = 5.0


def method_views(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
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


def best_cfg_slice_and_str(df: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, str]:
    grouped = df.groupby(ALL_PARAMS, dropna=False)[metric]
    best_idx = grouped.mean().idxmax()
    mask = pd.Series(True, index=df.index)
    for k, v in zip(ALL_PARAMS, best_idx):
        mask &= (df[k].astype(str) == str(v))
    cfg_str = ", ".join(f"{k}={v}" for k, v in zip(ALL_PARAMS, best_idx))
    return df.loc[mask], cfg_str


def pct_improvement(best_df: pd.DataFrame, default_df: pd.DataFrame, metric: str) -> float:
    best_mean = float(pd.to_numeric(best_df[metric], errors="raise").mean())
    def_mean = float(pd.to_numeric(default_df[metric], errors="raise").mean())
    if def_mean == 0.0:
        return float("inf") if best_mean > 0 else 0.0
    return (best_mean - def_mean) / def_mean * 100.0


def main(results_dir: Path) -> None:
    dataframes = load_csv_files(results_dir, csv_file_name="metrics.csv")

    for metric in METRICS:
        rows = []
        for dataset_name, df in dataframes.items():
            best_views = method_views(df)
            default_views = dict(method_views(filter_all_params_to_default(df)))

            for method_name, df_method in best_views:
                df_default = default_views.get(method_name)
                if df_default is None or df_default.empty:
                    continue
                df_best, cfg_str = best_cfg_slice_and_str(df_method, metric)
                pct = pct_improvement(df_best, df_default, metric)
                if pd.isna(pct) or pct <= THRESHOLD_PCT or pct in (float("inf"), float("-inf")):
                    continue
                rows.append(
                    {
                        "dataset": dataset_name,
                        "method": method_name,
                        "pct_improvement": f"{pct:+.2f}%",
                        "best_params": cfg_str,
                    }
                )

        out = pd.DataFrame(rows, columns=["dataset", "method", "pct_improvement", "best_params"])
        if not out.empty:
            out = out.sort_values(by=["dataset", "method"], kind="stable", ignore_index=True)

        print(f"\n===== COMPACT — metric: {metric} =====")
        print(tabulate(out, headers="keys", tablefmt="plain", showindex=False))


if __name__ == "__main__":
    results_dir = Path("./experiments/2_grid/results/")
    main(results_dir)
