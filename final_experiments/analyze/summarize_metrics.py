# file: summarize_metrics_split_columns.py
from pathlib import Path
import pandas as pd

from final_experiments.analyze.utils import (
    load_csv_files,
    filter_all_params_to_default,
    merge_local_explainers,
)

RESULTS_DIR = Path("./experiments/2_grid/results/")
CSV_FILE = "metrics.csv"


def _pick_metric_cols(df: pd.DataFrame) -> list[str]:
    num_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    blacklist_suffixes = ("_idx",)
    blacklist_exact = {"seed"}
    return [c for c in num_cols if not c.endswith(blacklist_suffixes) and c not in blacklist_exact]


def _mean_std(df: pd.DataFrame, metrics: list[str]) -> dict[str, tuple[float, float, int]]:
    out = {}
    if df.empty:
        for m in metrics:
            out[m] = (float("nan"), float("nan"), 0)
        return out
    agg = df[metrics].agg(["mean", "std", "count"]).T
    for m in metrics:
        row = agg.loc[m]
        out[m] = (row["mean"], row["std"], int(row["count"]))
    return out


def _fmt(mean: float, std: float, n: int, colw: int = 18) -> str:
    if n == 0:
        return f"{'-':<{colw}}"
    return f"{mean:.4f}±{std:.4f} (n={n})".ljust(colw)


def summarize_per_dataset(dfs: dict[str, pd.DataFrame], metrics: list[str]) -> None:
    for dataset, df in dfs.items():
        df_cfire = merge_local_explainers(df)
        methods = sorted(df["expl_method"].dropna().unique().tolist())
        cols = [("CFIRE", df_cfire)] + [(m, df[df["expl_method"] == m]) for m in methods]

        print(f"\n### DATASET = {dataset} (n={len(df)})")
        header = ["Metric"] + [name for name, _ in cols]
        colw = 18
        print("  ".join([f"{header[0]:<25}"] + [f"{h:<{colw}}" for h in header[1:]]))
        print("  ".join([f"{'-'*25:<25}"] + [f"{'-'*colw:<{colw}}" for _ in cols]))

        for m in metrics:
            row = [f"{m:<25}"]
            for _, d in cols:
                mean, std, n = _mean_std(d, [m])[m]
                row.append(_fmt(mean, std, n, colw))
            print("  ".join(row))


def summarize_global(dfs: dict[str, pd.DataFrame], metrics: list[str]) -> None:
    if not dfs:
        return
    df_cfire_all = pd.concat([merge_local_explainers(df) for df in dfs.values()], ignore_index=True)
    all_methods = sorted(set().union(*[df["expl_method"].dropna().unique().tolist() for df in dfs.values()]))
    cols = [("CFIRE", df_cfire_all)]
    for m in all_methods:
        df_m_all = pd.concat([df[df["expl_method"] == m] for df in dfs.values()], ignore_index=True)
        cols.append((m, df_m_all))

    print("\n### GLOBAL (ALL DATASETS)")
    header = ["Metric"] + [name for name, _ in cols]
    colw = 18
    print("  ".join([f"{header[0]:<25}"] + [f"{h:<{colw}}" for h in header[1:]]))
    print("  ".join([f"{'-'*25:<25}"] + [f"{'-'*colw:<{colw}}" for _ in cols]))

    for m in metrics:
        row = [f"{m:<25}"]
        for _, d in cols:
            mean, std, n = _mean_std(d, [m])[m]
            row.append(_fmt(mean, std, n, colw))
        print("  ".join(row))


def main() -> None:
    dfs = load_csv_files(RESULTS_DIR, csv_file_name=CSV_FILE)
    dfs = {k: filter_all_params_to_default(v) for k, v in dfs.items()}
    probe = next((d for d in dfs.values() if not d.empty), None)
    if probe is None:
        print("No data after filtering.")
        return
    metrics = _pick_metric_cols(probe)

    summarize_per_dataset(dfs, metrics)
    summarize_global(dfs, metrics)


if __name__ == "__main__":
    main()
