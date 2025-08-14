from pathlib import Path
import pandas as pd
from pandas.errors import EmptyDataError

# --- helpers ---
def safe_read_csv(p: Path) -> pd.DataFrame:
    """Read CSV or return empty DataFrame if file is empty or unreadable."""
    try:
        if not p.exists() or p.stat().st_size == 0:
            return pd.DataFrame()
        return pd.read_csv(p)
    except EmptyDataError:
        return pd.DataFrame()

def load_results(root: Path):
    df_ok, df_fail = {}, {}
    if root.exists():
        for subdir in root.iterdir():
            if not subdir.is_dir():
                continue
            ok_file = subdir / "results.csv"
            fail_file = subdir / "failed_runs.csv"
            if ok_file.exists():
                df_ok[subdir.name] = safe_read_csv(ok_file)
            if fail_file.exists():
                df_fail[subdir.name] = safe_read_csv(fail_file)
    return df_ok, df_fail

# --- load ---
results_no_subset = Path("./experiments/2_grid/results_no_subset/")
results = Path("./experiments/2_grid/results/")

dataframes_subset, fails_subset = load_results(results)
dataframes_no_subset, fails_no_subset = load_results(results_no_subset)

# --- key alignment ---
common = sorted(set(dataframes_subset) & set(dataframes_no_subset))

timing_cols = ["t_explanations", "t_rule_candidates", "t_compose_rules"]
key_cols = ["model_idx", "cfire_config_idx", "cfire_seed"]

for name in common:
    df_s = dataframes_subset[name]
    df_ns = dataframes_no_subset[name]

    # successes / failures
    succ_s = len(df_s)
    succ_ns = len(df_ns)
    fail_s = len(fails_subset.get(name, pd.DataFrame()))
    fail_ns = len(fails_no_subset.get(name, pd.DataFrame()))

    # merge for timing comparison
    left = df_s[key_cols + timing_cols] if not df_s.empty else pd.DataFrame(columns=key_cols + timing_cols)
    right = df_ns[key_cols + timing_cols] if not df_ns.empty else pd.DataFrame(columns=key_cols + timing_cols)
    merged = left.merge(right, on=key_cols, suffixes=("_subset", "_no_subset"), how="inner")

    print(f"\n=== {name} ===")
    print(f"Subset: {succ_s} successes, {fail_s} fails | No subset: {succ_ns} successes, {fail_ns} fails")

    # per-row compact print
    for _, row in merged.iterrows():
        s1, s2, s3 = (row[f"{c}_subset"] for c in timing_cols)
        n1, n2, n3 = (row[f"{c}_no_subset"] for c in timing_cols)
        d1, d2, d3 = s1 - n1, s2 - n2, s3 - n3
        print(
            f"{int(row['model_idx']):02d}-{int(row['cfire_config_idx']):02d}-{int(row['cfire_seed'])} | "
            f"Subset: {s1:.2f}s, {s2:.2f}s, {s3:.2f}s | "
            f"No subset: {n1:.2f}s, {n2:.2f}s, {n3:.2f}s | "
            f"Δ: {d1:.2f}s, {d2:.2f}s, {d3:.2f}s"
        )

    # simple mean stats (per-dataset)
    if not merged.empty:
        mean_subset = merged[[f"{c}_subset" for c in timing_cols]].mean()
        mean_nosub = merged[[f"{c}_no_subset" for c in timing_cols]].mean()
        print("\nMeans:")
        for c in timing_cols:
            print(f"{c}: subset={mean_subset[f'{c}_subset']:.2f}s, no_subset={mean_nosub[f'{c}_no_subset']:.2f}s, "
                  f"Δ={(mean_subset[f'{c}_subset']-mean_nosub[f'{c}_no_subset']):.2f}s")
