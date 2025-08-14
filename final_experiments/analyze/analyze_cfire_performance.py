from pathlib import Path
import pandas as pd
from pandas.errors import EmptyDataError

# --- helpers (reused) ---
def safe_read_csv(p: Path) -> pd.DataFrame:
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

# --- your existing helpers ---
def get_stat_str(df: pd.DataFrame, metric_column: str):
    mean_val = df[metric_column].mean()
    std_val = df[metric_column].std()
    return f"{mean_val:.2f} ± {std_val:.2f}"

def apply_tie_breaker(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby("model_idx")["val_acc"].idxmax()
    return df.loc[idx]

# --- load ---
results_dir = Path("./experiments/2_grid/results/")
dataframes, fails = load_results(results_dir)

for dataset, df in dataframes.items():
    print(f"\n\n{'='*15} ANALYSIS FOR DATASET: '{dataset}' {'='*15}")

    # precompute failed counts per method from failed_runs.csv (if present)
    failed_df = fails.get(dataset, pd.DataFrame())
    failed_counts = (
        failed_df["expl_method"].value_counts().to_dict()
        if not failed_df.empty and "expl_method" in failed_df.columns
        else {}
    )

    print("\n# --- INDIVIDUAL METHOD RESULTS ---")
    unique_methods = df['expl_method'].unique()

    for method in unique_methods:
        method_df = df[df['expl_method'] == method]
        succeeded = len(method_df)
        failed = int(failed_counts.get(method, 0))
        total = succeeded + failed
        suffix = f" ({failed}/{total} failed runs)" if failed > 0 else ""

        print(f"{method:<10} F1: {get_stat_str(method_df, 'test_f1_weighted')}   "
              f"Precision: {get_stat_str(method_df, 'test_precision_weighted')}{suffix}")

    print()
    df_tie = apply_tie_breaker(df)
    print(f"tie-break  F1: {get_stat_str(df_tie, 'test_f1_weighted')}   Precision: {get_stat_str(df_tie, 'test_precision_weighted')}")

    # for each config, pick best wrt to TEST performance
    df_greedy = df.loc[df.groupby('model_idx')['test_f1_weighted'].idxmax()].reset_index(drop=True)
    print(f"greedy     F1: {get_stat_str(df_greedy, 'test_f1_weighted')}   Precision: {get_stat_str(df_greedy, 'test_precision_weighted')}")

    print("\n# Final Explanation Counts")
    explanation_counts = df_tie['expl_method'].value_counts()
    print(explanation_counts.to_string())

    print("\n" + "="*50)
