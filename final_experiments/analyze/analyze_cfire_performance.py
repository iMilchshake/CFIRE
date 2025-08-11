import pandas as pd
from pathlib import Path

def print_metric_stat(df: pd.DataFrame, metric_column: str):
    """Calculates and prints the mean/std for a specified metric column"""
    mean_val = df[metric_column].mean()
    std_val = df[metric_column].std()
    print(f"  {metric_column}: {mean_val:.2f} ± {std_val:.2f}")

def apply_tie_breaker(df: pd.DataFrame) -> pd.DataFrame:
    """For each model_idx, keeps only the row with the highest val_acc"""
    idx = df.groupby("model_idx")["val_acc"].idxmax()
    return df.loc[idx]


results_dir = Path("./experiments/test/results/")
dataframes = {}

if results_dir.exists():
    for subdir in results_dir.iterdir():
        if not subdir.is_dir():
            continue  # Skip any files in the results directory

        results_file = subdir / "results.csv"
        if results_file.exists():
            dataframes[subdir.name] = pd.read_csv(results_file)

for dataset, df in dataframes.items():
    print(f"\n\n{'='*15} ANALYSIS FOR DATASET: '{dataset}' {'='*15}")

    print("\n# --- INDIVIDUAL METHOD RESULTS ---")
    unique_methods = df['expl_method'].unique()

    for method in unique_methods:
        print(f"\n# Method: '{method}'")
        method_df = df[df['expl_method'] == method]
        print_metric_stat(method_df, "test_f1_weighted")
        print_metric_stat(method_df, "test_precision_weighted")
        print_metric_stat(method_df, "test_f1_macro")
        print_metric_stat(method_df, "test_precision_macro")

    print("\n\n# --- TIE-BREAKER RESULTS ---")

    df_tie = apply_tie_breaker(df)

    print("\n# Performance of Final Selection")
    print_metric_stat(df_tie, "test_f1_weighted")
    print_metric_stat(df_tie, "test_precision_weighted")
    print_metric_stat(df_tie, "test_f1_macro")
    print_metric_stat(df_tie, "test_precision_macro")

    print("\n# Final Explanation Counts")
    explanation_counts = df_tie['expl_method'].value_counts()
    print(explanation_counts.to_string())

    print("\n" + "="*50)
