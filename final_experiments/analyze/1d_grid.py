from pathlib import Path
import pandas as pd

from final_experiments.experiment import ThresholdBinarization

def safe_read_csv(p: Path) -> pd.DataFrame:
    """Fail-fast CSV loader."""
    return pd.read_csv(p)

def load_results(root: Path, ok_name: str = "results.csv") -> dict[str, pd.DataFrame]:
    """Map dataset-name -> DataFrame (no ALL aggregation)."""
    out: dict[str, pd.DataFrame] = {}
    if not root.exists():
        return out
    for subdir in root.iterdir():
        if not subdir.is_dir():
            continue
        f = subdir / ok_name
        if f.exists():
            out[subdir.name] = safe_read_csv(f)
    return out

# ---------- core metrics ----------
def get_stat_str(df: pd.DataFrame, metric_column: str) -> str:
    vals = pd.to_numeric(df[metric_column], errors="coerce").dropna()
    return "—" if vals.empty else f"{vals.mean():.2f}±{vals.std():.2f}"

def apply_tie_breaker(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical selection: max val_acc per (model_idx + all PARAMS)."""
    group_keys = ["model_idx"] + PARAMS
    idx = df.groupby(group_keys, dropna=False)["val_acc"].idxmax()
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
    return out

PARAMS = ["freq_threshold", "max_dt_depth", "bin_config"]
METRICS = ["test_f1_weighted", "test_acc", "rule_size"]

# kept for future extensibility
DEFAULT_PARAMS = {
    "freq_threshold": 0.01,
    "max_dt_depth": 7,
    "bin_config": ThresholdBinarization(threshold=0.01),
}

def analyze_dataset(name: str, df: pd.DataFrame) -> None:
    print(f"\n\n{'='*15} ANALYSIS FOR DATASET: '{name}' {'='*15}")
    df_cfire = apply_tie_breaker(df)
    for param in PARAMS:
        table = metric_table_simple(df_cfire, param, METRICS)
        print(f"\n[param = {param}]")
        print(table.to_string())

def analyze_results(root: Path) -> None:
    dataframes = load_results(root)
    for dataset, df in dataframes.items():
        analyze_dataset(dataset, df)

# ---------- entry point ----------
if __name__ == "__main__":
    results_dir = Path("./experiments/2_grid/results/")
    analyze_results(results_dir)
