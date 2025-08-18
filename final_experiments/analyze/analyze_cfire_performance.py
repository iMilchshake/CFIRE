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
            ok_file = subdir / "metrics.csv"
            fail_file = subdir / "failed_runs.csv"
            if ok_file.exists():
                df_ok[subdir.name] = safe_read_csv(ok_file)
            if fail_file.exists():
                df_fail[subdir.name] = safe_read_csv(fail_file)
    return df_ok, df_fail

# --- metrics + selection ---
def get_stat_str(df: pd.DataFrame, metric_column: str):
    if df is None or df.empty or metric_column not in df.columns:
        return "—"
    vals = pd.to_numeric(df[metric_column], errors="coerce").dropna()
    if vals.empty:
        return "—"
    return f"{vals.mean():.2f}±{vals.std():.2f}"

def apply_tie_breaker(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "model_idx" not in df.columns or "val_acc" not in df.columns:
        return pd.DataFrame(columns=df.columns if df is not None else None)
    idx = df.groupby("model_idx")["val_acc"].idxmax()
    return df.loc[idx]

def method_tie(df: pd.DataFrame, method: str) -> pd.DataFrame:
    if df is None or df.empty or "expl_method" not in df.columns:
        return pd.DataFrame(columns=df.columns if df is not None else None)
    return apply_tie_breaker(df[df["expl_method"] == method])

# --- load ---
results_dir = Path("./experiments/2_grid/results/")
dataframes, fails = load_results(results_dir)

for dataset, df in dataframes.items():
    print(f"\n\n{'='*15} ANALYSIS FOR DATASET: '{dataset}' {'='*15}")

    failed_df = fails.get(dataset, pd.DataFrame())
    failed_counts = (
        failed_df["expl_method"].value_counts().to_dict()
        if not failed_df.empty and "expl_method" in failed_df.columns
        else {}
    )
    succeeded_counts = df["expl_method"].value_counts() if "expl_method" in df.columns else pd.Series(dtype=int)

    # CFIRE (overall tie-break), Greedy, then the three individual methods
    df_cfire = apply_tie_breaker(df)
    if not df.empty and "model_idx" in df.columns and "test_f1_weighted" in df.columns:
        df_greedy = df.loc[df.groupby("model_idx")["test_f1_weighted"].idxmax()]
    else:
        df_greedy = pd.DataFrame(columns=df.columns if df is not None else None)
    df_ks = method_tie(df, "kernelshap")  # CFIRE-KS
    df_li = method_tie(df, "lime")        # CFIRE-LI
    df_ig = method_tie(df, "IG")          # CFIRE-IG

    cols = [("CFIRE", df_cfire), ("Greedy", df_greedy),
            ("CFIRE-KS", df_ks), ("CFIRE-LI", df_li), ("CFIRE-IG", df_ig)]

    colw = 12
    header = ["Metric"] + [name for name, _ in cols]
    print("\n" + "  ".join([f"{header[0]:<10}"] + [f"{h:<{colw}}" for h in header[1:]]))
    print("  ".join([f"{'-'*10:<10}"] + [f"{'-'*colw:<{colw}}" for _ in cols]))

    for label, metr in [("F1", "test_f1_weighted"),
                        ("Precision", "test_precision_weighted"),
                        ("Size", "rule_size")]:
        cells = [f"{label:<10}"]
        for _, d in cols:
            cells.append(f"{get_stat_str(d, metr):<{colw}}")
        print("  ".join(cells))

    # Failure warnings only (print nothing if no failures)
    warnings = []
    all_methods = sorted(set(succeeded_counts.index).union(set(failed_counts.keys())))
    for m in all_methods:
        fail = int(failed_counts.get(m, 0))
        if fail > 0:
            succ = int(succeeded_counts.get(m, 0))
            total = fail + succ
            warnings.append(f"[warn] {fail}/{total} runs failed for explainer '{m}'")
    if warnings:
        print()
        for w in warnings:
            print(w)

    # Final explanation counts (compact table)
    counts = df_cfire["expl_method"].value_counts().sort_values(ascending=False) if not df_cfire.empty and "expl_method" in df_cfire.columns else pd.Series(dtype=int)
    print("\n# Final Explanation Counts")
    print(f"{'Method':<12}{'Count':>6}")
    print(f"{'-'*12}{'-'*6}")
    for m, c in counts.items():
        print(f"{m:<12}{c:>6}")
