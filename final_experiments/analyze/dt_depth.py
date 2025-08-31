# file: dt_depth_delta.py

from pathlib import Path
import pandas as pd
import numpy as np

from final_experiments.analyze.utils import (
    load_csv_files,
    ALL_PARAMS,
    filter_other_params_to_default,
    merge_local_explainers,
)

RESULTS_DIR = Path("./experiments/2_grid/results/")
CSV_FILE = "metrics.csv"

DEPTH_COL = "max_dt_depth"
D7, D2 = 7, 14


def _rel_change(a: pd.Series, b: pd.Series) -> pd.Series:
    # symmetric % change → stable n even when a==0
    denom = (a.abs() + b.abs()).replace(0, np.nan)
    return 200.0 * (b - a) / denom


def _summarize(df7: pd.DataFrame, df2: pd.DataFrame, metrics: list[str]) -> dict[str, tuple[float, float, int]]:
    out = {}
    for m in metrics:
        a = df7[m].astype(float).reset_index(drop=True)
        b = df2[m].astype(float).reset_index(drop=True)
        assert len(a) == len(b), f"mismatch in {m}: {len(a)} vs {len(b)}"
        delta = _rel_change(a, b)
        delta = delta.dropna()
        out[m] = (float(delta.mean()), float(delta.std(ddof=1)) if len(delta) > 1 else 0.0, int(len(delta)))
    return out


def _print_cfire_source_counts(df_cfire: pd.DataFrame, depth_label: str) -> None:
    counts = df_cfire["expl_method"].astype(str).value_counts()
    parts = [f"{k}={int(v)}" for k, v in counts.items()]
    print(f"[CFIRE sources @{depth_label}] n={len(df_cfire)} :: " + ", ".join(parts))


def main():
    data = load_csv_files(RESULTS_DIR, csv_file_name=CSV_FILE)

    groups_order = ["CFIRE", "Greedy", "IG", "lime", "kernelshap"]
    global_buckets: dict[str, dict[str, list[pd.Series]]] = {g: {} for g in groups_order}

    for dataset, df in data.items():
        # lock all params to default EXCEPT depth
        df = filter_other_params_to_default(df, DEPTH_COL)

        df7, df2 = df[df[DEPTH_COL] == D7], df[df[DEPTH_COL] == D2]
        assert not df7.empty and not df2.empty, f"{dataset}: missing depth {D7} or {D2}"

        # build groups per your spec
        groups = {
            "CFIRE": (merge_local_explainers(df7), merge_local_explainers(df2)),
            "Greedy": (
                df7.loc[df7.groupby("model_idx")["test_f1_weighted"].idxmax()],
                df2.loc[df2.groupby("model_idx")["test_f1_weighted"].idxmax()],
            ),
            "IG": (df7[df7["expl_method"] == "IG"], df2[df2["expl_method"] == "IG"]),
            "lime": (df7[df7["expl_method"] == "lime"], df2[df2["expl_method"] == "lime"]),
            "kernelshap": (df7[df7["expl_method"] == "kernelshap"], df2[df2["expl_method"] == "kernelshap"]),
        }

        # sanity: print CFIRE merge source distribution for 7 and 2
        _print_cfire_source_counts(groups["CFIRE"][0], f"{D7}")
        _print_cfire_source_counts(groups["CFIRE"][1], f"{D2}")

        # numeric metrics only (exclude hyperparams + depth)
        metrics = [
            c for c in df.columns
            if c not in ALL_PARAMS and c != DEPTH_COL and pd.api.types.is_numeric_dtype(df[c])
        ]

        print(f"\n### DATASET = {dataset} (Δ% {D7}→{D2})")
        header = ["Metric"] + groups_order
        print("  ".join([f"{h:<25}" for h in header]))
        print("  ".join([f"{'-'*25}" for _ in header]))

        for m in metrics:
            row = [f"{m:<25}"]
            for name in groups_order:
                g7, g2 = groups[name]
                if g7.empty or g2.empty:
                    row.append(f"{'-':<25}")
                    continue
                # summarize + print
                mean, std, n = _summarize(g7, g2, [m])[m]
                row.append(f"{mean:+.2f}%±{std:.2f}% (n={n})".ljust(25))
                # accumulate for global
                delta = _rel_change(
                    g7[m].astype(float).reset_index(drop=True),
                    g2[m].astype(float).reset_index(drop=True),
                ).dropna()
                global_buckets[name].setdefault(m, []).append(delta)
            print("  ".join(row))

    # ---- GLOBAL (across datasets) ----
    print(f"\n### GLOBAL (all datasets) (Δ% {D7}→{D2})")
    header = ["Metric"] + groups_order
    print("  ".join([f"{h:<25}" for h in header]))
    print("  ".join([f"{'-'*25}" for _ in header]))

    # union of metrics across groups
    all_metrics = sorted(set().union(*[set(metrics_dict.keys()) for metrics_dict in global_buckets.values()]))

    for m in all_metrics:
        row = [f"{m:<25}"]
        for name in groups_order:
            series_list = global_buckets[name].get(m, [])
            if not series_list:
                row.append(f"{'-':<25}")
                continue
            combined = pd.concat(series_list, axis=0).dropna()
            if combined.empty:
                row.append(f"{'-':<25}")
                continue
            mean, std, n = float(combined.mean()), float(combined.std(ddof=1)) if len(combined) > 1 else 0.0, int(len(combined))
            row.append(f"{mean:+.2f}%±{std:.2f}% (n={n})".ljust(25))
        print("  ".join(row))


if __name__ == "__main__":
    main()
