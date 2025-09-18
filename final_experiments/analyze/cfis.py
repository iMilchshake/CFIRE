from pathlib import Path
import pandas as pd

from final_experiments.analyze.utils import (
    load_csv_files,
    get_stat_str,
    merge_local_explainers,
    filter_all_params_to_default,
)

COVERAGE_METRICS = [
    "mean_coverage_ratio",
    "mean_single_coverage_ratio",
    "mean_nodes_per_sample",
    "mean_duplicate_nodes_ratio",
    "total_frequent_node_count",
]

METHOD_FOR_DATASET_TABLE = "CFIRE"  # CFIRE, Greedy, IG, lime, kernelshap


def main(results_dir: Path, fix_all_to_default: bool = True) -> None:
    dataframes = load_csv_files(results_dir, csv_file_name="metrics.csv")

    pooled = {"CFIRE": [], "Greedy": [], "IG": [], "lime": [], "kernelshap": []}
    per_dataset: dict[str, dict[str, pd.DataFrame]] = {}

    for ds, df in dataframes.items():
        if fix_all_to_default:
            df = filter_all_params_to_default(df)

        frames = {
            "CFIRE": merge_local_explainers(df),
            "Greedy": df.loc[df.groupby("model_idx")["test_f1_weighted"].idxmax()],
            "IG": df[df["expl_method"] == "IG"],
            "lime": df[df["expl_method"] == "lime"],
            "kernelshap": df[df["expl_method"] == "kernelshap"],
        }
        per_dataset[ds] = frames

        for k, d in frames.items():
            cols = [c for c in COVERAGE_METRICS if c in d.columns]
            if cols:
                pooled[k].append(d[cols])

    pooled = {k: (pd.concat(v, axis=0, ignore_index=True) if v else pd.DataFrame())
              for k, v in pooled.items()}

    # 1) aggregate over datasets (one line per dataset, columns = metrics) for METHOD_FOR_DATASET_TABLE
    print(f"\n### AGGREGATE OVER DATASETS (method={METHOD_FOR_DATASET_TABLE})")
    colw = 22
    header = ["Dataset"] + COVERAGE_METRICS
    print("  ".join([f"{header[0]:<20}"] + [f"{h:<{colw}}" for h in header[1:]]))
    print("  ".join([f"{'-'*20:<20}"] + [f"{'-'*colw:<{colw}}" for _ in COVERAGE_METRICS]))
    for ds, frames in sorted(per_dataset.items()):
        d = frames[METHOD_FOR_DATASET_TABLE]
        cells = [f"{ds:<20}"]
        for metr in COVERAGE_METRICS:
            cells.append(f"{get_stat_str(d, metr):<{colw}}")
        print("  ".join(cells))

    # 2) aggregate over local explainers (metrics as rows, explainers as columns)
    print("\n### AGGREGATE OVER LOCAL EXPLAINERS (coverage metrics)")
    cols = [
        ("CFIRE", pooled["CFIRE"]),
        ("Greedy", pooled["Greedy"]),
        ("CFIRE-IG", pooled["IG"]),
        ("CFIRE-LI", pooled["lime"]),
        ("CFIRE-KS", pooled["kernelshap"]),
    ]
    colw = 16
    header = ["Metric"] + [name for name, _ in cols]
    print("  ".join([f"{header[0]:<26}"] + [f"{h:<{colw}}" for h in header[1:]]))
    print("  ".join([f"{'-'*26:<26}"] + [f"{'-'*colw:<{colw}}" for _ in cols]))
    for metr in COVERAGE_METRICS:
        row = [f"{metr:<26}"]
        for _, d in cols:
            row.append(f"{get_stat_str(d, metr):<{colw}}")
        print("  ".join(row))

    # 3) per-dataset view: two metrics as columns, split into three local explainers (IG/lime/kernelshap)
    print("\n### PER-DATASET (locals split) — mean_duplicate_nodes_ratio & total_frequent_node_count")
    locals_keys = [("IG", "IG"), ("lime", "lime"), ("kernelshap", "kernelshap")]
    metrics_twocol = ["mean_duplicate_nodes_ratio", "total_frequent_node_count"]

    subcolw = 20
    top_header = ["Dataset"] + [f"{name:^{subcolw*len(metrics_twocol)}}" for name, _ in locals_keys]
    print("  ".join([f"{top_header[0]:<20}"] + [f"{h:<{subcolw*len(metrics_twocol)}}" for h in top_header[1:]]))

    sub_header = [" " * 20] + [
        "  ".join([f"{m:<{subcolw}}" for m in metrics_twocol]) for _ in locals_keys
    ]
    print("  ".join(sub_header))

    sep = "  ".join([f"{'-'*20:<20}"] + [f"{'-'*(subcolw*len(metrics_twocol)):<{subcolw*len(metrics_twocol)}}" for _ in locals_keys])
    print(sep)

    for ds, frames in sorted(per_dataset.items()):
        row = [f"{ds:<20}"]
        for _, key in locals_keys:
            d = frames[key]
            for metr in metrics_twocol:
                row.append(f"{get_stat_str(d, metr):<{subcolw}}")
        print("  ".join(row))


if __name__ == "__main__":
    main(Path("./experiments/2_grid/results/"), fix_all_to_default=True)
