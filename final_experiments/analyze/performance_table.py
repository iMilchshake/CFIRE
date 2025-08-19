from pathlib import Path

from final_experiments.analyze.utils import (
    load_csv_files,
    get_stat_str,
    merge_local_explainers,
    filter_all_params_to_default,
)


def main(results_dir: Path, fix_all_to_default: bool = False) -> None:
    dataframes = load_csv_files(results_dir, csv_file_name="metrics.csv")
    fails = load_csv_files(results_dir, csv_file_name="failed_runs.csv")

    for dataset_name, df in dataframes.items():

        print(f"\n### DATASET = {dataset_name}")

        if fix_all_to_default:
            df = filter_all_params_to_default(df)

        df_cfire = merge_local_explainers(df)
        df_greedy = df.loc[df.groupby("model_idx")["test_f1_weighted"].idxmax()]
        df_IG = df[df["expl_method"] == "IG"]
        df_lime = df[df["expl_method"] == "lime"]
        df_kernelshap = df[df["expl_method"] == "kernelshap"]

        cols = [("CFIRE", df_cfire), ("Greedy", df_greedy),
                ("CFIRE-KS", df_kernelshap), ("CFIRE-LI", df_lime), ("CFIRE-IG", df_IG)]

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

        # Failure warnings
        failed_df = fails.get(dataset_name, None)
        if failed_df is not None:
            failed_counts = failed_df["expl_method"].value_counts().to_dict()
            succeeded_counts = df["expl_method"].value_counts()
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

        counts = df_cfire["expl_method"].value_counts().sort_values(ascending=False)
        print("\n# Final Explanation Counts")
        print(f"{'Method':<12}{'Count':>6}")
        print(f"{'-'*12}{'-'*6}")
        for m, c in counts.items():
            print(f"{m:<12}{c:>6}")

if __name__ == "__main__":
    results_dir = Path("./experiments/2_grid/results/")
    print("### DEFAULT ONLY")

    print(f"\n\n{'='*15} DEFAULT PARAMETERS ONLY' {'='*15}")
    main(results_dir, fix_all_to_default=True)
    print(f"\n\n{'='*15} ALL PARAMETERS ' {'='*15}")
    main(results_dir, fix_all_to_default=False)
