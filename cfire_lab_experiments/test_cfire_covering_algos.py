import logging
import sys
import time
from glob import glob
from pathlib import Path

import pandas as pd

from cfire.nodeselection import (
    greedy_cover,
    dedup_greedy_cover,
    get_inv_freq_set_cover,
    optimal_min_rules_cover,
    solve_ilp_set_cover, get_ilp_solver,
)
from cfire_lab_experiments.utils_cfire import (
    load_data,
    load_model,
    evaluate_cfire,
    recalculate_rule_composition,
    fit_cfire,
)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    # build model & CFIRE
    X_val, X_test, n_dim, n_classes = load_data()

    model_paths = sorted(glob("./cfire_lab_experiments/models/tmp_*.ckpt"))
    explanation_paths = sorted(glob("./cfire_lab_experiments/models/explanations_*.pt"))
    results = []

    MAX_MODELS = 10

    model_paths = model_paths[:MAX_MODELS]
    explanation_paths = explanation_paths[:MAX_MODELS]

    COMPOSITION_CONFIGS = [
        ("ilp_min_rule_size", get_ilp_solver(lambda n: n.dnf.n_rules)),
        ("default", greedy_cover),
        ("default_dedup", dedup_greedy_cover),
        ("inv_freq_set_cover_a=0.5", get_inv_freq_set_cover(0.5)),
        ("inv_freq_set_cover_a=1.0", get_inv_freq_set_cover(1.0)),
        ("inv_freq_set_cover_a=1.5", get_inv_freq_set_cover(1.5)),
        ("inv_freq_set_cover_a=2.0", get_inv_freq_set_cover(2.0)),
        # ("optimal_min_rules_cover", optimal_min_rules_cover),
    ]
    SEEDS = [42, 43, 44]

    for model_idx, model_path in enumerate(model_paths):
        logging.info(f"MODEL_IDX = {model_idx}")
        for seed in SEEDS:

            model = load_model(n_dim, n_classes, Path(model_path))
            logging.info(f"\tFIT CFIRE SEED={seed}")
            cfire = fit_cfire(model, X_val, Path(explanation_paths[model_idx]), seed)

            for comp_name, comp_fn in COMPOSITION_CONFIGS:
                logging.info(f"\t\tcomputing {comp_name}")
                t0 = time.perf_counter()
                _, nodes = recalculate_rule_composition(cfire, comp_fn)
                time_elapsed = time.perf_counter() - t0
                node_counts = [len(class_nodes) for class_nodes in nodes]
                eval_results = evaluate_cfire(cfire, model, X_val, X_test)
                results.append(
                    {
                        "model_idx": model_idx,
                        "seed": seed,
                        "composition": comp_name,
                        "node_counts": node_counts,
                        "comp_time": time_elapsed,
                        **eval_results,
                    }
                )

            # get best config for this model/seed configuration
            model_seed_results = [
                r for r in results if r["model_idx"] == model_idx and r["seed"] == seed
            ]
            best_result = max(model_seed_results, key=lambda r: r["val_acc"])
            best_result_named = {**best_result, "composition": "best_val_acc"}
            results.append(best_result_named)

    df = pd.DataFrame(results)
    df.to_csv("cfire_eval_results.csv", index=False)


if __name__ == "__main__":
    main()
