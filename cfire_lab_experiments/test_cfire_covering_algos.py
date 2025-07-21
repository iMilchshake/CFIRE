import logging
import sys
from glob import glob
from pathlib import Path

import pandas as pd

from cfire.nodeselection import inv_freq_set_cover, greedy_cover, dedup_greedy_cover
from cfire_lab_experiments.test_cfire_utils import load_data, build_model, evaluate_cfire, recalculate_rule_composition, \
    fit_cfire


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    # build model & CFIRE
    X_val, X_test, n_dim, n_classes = load_data()

    model_paths = sorted(glob("./models/tmp_*.ckpt"))
    explanation_paths = sorted(glob("./models/explanations_*.pt"))
    results = []

    COMPOSITION_CONFIGS = [
        ("default", greedy_cover),
        ("default_dedup", dedup_greedy_cover),
        ("inv_freq_set_cover", inv_freq_set_cover),
    ]
    SEEDS = [42, 43, 44, 45, 46]

    for model_idx, model_path in enumerate(model_paths):
        logging.info(f"MODEL_IDX = {model_idx}")
        for seed in SEEDS:

            model = build_model(n_dim, n_classes, Path(model_path))
            logging.info(f"\tFIT CFIRE SEED={seed}")
            cfire = fit_cfire(model, X_val, Path(explanation_paths[model_idx]), seed)

            for comp_name, comp_fn in COMPOSITION_CONFIGS:
                logging.info(f"\t\tfinished {comp_name}")
                _, nodes = recalculate_rule_composition(cfire, comp_fn)
                node_counts = [len(class_nodes) for class_nodes in nodes]
                eval_results = evaluate_cfire(cfire, model, X_val, X_test)
                results.append(
                    {
                        "model_idx": model_idx,
                        "seed": seed,
                        "composition": comp_name,
                        "node_counts": node_counts,
                        **eval_results,
                    }
                )

    df = pd.DataFrame(results)
    df.to_csv("cfire_eval_results.csv", index=False)


if __name__ == "__main__":
    main()
