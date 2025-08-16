from itertools import product
import logging
from pathlib import Path

from final_experiments.experiment import (
    CFIREExperiment,
    CFIREConfig,
    ThresholdBinarization,
    run_experiment,
    TopKBinarization,
)
from final_experiments.logger import init_logger

# Unique experiment name
EXPERIMENTS_NAME = "2_grid"

# With how many models/seeds should each experiment be evaluated?
N_MODELS = 50
N_SEEDS = 1

# Define all experiment permutations
PARAMS = {
    "freq_threshold": [0.001, 0.01, 0.1, 0.25], 
    "max_dt_depth": [14, 7, 3, 2], 
    "bin_config": [ThresholdBinarization(threshold=0.01), ThresholdBinarization(threshold=0.1), ThresholdBinarization(threshold=0.25),
                   TopKBinarization(k=2), TopKBinarization(k=3)],
}

DATASETS=[
    "abalone",
    "breastw",
    "spambase",
    "beans",
    "ionosphere",
    "breastcancer",
    "btsc",
    "spf",
    "wine",
    "diggle",
    "iris",
    "vehicle",
    "autouniv",
]

if __name__ == "__main__":
    experiments_dir = Path(f"./experiments/") / EXPERIMENTS_NAME
    init_logger(experiments_dir)

    cfire_configs = [
        CFIREConfig(
            freq_threshold=freq_threshold,
            bin_config=bin_config,
            max_dt_depth=max_dt_depth,
        )
        for freq_threshold, max_dt_depth, bin_config in product(
            PARAMS["freq_threshold"],
            PARAMS["max_dt_depth"],
            PARAMS["bin_config"],
        )
    ]

    for dataset in DATASETS:
        experiment = CFIREExperiment(
            dataset_name=dataset,
            n_models=N_MODELS,
            n_seeds=N_SEEDS,
            cfire_configs=cfire_configs,
        )
        run_experiment(experiment, experiments_dir, timeout=1200)
