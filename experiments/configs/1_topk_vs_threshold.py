import logging
import sys
from itertools import product
from pathlib import Path

from final_experiments.experiment import (
    CFIREExperiment,
    ThresholdBinarization,
    run_experiment,
    TopKBinarization,
)
from final_experiments.logging import init_logger

# Unique experiment name
EXPERIMENTS_NAME = "1_topk_vs_threshold_bin"

# With how many models/seeds should each experiment be evaluated?
N_MODELS = 2
N_SEEDS = 3

# Define all experiment permutations
PARAMS = {
    "freq_threshold": [0.01],
    "max_dt_depth": [7],
    "dataset_name": ["abalone", "wine", "iris"],
    "bin_config": [ThresholdBinarization(threshold=0.01), TopKBinarization(k=2)],
}

if __name__ == "__main__":
    experiments_dir = Path(f"./experiments/") / EXPERIMENTS_NAME
    init_logger(experiments_dir)

    experiments = [
        CFIREExperiment(
            **dict(zip(list(PARAMS.keys()), vals)),
            n_models=N_MODELS,
            n_seeds=N_SEEDS,
        )
        for vals in product(*PARAMS.values())
    ]

    for experiment_idx, experiment in enumerate(experiments):
        run_experiment(experiment, experiment_idx, experiments_dir)
