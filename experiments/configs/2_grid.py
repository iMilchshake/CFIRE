from itertools import product
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
N_MODELS = 10
N_SEEDS = 3

# Define all experiment permutations
PARAMS = {
    "freq_threshold": [0.01, 0.1],
    "max_dt_depth": [7, 3],
    "bin_config": [ThresholdBinarization(threshold=0.01), TopKBinarization(k=2)],
}
DATASETS = ["abalone", "wine", "iris"]

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
        run_experiment(experiment, experiments_dir)
