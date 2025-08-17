import logging
from pathlib import Path

from final_experiments.experiment import (
    CFIREExperiment,
    CFIREConfig,
    ThresholdBinarization,
    run_experiment,
)
from final_experiments.logger import init_logger

# Unique experiment name
EXPERIMENTS_NAME = "3_default"

# With how many models/seeds should each experiment be evaluated?
N_MODELS = 10
N_SEEDS = 1

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

    # just use default parameters!
    cfire_configs = [
        CFIREConfig(
            freq_threshold=0.01,
            bin_config=ThresholdBinarization(threshold=0.01),
            max_dt_depth=7,
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
