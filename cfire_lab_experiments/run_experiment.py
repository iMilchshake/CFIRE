"""high level entry to run cfire experiments"""
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from torch.utils.data import DataLoader

from cfire_lab_experiments.util import loader_to_tensor
from cfire_lab_experiments.utils_cfire import init_cfire, load_model, fit_cfire
from lxg.datasets import dataset_callables
from .utils_model import init_model_and_explanations


@dataclass
class CFIREExperiment:
    """stores configuration for *one* experiment (= one configuration of cfire parameters)"""

    dataset: str
    n_models: int
    n_seeds: int

    freq_threshold: float
    bin_threshold: float
    # bin_top_k: int
    max_dt_depth: int


class CFIREDataset(NamedTuple):
    train_loader: DataLoader
    test_loader: DataLoader
    val_loader: DataLoader
    n_dim: int
    n_classes: int


def initialize_experiment(experiment: CFIREExperiment):
    """load dataset, train models, get local explanations"""
    model_dir = Path(f"./experiments/models/{experiment.dataset}/")
    model_dir.mkdir(parents=True, exist_ok=True)
    dataset_fn = dataset_callables[experiment.dataset]

    dataset = CFIREDataset._make(dataset_fn())  # convert tuple to named tuple

    paths = init_model_and_explanations(dataset, experiment.n_models, model_dir)
    return paths, dataset

def initialize_cfire_instances(paths, dataset: CFIREDataset, experiment: CFIREExperiment):
    cfire_instances = []
    for model_path, expl_path in paths:
        for seed in range(experiment.n_seeds):
            model = load_model(dataset.n_dim, dataset.n_classes, model_path)
            cfire = init_cfire(
                model=model,
                explanation_path=expl_path,
                frequency_threshold=experiment.freq_threshold,
                bin_threshold=experiment.bin_threshold,
                bin_top_k=None, # TODO: add this
                max_dt_depth=experiment.max_dt_depth
            )
            cfire_instances.append((cfire, model)) # idea is to add them to a list so we can later parallelize this?

def main():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    # define experiment
    experiment = CFIREExperiment(
        dataset="abalone",
        n_models=2,
        n_seeds=2,
        freq_threshold=0.01,
        bin_threshold=0.01,
        max_dt_depth=7,
    )

    paths, dataset = initialize_experiment(experiment)


            # TODO: but for now we just do it sequentially

    logging.info("loading validation set")
    X_val, y_val = loader_to_tensor(dataset.val_loader)
    for cfire, model in cfire_instances:
        logging.info(f"fitting cfire")
        y_val_model_pred = model.predict_batch(X_val)
        cfire.fit(X_val.numpy(), y_val_model_pred.numpy())
        logging.info("done")


if __name__ == "__main__":
    main()
