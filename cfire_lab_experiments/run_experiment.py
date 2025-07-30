"""high level entry to run cfire experiments"""

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Optional, Callable

import numpy as np
import torch
from joblib import Parallel, delayed
from joblib.parallel import default_parallel_config
from torch.utils.data import DataLoader

from cfire.cfire_module import CFIRE
from cfire.util import __preprocess_explanations_ext
from cfire_lab_experiments.util import loader_to_tensor
from cfire_lab_experiments.utils_cfire import load_model
from lxg.datasets import dataset_callables, RandomSeed
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


@dataclass
class CFIRETask:
    cfire_seed: int
    explanations_np: np.ndarray
    X_val_np: np.ndarray
    y_val_model_pred_np: np.ndarray
    exp: CFIREExperiment


def initialize_experiment(experiment: CFIREExperiment):
    """load dataset, train models, get local explanations"""
    model_dir = Path(f"./experiments/models/{experiment.dataset}/")
    model_dir.mkdir(parents=True, exist_ok=True)
    dataset_fn = dataset_callables[experiment.dataset]

    dataset = CFIREDataset._make(dataset_fn())  # convert tuple to named tuple

    paths = init_model_and_explanations(dataset, experiment.n_models, model_dir)
    return paths, dataset


def init_cfire(
    explanations: np.ndarray,
    frequency_threshold: float,
    bin_threshold: Optional[float],
    bin_top_k: Optional[int],
    max_dt_depth: int,
):
    """high level wrapper to run a CFIRE experiment"""

    if sum(arg is not None for arg in (bin_threshold, bin_top_k)) != 1:
        raise ValueError("Specify either bin_threshold OR bin_top_k")

    expl_bin: Callable = (
        lambda x: __preprocess_explanations_ext(
            x, threshold=bin_threshold, top_k=bin_top_k
        )
        > 0
    )

    cfire = CFIRE(
        localexplainer_fn=None,
        explanations=explanations,
        inference_fn=None,
        expl_binarization_fn=expl_bin,
        frequency_threshold=frequency_threshold,
        max_dt_depth=max_dt_depth,
    )
    cfire._verbose = False  # disable debug prints
    return cfire


def init_tasks(
    paths, dataset: CFIREDataset, experiment: CFIREExperiment
) -> list[CFIRETask]:
    tasks = []
    for model_path, expl_path in paths:

        # precalculate model inputs / predictions
        model = load_model(dataset.n_dim, dataset.n_classes, model_path)
        X_val_t, _ = loader_to_tensor(dataset.val_loader)
        y_val_model_pred_t = model.predict_batch(X_val_t)

        X_val_np = X_val_t.detach().cpu().numpy()
        y_val_model_pred_np = y_val_model_pred_t.detach().cpu().numpy()
        X_val_np.setflags(write=False)
        y_val_model_pred_np.setflags(write=False)

        # load explanations
        explanations_np = torch.load(expl_path).detach().cpu().numpy()
        explanations_np.setflags(write=False)

        for seed in range(experiment.n_seeds):
            tasks.append(
                CFIRETask(
                    cfire_seed=seed,
                    explanations_np=explanations_np,
                    X_val_np=X_val_np,
                    y_val_model_pred_np=y_val_model_pred_np,
                    exp=experiment,
                )
            )

    return tasks


def _run_task(task: CFIRETask):
    print(
        f"types: X={type(task.X_val_np)}, Y={type(task.y_val_model_pred_np)}, "
        f"E={type(task.explanations_np)}"
    )
    with RandomSeed(task.cfire_seed):
        cfire = init_cfire(
            explanations=task.explanations_np,
            frequency_threshold=task.exp.freq_threshold,
            bin_threshold=task.exp.bin_threshold,
            bin_top_k=None,  # TODO: add this?
            max_dt_depth=task.exp.max_dt_depth,
        )
    cfire.fit(task.X_val_np, task.y_val_model_pred_np)


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    # define experiment
    experiment = CFIREExperiment(
        dataset="abalone",
        n_models=1,
        n_seeds=15,
        freq_threshold=0.01,
        bin_threshold=0.01,
        max_dt_depth=7,
    )

    paths, dataset = initialize_experiment(experiment)
    tasks = init_tasks(paths, dataset, experiment)

    for n_workers in [5, 5, 5]:
        logging.info(f"starting n_workers={n_workers}")
        t0 = time.time()
        print(default_parallel_config)
        Parallel(
            n_jobs=n_workers,
            prefer="processes",
            verbose=50,
        )(delayed(_run_task)(t) for t in tasks)
        elapsed = time.time() - t0
        logging.info(f"n_workers={n_workers} -> t={elapsed}")


if __name__ == "__main__":
    main()
