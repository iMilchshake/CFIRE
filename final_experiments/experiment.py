"""high level entry to run cfire experiments"""

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import torch
from joblib import Parallel, delayed
from joblib.parallel import default_parallel_config
from torch import Tensor

from cfire.cfire_module import CFIRE
from cfire.util import __preprocess_explanations_ext
from cfire_lab_experiments.util import loader_to_tensor
from cfire_lab_experiments.utils_cfire import load_model
from final_experiments.models import load_or_train_models, ModelFiles
from final_experiments.types import CFIREDataset
from lxg.datasets import dataset_callables, RandomSeed


@dataclass
class CFIREExperiment:
    """stores configuration for one experiment evaluating one set of cfire parameters for multiple models and seeds"""

    dataset: str
    n_models: int
    n_seeds: int

    freq_threshold: float
    bin_threshold: float
    # bin_top_k: int
    max_dt_depth: int


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

    models = load_or_train_models(dataset, experiment.n_models, model_dir)
    return models, dataset


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

    expl_bin: Callable = (lambda x: __preprocess_explanations_ext(x, threshold=bin_threshold, top_k=bin_top_k) > 0)

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


def init_cfire_tasks(
    models: list[ModelFiles], dataset: CFIREDataset, experiment: CFIREExperiment
) -> list[CFIRETask]:
    tasks = []
    for model_files in models:

        # precalculate model inputs / predictions / explanations
        model = load_model(dataset.n_dim, dataset.n_classes, model_files.model_path)
        X_val, _ = loader_to_tensor(dataset.val_loader)
        y_val_model_pred: Tensor = model.predict_batch(X_val)

        # convert to read only numpy arrays
        X_val_np = X_val.detach().cpu().numpy()
        y_val_model_pred_np = y_val_model_pred.detach().cpu().numpy()
        X_val_np.setflags(write=False)
        y_val_model_pred_np.setflags(write=False)

        # load explanations
        explanations_np = torch.load(model_files.expl_path).detach().cpu().numpy()
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


def run_cfire_task(task: CFIRETask):
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
        n_models=3,
        n_seeds=12,
        freq_threshold=0.01,
        bin_threshold=0.01,
        max_dt_depth=7,
    )

    models, dataset = initialize_experiment(experiment)
    logging.info(f"initialized experiment")

    tasks = init_cfire_tasks(models, dataset, experiment)
    logging.info(f"initialized {len(tasks)} cfire tasks")

    # run tasks concurrently
    n_workers = 6
    logging.info(f"starting processing with n_workers={n_workers}")
    Parallel(
        n_jobs=n_workers,
        prefer="processes",
        verbose=50,
    )(delayed(run_cfire_task)(t) for t in tasks)
    logging.info("DONE")

if __name__ == "__main__":
    main()
