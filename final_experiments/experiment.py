"""high level entry to run cfire experiments"""
import logging
import os
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import psutil
import torch
from joblib import Parallel, delayed
from torch import Tensor

from cfire.cfire_module import CFIRE
from cfire.util import __preprocess_explanations_ext
from final_experiments.pruning import decide_by_wins, prune_rules
from final_experiments.pruning_metrics import compute_rule_metrics
from lxg.datasets import dataset_callables, RandomSeed
from .evaluate import evaluate_cfire
from .models import load_or_train_models, ModelFiles, load_model
from .types import CFIREDataset
from .util import loader_to_tensor


@dataclass(frozen=True)
class ThresholdBinarization:
    threshold: float

@dataclass(frozen=True)
class TopKBinarization:
    k: int

BinarizationConfig = Union[ThresholdBinarization, TopKBinarization]


@dataclass
class CFIREConfig:
    """ Hyperparameters for initializing a cfire instance """
    freq_threshold: float
    bin_config: BinarizationConfig
    max_dt_depth: int

@dataclass
class CFIREExperiment:
    """ Defines an Experiment for various cfire configurations """

    # which hyperparameter configurations?
    cfire_configs: list[CFIREConfig]

    # on which dataset?
    dataset_name: str

    # and how many samples (#models / #cfire seeds)?
    n_models: int
    n_seeds: int



@dataclass
class CFIRETask:
    """ All required data (for an isolated process) to fit and evaluate a cfire config"""

    # for cfire initialization
    cfire_config: CFIREConfig

    # unique identifiers for export
    cfire_config_idx: int
    cfire_seed: int
    model_idx: int

    # input data for `.fit()`
    explanations_np: np.ndarray
    X_val_np: np.ndarray
    X_test_np: np.ndarray
    y_val_model_pred_np: np.ndarray
    y_test_model_pred_np: np.ndarray


def initialize_experiment(experiment: CFIREExperiment, experiments_dir: Path):
    """load dataset, train models, get local explanations"""
    model_dir = experiments_dir / "models" / experiment.dataset_name
    model_dir.mkdir(parents=True, exist_ok=True)

    dataset_fn = dataset_callables[experiment.dataset_name]
    dataset = CFIREDataset._make(dataset_fn())  # convert tuple to named tuple

    models = load_or_train_models(dataset, experiment.n_models, model_dir)

    return models, dataset

def binarize_explanations(x: np.ndarray, *, binning: BinarizationConfig) -> np.ndarray:
    """ wrapper function that performs explanation binarization based on a binarization config """
    if isinstance(binning, ThresholdBinarization):
        return __preprocess_explanations_ext(x, threshold=binning.threshold, top_k=None) > 0
    if isinstance(binning, TopKBinarization):
        return __preprocess_explanations_ext(x, threshold=None, top_k=binning.k) > 0
    raise TypeError(f"Unsupported binning config: {type(binning)}")

def init_cfire(task: CFIRETask):
    """ initializes a CFIRE object based on a CFIRETask """
    expl_bin = partial(binarize_explanations, binning=task.cfire_config.bin_config)

    cfire = CFIRE(
        localexplainer_fn=None,
        explanations=task.explanations_np,
        inference_fn=None,
        expl_binarization_fn=expl_bin,
        frequency_threshold=task.cfire_config.freq_threshold,
        max_dt_depth=task.cfire_config.max_dt_depth,
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
        X_test, _ = loader_to_tensor(dataset.test_loader)
        y_val_model_pred: Tensor = model.predict_batch(X_val)
        y_test_model_pred: Tensor = model.predict_batch(X_test)

        # convert to read only numpy arrays
        X_val_np = X_val.detach().cpu().numpy()
        X_test_np = X_test.detach().cpu().numpy()
        y_val_model_pred_np = y_val_model_pred.detach().cpu().numpy()
        y_test_model_pred_np = y_test_model_pred.detach().cpu().numpy()

        X_val_np.setflags(write=False)
        X_test_np.setflags(write=False)
        y_val_model_pred_np.setflags(write=False)
        y_test_model_pred_np.setflags(write=False)

        # load explanations
        explanations_np = torch.load(model_files.expl_path).detach().cpu().numpy()
        explanations_np.setflags(write=False)

        # construct CFIRETask's
        for cfire_config_idx, cfire_config in enumerate(experiment.cfire_configs):
            for seed in range(experiment.n_seeds):
                tasks.append(
                    CFIRETask(
                        cfire_config=cfire_config,
                        cfire_config_idx=cfire_config_idx,
                        cfire_seed=seed,
                        model_idx=model_files.model_idx,
                        explanations_np=explanations_np,
                        X_val_np=X_val_np,
                        X_test_np=X_test_np,
                        y_val_model_pred_np=y_val_model_pred_np,
                        y_test_model_pred_np=y_test_model_pred_np,
                    )
                )

    return tasks


def run_cfire_task(task: CFIRETask):
    """ perform all single threaded cfire steps """

    with RandomSeed(task.cfire_seed):
        cfire = init_cfire(task) # initialize cfire inside the process to reduce IPC
    cfire.fit(task.X_val_np, task.y_val_model_pred_np)

    # TODO: test various set covering algorithms (be careful with ILP multi threaded lolz)

    metrics_before_prune = evaluate_cfire(
        cfire,
        task.X_val_np,
        task.X_test_np,
        task.y_val_model_pred_np,
        task.y_test_model_pred_np,
    )

    rule_metrics_before_prune = compute_rule_metrics(cfire, task.X_val_np)
    decision = decide_by_wins(rule_metrics_before_prune, win_threshold=0)
    new_rules = prune_rules(cfire.dnf.rules, decision.to_remove)

    # temp replace rules
    old_rules = cfire.dnf.rules
    cfire.dnf.rules = new_rules

    rule_metrics_after_prune = compute_rule_metrics(cfire, task.X_val_np)

    # restore rules
    cfire.dnf.rules = old_rules

    metrics_after_prune = evaluate_cfire(
        cfire,
        task.X_val_np,
        task.X_test_np,
        task.y_val_model_pred_np,
        task.y_test_model_pred_np,
    )

    return task, cfire, metrics_after_prune # TODO: non-typed return >:(

def run_experiment(experiment: CFIREExperiment, experiments_dir: Path):
    """ Run one experiment and store results in the provided directory.
    Pass a unique experiment index if multiple experiments are evaluated in the same directory (e.g. grid search). """

    logging.info(f"starting experiment: {experiment} -> {experiments_dir}")

    models, dataset = initialize_experiment(experiment, experiments_dir)
    logging.info(f"initialized experiment")

    tasks = init_cfire_tasks(models, dataset, experiment)
    logging.info(f"initialized {len(tasks)} cfire tasks")

    # run tasks concurrently, collect results
    n_cores = psutil.cpu_count(logical=False)  # consider physical cores only
    n_jobs = int(os.getenv("N_JOBS", n_cores)) # use n_cores as fallback
    cfire_results = Parallel(
        n_jobs=n_jobs,
        prefer="processes",
        verbose=10,
    )(delayed(run_cfire_task)(t) for t in tasks)
    logging.info(f"completed {len(tasks)} cfire tasks")

    # save results to disk
    results_path = experiments_dir / "results" / experiment.dataset_name
    results_path.mkdir(parents=True, exist_ok=True)
    rows = []
    for task, cfire, metrics in cfire_results:
        rows.append(
            {
                "model_idx": task.model_idx,
                "cfire_config_idx": task.cfire_config_idx,
                "cfire_seed": task.cfire_seed,
                **task.cfire_config.__dict__,
                **metrics,
            }
        )

        # TODO: disabled for now until we need it.
        # dump a few relevant attributes of the final cfire object that might be relevant in the future
        # cfire.partial_dump(results_path / "cfire_dumps" / f"cfire_{experiment_idx}_{task.model_idx}_{task.cfire_seed}")

    # TODO: how to deal with already existing results in the future? (overwrite? ensure unique name?)
    df = pd.DataFrame(rows)
    df.to_csv(results_path / f"results.csv", index=False)
    logging.info("finished saving results")


# TODO: this main() is just for testing -> we need some place to define various different experiments.
def main():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    # TODO: also define model directory at top level
    # TODO: in future me might want sub dirs like "gridsearch","setcover",...
    experiments_dir = Path(f"./experiments/test/")

    # TODO: build grid search that creates instances of experiment
    dataset_names = [
        # "diggle", # OpenMLError: Dataset with data_id 694 not found. :(
        # "vehicle", # wow, this takes ~3 minutes per cfire, but paper claims 20 sec?
        "abalone",
        # "beans", # broken interface, requires `random_state` as input?
        "wine",
        "iris", # e.g. here i observer -10% performance? (because we ensure that all classes are predicted? Ah test set is very small xd).
    ]

    experiments = [
        CFIREExperiment(
            dataset_name=dataset_name,
            n_models=2,  # cfire paper uses 50 models
            n_seeds=3,
            cfire_configs=[
                CFIREConfig(
                    freq_threshold=0.01,
                    bin_config=ThresholdBinarization(threshold=0.01),
                    max_dt_depth=7,
                ),
                CFIREConfig(
                    freq_threshold=0.01,
                    bin_config=TopKBinarization(k=2),
                    max_dt_depth=7,
                ),
            ],
        )
        for dataset_name in dataset_names
    ]

    for experiment in experiments:
        run_experiment(experiment, experiments_dir)


if __name__ == "__main__":
    main()
