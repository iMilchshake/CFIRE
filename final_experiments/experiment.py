"""high level entry to run cfire experiments"""

import logging
import os
import sys
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import psutil
import torch
from pebble import ProcessPool
from torch import Tensor

from cfire.cfire_module import CFIRE
from cfire.util import __preprocess_explanations_ext
from lxg.datasets import RandomSeed
from .evaluate import evaluate_cfire
from .models import load_model, PretrainedModel, get_pretrained_models


@dataclass(frozen=True)
class ThresholdBinarization:
    threshold: float


@dataclass(frozen=True)
class TopKBinarization:
    k: int


BinarizationConfig = Union[ThresholdBinarization, TopKBinarization]


@dataclass
class CFIREConfig:
    """Hyperparameters for initializing a cfire instance"""

    freq_threshold: float
    bin_config: BinarizationConfig
    max_dt_depth: int


@dataclass
class CFIREExperiment:
    """Defines an Experiment for various cfire configurations"""

    # which hyperparameter configurations?
    cfire_configs: list[CFIREConfig]

    # on which dataset?
    dataset_name: str

    # and how many samples (#models / #cfire seeds)?
    n_models: int
    n_seeds: int


@dataclass
class CFIRETask:
    """All required data (for an isolated process) to fit and evaluate a cfire config"""

    # for cfire initialization
    cfire_config: CFIREConfig

    # unique identifiers for export
    cfire_config_idx: int
    cfire_seed: int
    model_idx: int
    expl_method: str

    # input data for `.fit()`
    explanations_np: np.ndarray
    X_val_np: np.ndarray
    X_test_np: np.ndarray
    y_val_model_pred_np: np.ndarray
    y_test_model_pred_np: np.ndarray


def initialize_experiment(experiment: CFIREExperiment):
    """load dataset, pretrained models and explanations based on dataset name"""

    # load pretrained models and val / test data
    models, X_val, X_test = get_pretrained_models(
        Path(f"./data/cfire/{experiment.dataset_name}")
    )

    models = models[: experiment.n_models]

    # determine data seed, ensure constraint that only one data seed it used
    data_seeds = set([model.data_seed for model in models])
    assert len(data_seeds) == 1
    data_seed = data_seeds.pop()

    return models, X_val, X_test


def binarize_explanations(x: np.ndarray, *, binning: BinarizationConfig) -> np.ndarray:
    """wrapper function that performs explanation binarization based on a binarization config"""
    if isinstance(binning, ThresholdBinarization):
        return (
            __preprocess_explanations_ext(x, threshold=binning.threshold, top_k=None)
            > 0
        )
    if isinstance(binning, TopKBinarization):
        return __preprocess_explanations_ext(x, threshold=None, top_k=binning.k) > 0
    raise TypeError(f"Unsupported binning config: {type(binning)}")


def init_cfire(task: CFIRETask):
    """initializes a CFIRE object based on a CFIRETask"""
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
    models: list[PretrainedModel], X_val, X_test, experiment: CFIREExperiment
) -> list[CFIRETask]:
    tasks = []
    for model_info in models:
        assert model_info.dataset == experiment.dataset_name

        # precalculate model inputs / predictions / explanations
        model = load_model(model_info.model_dims, model_info.model_path)
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
        for expl_method, expl_path in model_info.explanations.items():
            explanations_np = torch.load(expl_path).detach().cpu().numpy()
            explanations_np.setflags(write=False)

            # construct CFIRETask's
            for cfire_config_idx, cfire_config in enumerate(experiment.cfire_configs):
                for seed in range(experiment.n_seeds):
                    tasks.append(
                        CFIRETask(
                            cfire_config=cfire_config,
                            cfire_config_idx=cfire_config_idx,
                            cfire_seed=seed,
                            model_idx=model_info.model_idx,
                            expl_method=expl_method,
                            explanations_np=explanations_np,
                            X_val_np=X_val_np,
                            X_test_np=X_test_np,
                            y_val_model_pred_np=y_val_model_pred_np,
                            y_test_model_pred_np=y_test_model_pred_np,
                        )
                    )

    return tasks


def run_cfire_task(task: CFIRETask):
    """perform all single threaded cfire steps"""

    with RandomSeed(task.cfire_seed):
        cfire = init_cfire(task)  # initialize cfire inside the process to reduce IPC
    cfire.fit(task.X_val_np, task.y_val_model_pred_np)

    # TODO: test various set covering algorithms (be careful with ILP multi threaded lolz)

    metrics_before_prune = evaluate_cfire(
        cfire,
        task.X_val_np,
        task.X_test_np,
        task.y_val_model_pred_np,
        task.y_test_model_pred_np,
    )

    # rule_metrics_before_prune = compute_rule_metrics(cfire, task.X_val_np)
    # decision = decide_by_wins(rule_metrics_before_prune, win_threshold=0)
    # new_rules = prune_rules(cfire.dnf.rules, decision.to_remove)
    #
    # # temp replace rules
    # old_rules = cfire.dnf.rules
    # cfire.dnf.rules = new_rules
    #
    # rule_metrics_after_prune = compute_rule_metrics(cfire, task.X_val_np)
    #
    # # restore rules
    # cfire.dnf.rules = old_rules
    #
    # metrics_after_prune = evaluate_cfire(
    #     cfire,
    #     task.X_val_np,
    #     task.X_test_np,
    #     task.y_val_model_pred_np,
    #     task.y_test_model_pred_np,
    # )

    return metrics_before_prune


def run_parallel_tasks_with_timeout(tasks, task_fn, timeout, n_workers):
    """Run tasks in parallel with a hard per-task wall-clock timeout."""

    results = [None] * len(tasks)
    total = len(tasks)

    with ProcessPool(max_workers=n_workers) as pool:
        futures = []
        for t in tasks:
            futures.append(pool.schedule(task_fn, args=(t,), timeout=timeout))

        for idx, future in enumerate(futures):
            try:
                results[idx] = future.result()
                logging.info(f"finished task {idx+1}/{total}")
            except FutureTimeout:
                logging.warning(f"Task timed out after {timeout}s: index={idx}")
            except Exception as exc:
                logging.error(f"Task {idx} failed: {exc}")

    return results


def run_experiment(
    experiment: CFIREExperiment, experiments_dir: Path, timeout: int = 120, use_seq=False,
):
    """Run one experiment and store results in the provided directory."""

    logging.info(f"starting experiment: {experiment} -> {experiments_dir}")

    models, X_val, X_test = initialize_experiment(experiment)
    logging.info(f"initialized experiment")

    tasks = init_cfire_tasks(models, X_val, X_test, experiment)
    logging.info(f"initialized {len(tasks)} cfire tasks")

    if use_seq:
        cfire_results = [run_cfire_task(t) for t in tasks]
    else:
        n_cores = psutil.cpu_count(logical=False)  # consider physical cores only
        n_workers = int(
            os.getenv("N_WORKERS", n_cores)
        )  # use n_cores as default fallback
        cfire_results = run_parallel_tasks_with_timeout(
            tasks, run_cfire_task, timeout=timeout, n_workers=n_workers
        )
    logging.info(f"completed {len(tasks)} cfire tasks")

    # save results to disk
    results_path = experiments_dir / "results" / experiment.dataset_name
    results_path.mkdir(parents=True, exist_ok=True)
    success_rows = []
    failed_rows = []

    for task, metrics in zip(tasks, cfire_results):
        row = {
            "model_idx": task.model_idx,
            "cfire_config_idx": task.cfire_config_idx,
            "cfire_seed": task.cfire_seed,
            "expl_method": task.expl_method,
            **task.cfire_config.__dict__,
        }
        if metrics is not None:
            row.update(metrics)
            success_rows.append({**row, **metrics})
        else:
            failed_rows.append(row)

    df_success = pd.DataFrame(success_rows)
    df_success.to_csv(results_path / "results.csv", index=False)
    df_failed = pd.DataFrame(failed_rows)
    df_failed.to_csv(results_path / "failed_runs.csv", index=False)

    logging.info(
        f"saved {len(success_rows)} successful runs and {len(failed_rows)} failed runs"
    )


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
        # "iris", # e.g. here i observer -10% performance? (because we ensure that all classes are predicted? Ah test set is very small xd).
    ]

    experiments = [
        CFIREExperiment(
            dataset_name=dataset_name,
            n_models=50,
            n_seeds=1,
            cfire_configs=[
                CFIREConfig(
                    freq_threshold=0.01,
                    bin_config=ThresholdBinarization(threshold=0.01),
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
