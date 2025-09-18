import logging

from final_experiments.experiment import (
    CFIREExperiment,
    CFIREConfig,
    ThresholdBinarization,
    initialize_experiment,
    init_cfire_tasks,
    run_cfire_task,
)

if __name__ == "__main__":
    cfire_configs = [CFIREConfig(
            freq_threshold=0.01,
            bin_config=ThresholdBinarization(threshold=0.01),
            max_dt_depth=7)]

    experiment = CFIREExperiment(
        dataset_name="breastw",
        n_models=1,
        n_seeds=1,
        cfire_configs=cfire_configs,
    )

    models, X_val, X_test = initialize_experiment(experiment)
    logging.info(f"initialized experiment")

    tasks = init_cfire_tasks(models, X_val, X_test, experiment)
    logging.info(f"initialized {len(tasks)} cfire tasks")

    task = next(t for t in tasks if t.task_idx == 0)

    print(task)
    print(task.expl_method)

    result = run_cfire_task(task)
    print(result["metrics"])
