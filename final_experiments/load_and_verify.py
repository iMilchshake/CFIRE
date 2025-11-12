import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from final_experiments.models import get_pretrained_models, load_model
from lxg.models import DNFClassifier
from sklearn.metrics import f1_score


if __name__ == "__main__":
    results_dir = Path("./experiments/5_final_with_artifacts/results")

    for dataset_dir in sorted(results_dir.iterdir()):
        dataset = dataset_dir.name
        task_results_dir = dataset_dir / "task_results"

        print(f"\n{dataset}:")

        # load blackbox models and inputs for dataset
        models, X_val, X_test = get_pretrained_models(Path(f"./data/cfire/{dataset}"))

        for task_path in sorted(task_results_dir.glob("task_*.pkl")):
            # get pre-computed task/artifacts from experiment pipeline
            with open(task_path, 'rb') as f:
                task = pickle.load(f)

            # get model predictions
            model_info = [m for m in models if m.model_idx == task["model_idx"]][0]
            model = load_model(model_info.model_dims, model_info.model_path)
            y_val_pred = model.predict_batch(X_val).detach().cpu().numpy()
            y_test_pred = model.predict_batch(X_test).detach().cpu().numpy()

            # sanity check: predictions match with pre-computed predictions
            assert np.allclose(task["y_val_model_pred"], y_val_pred)
            assert np.allclose(task["y_test_model_pred"], y_test_pred)

            for variant in ["original", "safe_prune", "best_prune"]:

                # get cfire predictions
                dnf = DNFClassifier(task[f"dnf_rules_{variant}"], "accuracy")
                dnf.compute_rule_performance(task["X_val"], task["y_val_model_pred"])
                y_cfire = dnf.predict(task["X_test"])
                f1 = f1_score(task["y_test_model_pred"], y_cfire, average="weighted", zero_division=0)

                # sanity check: f1 matches pre-computed f1
                csv_name = "metrics.csv" if variant == "original" else f"metrics_{variant}.csv"
                df = pd.read_csv(dataset_dir / csv_name)
                row = df[(df["model_idx"] == task["model_idx"]) & (df["expl_method"] == task["expl_method"])].iloc[0]
                saved_f1 = row["test_f1_weighted"]
                assert np.isclose(f1, saved_f1), f"{variant} F1 mismatch!"

            # all asserts passed -> success
            print(f"  task_idx={task['task_idx']}, model_idx={task['model_idx']}, expl_method={task['expl_method']} - all checks passed")

    print("done")
