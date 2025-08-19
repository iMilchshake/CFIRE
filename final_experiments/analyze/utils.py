from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from final_experiments.experiment import ThresholdBinarization

ALL_PARAMS = ["freq_threshold", "max_dt_depth", "bin_config"]

DEFAULT_PARAMS = {
    "freq_threshold": 0.01,
    "max_dt_depth": 7,
    "bin_config": ThresholdBinarization(threshold=0.01),
}

def load_csv_files(root: Path, csv_file_name: str) -> dict[str, pd.DataFrame]:
    """ load csv file for each dataset """
    if not root.exists():
        raise FileNotFoundError("No datasets found")

    out: dict[str, pd.DataFrame] = {}
    for dataset_subdir in root.iterdir():
        if not dataset_subdir.is_dir():
            continue
        csv_file = dataset_subdir / csv_file_name
        if csv_file.exists():
            if csv_file.stat().st_size == 1: # skip empty csv's
                continue
            out[dataset_subdir.name] = pd.read_csv(csv_file)
    return out

def get_stat_str(df: pd.DataFrame, metric_column: str) -> str:
    vals = pd.to_numeric(df[metric_column], errors="raise").dropna() # coerce?
    return "—" if vals.empty else f"{vals.mean():.2f}±{vals.std():.2f}"

def merge_local_explainers(df: pd.DataFrame, group_by: list[str] | None = None) -> pd.DataFrame:
    if group_by is None:
        group_by = ["model_idx"] + ALL_PARAMS
    idx = df.groupby(group_by, dropna=False)["val_acc"].idxmax()
    return df.loc[idx]

def get_metric_table(df_sel: pd.DataFrame, hyperparam: str, metrics: list[str]) -> pd.DataFrame:
    grouped = df_sel.groupby(df_sel[hyperparam], dropna=False)
    rows, index_vals = [], []
    for k, g in grouped:
        index_vals.append(k)
        rows.append({m: get_stat_str(g, m) for m in metrics})
    out = pd.DataFrame(rows, index=index_vals)
    out.index.name = hyperparam
    try:
        out = out.sort_index()
    except Exception:
        pass
    return out.T


def filter_other_params_to_default(df: pd.DataFrame, target_param: str) -> pd.DataFrame:
    """ filter dataframe to fixed default parameters, but keep all values of target_param """
    mask = pd.Series(True, index=df.index)
    for p in ALL_PARAMS:
        if p == target_param:
            continue
        mask &= df[p].astype(str) == str(DEFAULT_PARAMS[p])
    return df.loc[mask]
