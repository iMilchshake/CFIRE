import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import pickle

from lxg.datasets import _get_dim_classes
from lxg.models import make_ff
from lxg.util import restore_checkpoint, load_meta_data

def load_model(model_dims: tuple[int], model_path: Path) -> nn.Module:
    model = make_ff(model_dims, torch.nn.ReLU).to("cpu")
    restore_checkpoint(model_path, model, train=False)
    return model

@dataclass(frozen=True)
class PretrainedModel:
    dataset: str
    model_idx: int
    data_seed: int
    model_dims: tuple[int]
    model_path: Path
    explanations: Dict[str, Path]  # method -> path
    test_acc: float

def get_pretrained_models(path: Path) -> tuple[List[PretrainedModel], torch.Tensor, torch.Tensor]:
    models_dir = path / "models"
    expl_dir = path / "explanations"
    outputs_dir = path / "outputs"
    assert models_dir.exists(), f"no models dir found for {path}"
    assert expl_dir.exists(), f"no expl dir found for {path}"
    assert outputs_dir.exists(), f"no outputs dir found for {path}"

    # get model dimensions
    meta_data = load_meta_data(path)

    assert len(meta_data) == 1
    meta_data = meta_data[next(iter(meta_data))]
    hidden_dims = meta_data["modelparams"]
    X_test = meta_data["X"] # for some holy reason meta uses X/Y for test...
    X_val = meta_data["X_val"]
    assert meta_data["kwargs_data"] is None # TODO: just a test
    dim, n_classes = _get_dim_classes(meta_data["dataset"])
    model_dims = [dim] + hidden_dims + [n_classes]
    del meta_data

    ckpt_pattern  = re.compile(r"^(?P<dataset>.+?)_(?P<model_id>\d+)_(?P<data_seed>\d+)\.ckpt$")
    expl_pattern  = re.compile(r"^(?P<dataset>.+?)_(?P<model_id>\d+)_(?P<data_seed>\d+)_(?P<method>.+)\.pt$")

    out: List[PretrainedModel] = []
    for ckpt_path in sorted(models_dir.glob("*.ckpt")):
        m = ckpt_pattern.match(ckpt_path.name)
        assert m, f"Unexpected filename: {ckpt_path.name}"
        dataset   = m.group("dataset")
        model_id  = int(m.group("model_id"))
        data_seed = int(m.group("data_seed"))

        # load corresponding *_out.pkl
        out_pkl_path = outputs_dir / f"{dataset}_{model_id}_{data_seed}_out.pkl"
        assert out_pkl_path.exists(), f"no out.pkl found for {dataset}_{model_id}_{data_seed}"
        with out_pkl_path.open("rb") as _f:
            model_out = pickle.load(_f)
            test_accuracy = model_out["accuracy"]

        explanations: Dict[str, Path] = {}
        for pt_path in expl_dir.glob(f"{dataset}_{model_id}_{data_seed}_*.pt"):
            m2 = expl_pattern.match(pt_path.name)
            if not m2:
                continue
            method = m2.group("method")
            explanations[method] = pt_path

        out.append(
            PretrainedModel(
                dataset=dataset,
                model_idx=model_id,
                model_dims=model_dims,
                data_seed=data_seed,
                model_path=ckpt_path,
                explanations=explanations,
                test_acc=test_accuracy
            )
        )
    return out, X_val, X_test
