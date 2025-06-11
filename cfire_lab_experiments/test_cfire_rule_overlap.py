# this script evaluates CFIRE rules based on a trained model and explanations
# run `test_train.py` before!

from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from pathlib import Path
import random
import pickle

from lxg.datasets import RandomSeed
import lxg.datasets as datasets
from lxg.models import make_ff
from lxg.util import restore_checkpoint
from cfire.cfire_module import CFIRE
from cfire.util import __preprocess_explanations, __preprocess_explanations_ext

from cfire_lab_experiments.util import loader_to_tensor

# init data dirs
model_dir = Path("./models/")
experiment_dir = Path("./experiments/")
model_dir.mkdir(parents=True, exist_ok=True)
experiment_dir.mkdir(parents=True, exist_ok=True)

from .test_cfire import ks_fn_cached, pprint_dnf_rules, rule_size

def main():
    dataset = datasets.get_abalone()
    train_loader, test_loader, val_loader, n_dim, n_classes = dataset
    print(
        f"n_samples\ntrain: {len(train_loader.dataset)}\n val: {len(val_loader.dataset)}\n test: {len(test_loader.dataset)}"
    )
    X_train, y_train = loader_to_tensor(train_loader)
    X_val, y_val = loader_to_tensor(val_loader)
    X_test, y_test = loader_to_tensor(test_loader)
    model = make_ff([n_dim, 128, 128, n_classes], torch.nn.ReLU).to("cpu")
    # expl_binarization_fn = lambda x: __preprocess_explanations(x, filtering=0.01) > 0
    def expl_binarization_fn(x):
        return __preprocess_explanations_ext(x, top_k=2) > 0

    # restore training checkpoint
    model_path = model_dir / "tmp.ckpt"
    restore_checkpoint(model_path, model, train=False)

    y_train_model_pred = model.predict_batch(X_train).numpy()
    y_val_model_pred = model.predict_batch(X_val).numpy()
    y_test_model_pred = model.predict_batch(X_test).numpy()
    print(f"model train accuacy: {np.mean(y_train_model_pred == y_train.numpy())}")
    print(f"model val accuacy: {np.mean(y_val_model_pred == y_val.numpy())}")
    print(f"model test accuacy: {np.mean(y_test_model_pred == y_test.numpy())}")


    # run CFIRE
    seed = 42
    with RandomSeed(seed):
        cfire = CFIRE(
            localexplainer_fn=ks_fn_cached(model_dir / "explanations.pt"),
            inference_fn=model.predict_batch_softmax,
            expl_binarization_fn=expl_binarization_fn,
        )
        cfire.fit(X_val.numpy(), y_val_model_pred)

        val_cfire_out = cfire(X_val, explain=True)
        y_val_cfire_pred = np.array([t[0] for t in val_cfire_out])
        print(cfire.dnf.rules)
        print(cfire.dnf.rule_performances)
        val_acc = np.mean(y_val_model_pred == y_val_cfire_pred)
        print(val_acc)


        # ------------------------------------------------------------
        # 1. build a “clause × sample” hit-matrix from the CFIRE output
        # ------------------------------------------------------------

        cfire_out = val_cfire_out

        # collect every unique clause-key (cls_id, clause_id) in a stable order
        clause_keys = []
        for _, matches in cfire_out:
            for key, _ in matches:                   # key == (class_id, clause_idx)
                if key not in clause_keys:
                    clause_keys.append(key)

        key2col = {k: i for i, k in enumerate(clause_keys)}   # fast look-up

        n_samples  = len(cfire_out)
        n_clauses  = len(clause_keys)
        M = np.zeros((n_samples, n_clauses), dtype=bool)      # hit-matrix

        for s, (_, matches) in enumerate(cfire_out):
            for key, _ in matches:
                M[s, key2col[key]] = True

        # ------------------------------------------------------------
        # 2. sample-level statistics ( “how many clauses fire per sample?” )
        # ------------------------------------------------------------
        match_counts   = M.sum(axis=1)                         # integer array, size = n_samples
        collision_rate = (match_counts > 1).mean()

        print(f"share of samples covered by ≥2 clauses: {collision_rate:.2%}")
        print("histogram of #-matches per sample:")
        for k in np.unique(match_counts):
            print(f"  {k} : {(match_counts == k).sum()}")

        # ------------------------------------------------------------
        # 3. clause-level statistics (coverage and competitors)
        # ------------------------------------------------------------
        coverage = M.sum(axis=0)                               # samples per clause

        # overlap counts between every pair of clauses
        overlap_matrix = (M.astype(int).T @ M.astype(int))     # shape (n_clauses, n_clauses)

        class_of = np.array([cls for cls, _ in clause_keys])
        intra = np.zeros(n_clauses, int)                       # competitors inside the same class
        inter = np.zeros(n_clauses, int)                       # competitors from other classes

        for j in range(n_clauses):
            for k in range(j + 1, n_clauses):                  # j < k  (upper triangle)
                if overlap_matrix[j, k] == 0:
                    continue
                if class_of[j] == class_of[k]:
                    intra[j] += 1
                    intra[k] += 1
                else:
                    inter[j] += 1
                    inter[k] += 1

        print("\nTop 10 clauses by inter-class competition:")
        order = np.argsort(-inter)[:10]
        for idx in order:
            cls, cid = clause_keys[idx]
            print(f"cl {cls:2d} / term {cid:2d} | covers {coverage[idx]:4d} samples | "
                  f"intra competitors {intra[idx]:3d} | inter competitors {inter[idx]:3d}")

        # ------------------------------------------------------------
        # 4. strongest clause–clause overlaps (any class)
        # ------------------------------------------------------------
        pairs = []
        for j in range(n_clauses):
            for k in range(j + 1, n_clauses):
                cnt = overlap_matrix[j, k]
                if cnt > 0:
                    pairs.append((cnt, j, k))

        pairs.sort(reverse=True)                               # descending by cnt

        print("\nTop 20 clause-pair overlaps:")
        for cnt, j, k in pairs[:20]:
            same = class_of[j] == class_of[k]
            print(f"{clause_keys[j]}  <->  {clause_keys[k]} : {cnt:4d} samples "
                  f"({'same class' if same else 'different classes'})")

        # -> generally ALOT of overlap (inter and intra), 96%> points have more than 1 matching bounding box???
        # 1. it would be interesting to see how much of the overlap disappears by applying tie breaker
        #       -> can rules be heavily pruned by considering this?
        #       -> can i re-arrange rules, so that overlapping regions in which one rule wins anyway,
        #       we just make the loosing rule smaller? (we'd need to re-calculate the performance metrics tho)
        # 2. maybe also check the actual overlap of rules? compare areas? (how to do that for rules of different number of dimensions?)

if __name__ == "__main__":
    main()
