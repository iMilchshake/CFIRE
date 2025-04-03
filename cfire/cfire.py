# standard packages
import logging
import os, argparse
from copy import deepcopy

from tqdm import tqdm
from pathlib import Path
from functools import reduce
from time import time
from joblib import Parallel, delayed, parallel_backend
from itertools import product

# environment packages
import torch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, silhouette_score

from scipy.spatial.distance import pdist

# local files
import lxg
from lxg.datasets import nlp_tasks, NumpyRandomSeed, make_classification, _return_dataset
from lxg.util import (load_losses, load_accuracies, load_idxs_from_multiple_explanations, _load_meta_data_by_pid, \
    load_idxs_from_multiple_outputs, load_sklearn_outputs,\
    dump_pkl, load_pkl, load_idxs_from_multiple_models, load_sklearn_models,
    _get_outputs, _get_targets, safe_call)

from lxg.models import DNFClassifier, SimpleTorchWrapper

from cfire.util import __preprocess_explanations
import cfire._variables_cfire as _variables

from .gely import gely, gely_discriminatory

def print_data_cfg(cfg):
    n_feat, n_info, n_redu, n_rep, n_classes = cfg['n_features'], cfg['n_informative'], cfg['n_redundant'], cfg['n_repeated'], cfg['n_classes']
    n_uninf = n_feat - n_rep - n_redu - n_info
    print(f"{n_feat} ({n_info}, {n_redu}, {n_rep}, {n_uninf}):")


def play_around():

    _dataset_configs = [
    # implicit: n_useless = n_features - n_informative - n_redundant - n_repeated
    dict(n_samples=10_000, n_features=20, n_informative=4, n_redundant=2, n_repeated=0, n_classes=4, n_clusters_per_class=2, random_state=42),  # defaults; n_u=14
    dict(n_samples=10_000, n_features=20, n_informative=5, n_redundant=5, n_repeated=5, n_classes=4, n_clusters_per_class=2, random_state=42),  # n_u=5
    dict(n_samples=10_000, n_features=20, n_informative=10, n_redundant=0, n_repeated=10, n_classes=4, n_clusters_per_class=2, random_state=42),  # n_u=0
    dict(n_samples=10_000, n_features=20, n_informative=10, n_redundant=10, n_repeated=0, n_classes=4, n_clusters_per_class=2, random_state=42),  # n_u=0

    ]

    layers = [16, 16, 16]

    from sklearn.tree import DecisionTreeClassifier as DT
    for config in _dataset_configs:
        config['class_sep'] = 0.1
        print("#############")
        print_data_cfg(config)
        for y_flip in [0., 0.01, 0.1]:
            print(f"yflip{y_flip}")
            X, y = make_classification(**config, shuffle=False, flip_y=y_flip)
            n_samples = config['n_samples']
            train_size = 0.8
            batch_size_te = int(n_samples - train_size * n_samples)
            (X_tr, Y_tr), (X_te, Y_te), n_dim, n_classes = _return_dataset(X, y, train_size=train_size,
                                                                           batch_size=[32, batch_size_te],
                                                                           as_torch=False, random_state=0)
            (X_tr2, Y_tr2), (X_te2, Y_te2), _, _ = _return_dataset(X, y, train_size=train_size,
                                                                           batch_size=[32, batch_size_te],
                                                                           as_torch=False, random_state=0)
            assert np.all(X_tr2 == X_tr)

            _layers = [config['n_features']] + layers + [config['n_classes']]
            sk_dt = DT(max_depth=10)#(max_depth=10, max_leaf_nodes=10)

            sk_dt.fit(X_tr, Y_tr)

            print(f"\ttrain acc sk DT: {sk_dt.score(X_tr, Y_tr)}")
            print(f"\ttest acc sk DT: {sk_dt.score(X_te, Y_te)}")
            print(f"\tn_leaves: {sk_dt.get_n_leaves()}")
            print(f"\tdepth: {sk_dt.get_depth()}")
            print()
        print()

def exp1_make_classification_DT_vs_NN():
    '''
    for all task in classification_tasks
        for all argset in product([task], expl_methods_names,
                                         model_combinations[task],
                                         gely_threshold,
                                         explanation_relevance_thresholds,
                                         kmeans_max_bins,
                                         n_samples_percent,
                                         [matrix_mode:='sequential']
                                      )
        ])
            __exp1_make_classification_DT_vs_NN(**argset)  # compute_eval_rules
            
    '''
    import _variables_cfire
    classif_root = Path('./data/cfire/')
    classif_tasks = []
    for d in classif_root.iterdir():
        if d.is_dir() and 'classification' in str(d):
            classif_tasks.append(str(d).split('/')[-1])

    for task in classif_tasks:
        data_dir = './data/cfire/' + task + '/'
        classif_task_cfg = lxg.util.get_classification_config_from_fname(task)
        model_dir = Path(data_dir, 'models')
        results_dir = _variables_cfire.get_result_dir(task)

        model_dir = data_dir + 'models/';
        expl_dir = data_dir + 'explanations/';
        outputs_dir = data_dir + 'outputs/';
        losses_dir = data_dir + 'losses/';
        acc_dir = data_dir + 'accuracies/';

        model_seeds = lxg.util.get_all_model_seeds_sorted(data_dir)
        print(f"found {len(model_seeds)} models")
        # ----------------------------------------------------------

        metadata = _load_meta_data_by_pid(data_dir)  # also contains all train and test samples
        outputs = load_idxs_from_multiple_outputs(data_dir, model_seeds, idxs=[-1])
        for o in outputs:
            o['y_pred'] = o['output_distribution'].argmax(1).cpu().numpy
        explanations = load_idxs_from_multiple_explanations(data_dir, model_seeds, idxs=[-1], explanation_method='vg')


#
# def _is_frequent(pattern, D, threshold):
#     c = 0
#     if len(pattern) == 0:
#         return False, c
#     for d in D:
#         if pattern.issubset(d):
#             c += 1
#     is_frequent = c >= threshold
#     return is_frequent, c
#
#
# def _it(X, D) -> set:
#     '''return: all transactions that contain all items in X'''
#
#     tids = []
#
#     for x in X:  # set of items
#         _tids = []
#         for tid, y in enumerate(D):
#             if x in y:
#                 _tids.append(tid)
#         if len(_tids) > 0:
#             tids.append(set(_tids))
#
#     if len(tids) == 0:
#         return set()
#     _tids_all_x_appear_in = set.intersection(*tids)
#     return _tids_all_x_appear_in
#
#
# def _ti(Y, D) -> set:
#     '''return: all items common to all transactions in Y'''
#     '''_ti(_it(X)) -> can never be empty (if |Y| > 0), because at least all X have to be returned'''
#     if len(Y) == 0:
#         return set()
#     iids = [D[tid] for tid in Y]
#     return set.intersection(*iids)
#
#
# ## frequency check and closure operator using binary matrix instead of sets and list
#
# def _ti_binary_matrix(Y, B) -> set:
#     iids = B[list(Y), :]
#     intersected = reduce(lambda a,b: np.bitwise_and(a, b), list(iids))
#     intersected = np.argwhere(intersected).squeeze()
#     if len(intersected.shape) == 0:
#         intersected = np.expand_dims(intersected, 0)
#     intersected = list(intersected)
#     return set(intersected)
#
#
# def _it_binary_matrix(X, B) -> set:
#     tids = [set(list(np.argwhere(B[:, x]).squeeze(1))) for x in X]
#     return set.intersection(*tids)
#
#
# def _is_frequent_binary(itemset, B, threshold):
#     if len(itemset) == 0:
#         return False, 0
#     _itemset_binary = np.zeros(B.shape[1], dtype=int)
#     _itemset_binary[list(itemset)] = 1
#     match = B @ _itemset_binary
#     match = match == len(itemset)
#     n_occurences = np.sum(match)
#     is_frequent = n_occurences >= threshold
#     return is_frequent, n_occurences
#
#
# def check_class_variable(_C_prime, class_item_ids):
#     if class_item_ids is None:
#         return True
#     return len(_C_prime.intersection(class_item_ids)) == 1
#
#
# def list_closed(C, N, i, t,
#                   D, I, T, results, closure=None, D_binary=None, class_item_ids=None):
#
#     if closure is None:
#         closure = lambda X: _ti(_it(X, T, D), I, D)
#
#     idx = np.argwhere(I == i).squeeze()  # {k in I\C : k >= i}
#     X = I[idx:]
#     X = set(X) - C
#
#     if len(X) > 0:
#         _i_prime = min(X)
#         _C_prime: set = closure(C.union({_i_prime}))
#
#         if D_binary is not None:
#             is_frequent, count = _is_frequent_binary(_C_prime, D_binary, t)
#         else:
#             is_frequent, count = _is_frequent(_C_prime, D, t)
#
#         if is_frequent and len(_C_prime.intersection(N)) == 0:  # and check_class_variable(_C_prime, class_item_ids):
#             # print(sorted(_C_prime), count)
#             results.append((_C_prime, count))
#             idx = np.argwhere(I == _i_prime).squeeze()
#             if idx >= len(I)-1:  # we reached item with highest ordinal
#                 return
#             _next_i = I[idx + 1]  # _i_prime + 1  # leads to Index Out Of Range Error on I
#             list_closed(_C_prime, N, _next_i,  # What if min(X) == max(I)?
#                         t, D, I, T, results, closure, D_binary, class_item_ids)
#
#         idx = np.argwhere(I == _i_prime).squeeze()+1  # {k in I\C: k > _i_prime}
#         Y = I[idx:]  # no out of bounds, just empty
#         Y = set(Y) - C
#         if len(Y) > 0:
#             _i_prime_prime = min(Y)
#             list_closed(C, N.union({_i_prime}), _i_prime_prime, t,
#                         D, I, T, results, closure, D_binary, class_item_ids)
#
#     return
#
#
# import sys
#
# class recursionlimit:
#     def __init__(self, limit):
#         self.limit = limit
#
#     def __enter__(self):
#         self.old_limit = sys.getrecursionlimit()
#         sys.setrecursionlimit(self.limit)
#
#     def __exit__(self, type, value, tb):
#         sys.setrecursionlimit(self.old_limit)
#
#
#
# def gely(B, threshold, use_binary=False, remove_copmlete_transactions=True, targets=None, verbose=True):
#     '''
#
#     :param D: Is a binary matrix;
#               Columns = Items, index used as ordering
#               Tows = Transactions, index used as tid
#     :param threshold: frequency threshold used to determine if an itemset is considered frequent
#     :return:
#     '''
#
#     if 0 < threshold < 1:
#         threshold = int(B.shape[0] * threshold)
#         # print(f'threshold set to {threshold}')
#
#     n_full_transactions = None
#     if remove_copmlete_transactions:
#         n_I = B.shape[1]
#         _T_sizes = np.sum(B, 1)
#         unfull_transactions = _T_sizes < n_I
#         D = B[unfull_transactions]
#         n_full_transactions = np.sum(-1*(unfull_transactions-1))
#         # print(f"removed {n_full_transactions} transactions that contained all items")
#         if threshold < 1:
#             threshold = max(int((B.shape[1] - n_full_transactions) * threshold), 1)
#         else:
#             threshold = threshold - n_full_transactions
#             if threshold <= 0:
#                 threshold = 1
#             # print(f"adapted threshold to {threshold+n_full_transactions} - {n_full_transactions} = {threshold}")
#     else:
#         D = B
#
#     _size_T, _size_I = D.shape
#     T, I = np.arange(_size_T), np.arange(_size_I)
#     #  remove items that never occur in any transaction in D (ie, filter zero columns)
#     _non_zero_cols = np.argwhere(np.sum(D, 0) > 0).squeeze()
#     I = I[_non_zero_cols]
#     _t = threshold
#
#     class_item_ids = None
#     if targets is not None:
#         n_classes = len(np.unique(targets))
#         _targets_binarized = np.zeros([B.shape[0], n_classes])
#         _targets_binarized[:, targets] = 1
#         D = np.hstack([D, _targets_binarized])
#         class_item_ids = set(np.arange(D.shape[1] - n_classes, D.shape[1]))
#     # transform binary rows into transactions represented by lists of items (as indices)
#     _D = []
#     for _i, t in enumerate(D):
#         t_iids = np.argwhere(t).squeeze()
#         if len(t_iids.shape) == 0:
#             t_iids = np.expand_dims(t_iids, 0)
#         _D.append(list(t_iids))
#     _D = [set(d) for d in _D]
#
#     if use_binary:
#         D = D.astype(int)
#         closure = lambda X: _ti_binary_matrix(_it_binary_matrix(X, D), D)  # D not _D !
#     else:
#         closure = lambda X: _ti(_it(X, _D), _D)
#
#     _fcis = []
#     import numbers
#     if isinstance(I, numbers.Number):
#         I = [I]
#     try:
#         list_closed(
#             C=set(), N=set(), i=min(I), t=_t, D=_D, I=I, T=T,
#             results=_fcis, closure=closure, D_binary=D if use_binary else None,
#             class_item_ids=class_item_ids
#         )
#     except RecursionError as e:
#         if hasattr(e, 'message'):
#             if "recursion depth exceeded" in e.message:
#                 old_rlimit = sys.getrecursionlimit()
#                 new_rlimit = int(old_rlimit * 1.5)
#                 print(f"raising recursion limit from {old_rlimit} to {new_rlimit}")
#                 del _fcis
#                 _fcis = []
#                 with recursionlimit(new_rlimit):
#                     list_closed(
#                         C=set(), N=set(), i=min(I), t=_t, D=_D, I=I, T=T,
#                         results=_fcis, closure=closure, D_binary=D if use_binary else None,
#                         class_item_ids=class_item_ids
#                     )
#
#
#     # return list sorted by (largest itemsets, larger support)
#     _fcis = sorted(_fcis, key=lambda x: (-x[1], len(x[0])), reverse=True)
#     if n_full_transactions is not None:
#         _fcis = [(f[0], f[1]+n_full_transactions) for f in _fcis]
#     return _fcis
#
#
# def gely_test():
#     D = ['abde', 'bce', 'abde', 'abce', 'abcde', 'bcd']
#     support_thresh = 3
#     I = ['a', 'b', 'c', 'd', 'e']
#     _I = {k: v for v, k in enumerate(I)}
#
#     T = np.arange(len(D))
#     B = np.zeros((len(T), len(I)))
#     for tid, t in zip(T, D):
#         for _iid, i in enumerate(I):
#             B[tid, _iid] = 1 if i in t else 0
#
#     assert np.all(B ==
#                   np.array([
#                             [1, 1, 0, 1, 1],
#                             [0, 1, 1, 0, 1],
#                             [1, 1, 0, 1, 1],
#                             [1, 1, 1, 0, 1],
#                             [1, 1, 1, 1, 1],
#                             [0, 1, 1, 1, 0]
#                         ])
#                   )
#
#     fcis = gely(B, support_thresh)
#     for f, c in fcis:
#         print(f"{[I[i] for i in f]} x {c}")
#     pass

# ----------------------------------------------------------------------------------------------------------------------

def greedy_set_cover(X: set[int], F: list[set[int]]) -> list[int]:

    U = X.copy()
    C = []

    while len(U) > 0:
        _intersects = [len(U.intersection(f)) for f in F]
        _f_idx = np.argmax(_intersects)
        f = F[_f_idx]
        U = U - f
        C.append(_f_idx)

    return C



# ----------------------------------------------------------------------------------------------------------------------

def _kmeans1d(samples, n_bins):

    _KM = KMeans(n_clusters=n_bins, n_init=1, max_iter=300, init='k-means++', random_state=42)
    _KM.fit(np.expand_dims(samples, 1))
    return _KM.labels_


def _bins_from_labels(samples, labels):
    bins = []
    for l in np.unique(labels):
        samples_l = samples[labels == l]
        bins.append((min(samples_l), max(samples_l)))

    return bins

def _silhouette_score(samples, labels):
    n_labels = len(np.unique(labels))
    if n_labels == 1:
        _min, _max = np.min(samples), np.max(samples)
        return 1 if abs(_max - _min)**2 < 1e-3 else 0  # TODO magic number
    else:
        return silhouette_score(samples, labels)


def __kmeans_binning(samples, n_bins, n_jobs=None):

    if n_jobs is None:
        n_jobs = max(1, int(os.cpu_count()*0.9))

    # parallelize over dimensions because all dimensions will contain the same number of datapoints
    n_dims = len(samples)
    bins = []
    empty = [i for i, dim in enumerate(samples) if len(dim) == 0 ]  # all values in dim are zero

    if n_bins > 1:
        n_bins_per_dim = [min(n_bins, int(np.ceil(len(np.unique(dim))/5))) for dim in samples] # min 1 bin and maximally fraction of #samples
        _parallel_args = [(samples[dim], _n_bins) for dim in np.arange(n_dims) for _n_bins in np.arange(1, n_bins_per_dim[dim]+1) if n_bins_per_dim[dim] > 0]
        _labelings_to_dim_idx = []
        for i, n_bins in enumerate(n_bins_per_dim):
            _labelings_to_dim_idx.extend([i]*n_bins)  # x.extend([i]*0) = x
        assert len(_labelings_to_dim_idx) == sum(n_bins_per_dim)
        _parallel_sil_args = [(_labelings_to_dim_idx[i], i) for i in range(sum(n_bins_per_dim))]
        backend = 'threading'
        assert backend == 'threading', "DONT DO PROCESS BASED PARALLELISM HERE, " \
                                       "IF KMEANS BINNING IS CALLED FROM A PARALLEL ENVIRONMENT. " \
                                       "THIS CAUSED A MEMORY LEAK LAST TIME"
        with parallel_backend(backend=backend, n_jobs=1):
            labelings = Parallel()(delayed(_kmeans1d)(samples[dim], _n_bins)
                              for dim in np.arange(n_dims) for _n_bins in np.arange(1, n_bins_per_dim[dim]+1) if n_bins_per_dim[dim] > 0)
            sil_scores = Parallel()(delayed(_silhouette_score)(np.expand_dims(samples[_labelings_to_dim_idx[i]], 1), list(labelings[i]))
                                                              for i in np.arange(len(labelings)))
        sil_scores_nested = []
        labelings_nested = []
        idx_count = 0
        for _n_bins in n_bins_per_dim:
            if _n_bins == 0: continue
            sil_scores_nested.append(sil_scores[idx_count:idx_count+_n_bins])
            labelings_nested.append(labelings[idx_count:idx_count+_n_bins])
            idx_count += _n_bins
        assert (n_dims - sum(np.array(n_bins_per_dim)==0)) == len(sil_scores_nested)
        idxs_max_scores = [np.argmax(sil_score) for sil_score in sil_scores_nested]
        chosen_labelings = [labelings_nested[i][idx] for i, idx in enumerate(idxs_max_scores)]
        for idx in empty:
            idxs_max_scores.insert(idx, np.nan)
            chosen_labelings.insert(idx, np.nan)
        for i, samples_dim in enumerate(samples):
            if i in empty:
                bins.append([])
                continue
            bins.append(_bins_from_labels(samples_dim, chosen_labelings[i]))
    elif n_bins == 1:
        for i, samples_dim in enumerate(samples):
            if i in empty:
                bins.append([])
                continue
            _labels = np.zeros((len(samples_dim),))
            bins.append(_bins_from_labels(samples_dim, _labels))
    else:
        raise ValueError


    return bins


def kmeans_binning(e, n_bins=40, n_jobs=None, positive=True):

        # if positive, filter all out but > 0, else. the other way around
        _e = e if positive else -1 * e
        _filtered_by_sign = [_e.T[i, mask] for i, mask in enumerate((_e>0.).T)]
        if not positive:
            _filtered_by_sign = [-1 * f for f in _filtered_by_sign]
        return __kmeans_binning(_filtered_by_sign, n_bins, n_jobs)


def mean_distance(x): # 1-d array
    x = np.expand_dims(x, 1)
    mean_dist = np.mean(pdist(x, metric='euclidean'))
    return mean_dist


def __compute_bins_mean_dist(e, heuristic=mean_distance, max_bins=40, positive=True):
    # compute bins, if positive: for value >0, if not positive: values <0
    _direction = 1 if positive else -1
    _de = _direction * e
    _de = np.clip(_de, 0, np.max(_de))
    # compute bins, ignoring 0.

    _col_heur = []
    _ranges = []
    for col in _de.T:
        v = col[col > 0]
        if len(v) == 0:  # no values
            _ranges.append([np.NaN, np.NaN])
            _col_heur.append(np.NaN)
        elif len(v) == 1:
            _ranges.append([v[0], v[0]])
            _col_heur.append(0)
        else:
            _ranges.append([np.min(v), np.max(v)])
            _col_heur.append(heuristic(v))

    # _n_bins = [0 if min(r) is np.nan
    #                 else min(max_bins, np.round((max(r)-min(r))/ch))
    #                     if not np.isclose(ch, 0.)
    #                         else 1
    #            for r, ch in zip(_ranges, _col_heur)]
    _n_bins = []
    for r, ch in zip(_ranges, _col_heur):
        if min(r) is np.nan:
            _n_bins.append(0)
        elif np.isclose(ch, 0.):
            _n_bins.append(1)
        else:
            if max(r) == min(r):
                print(r)
            _n_bins.append(min(max_bins, np.round((max(r)-min(r))/ch)))

    print(f"got {min(_n_bins)} to {max(_n_bins)} bins")

    _bins = []
    for r, nb in zip(_ranges, _n_bins):
        # print(f"{r}, {nb}")
        if nb == 0:
            _bins.append([])
        elif nb == 1:
            _bins.append([[min(r), max(r)]])
        else:
            b = list(np.arange(min(r), max(r), (max(r)-min(r))/nb))+[max(r)]
            if not positive:
                b = [-1*bb for bb in b][::-1]
            b = [[b1, b2] for b1, b2 in zip(b[:-1], b[1:])]
            _bins.append(b)

    return _bins


def compute_bins(e, compute_bins_fn=kmeans_binning, binning_args=None):

    _bins_pos = compute_bins_fn(e, positive=True, **binning_args)
    _bins_neg = compute_bins_fn(e, positive=False, **binning_args)

    _bins = []
    for _bp, _bn in zip(_bins_pos, _bins_neg):
        _bins.append(_bn + _bp)

    _bins = [sorted(binning) for binning in _bins]

    for dim, binning in enumerate(_bins):
        # sanity check that no bins overlap and that all dimensions eitherhave bins or do not contain any non-zero attribution scores
        assert len(binning) > 0 or np.all(e[:, dim] == 0.)  # there have to be bins if the dim has any attributions
        if len(binning) == 1: continue
        for ((a1, b1), (a2, b2)) in zip(binning[:-1], binning[1:]):
            if a1 < a2:
                assert b1 < a2
            if a2 < a1:
                assert b2 < b1

    return _bins

# ----------------------------------------------------------------------------------------------------------------------

def binarize_explanations(e, bins):
    # binarize each dim independently
    binarized_rows = []

    for row in e:
        _binarized_row = []
        for attribution_score, binning in zip(row, bins):
            if len(binning) == 0:
                _b = [0]
            else:
                _b = np.zeros(len(binning))
                for i, (_start, _end) in enumerate(binning):
                    if type(attribution_score) is np.ndarray:
                        print("warn")
                    if _start <= attribution_score <= _end:
                        _b[i] = 1.
                        break
            # assert np.sum(_b) == 1.
            _binarized_row.append(_b)

        _binarized_row_merged = np.hstack(_binarized_row)
        binarized_rows.append(_binarized_row_merged)

    binarized_columns = np.array(binarized_rows)
    binarized_columns = np.ascontiguousarray(binarized_columns)
    return binarized_columns

# ----------------------------------------------------------------------------------------------------------------------



def item_to_model_bin(itemsets, bins, n_dims, n_models) -> list[tuple[int, int, tuple]]:
    item_model_bin = []
    for m in range(n_models):
        for dim in range(n_dims):
            bins_idx = m * n_dims + dim
            if len(bins[bins_idx]) == 0:
                item_model_bin.append((m, dim, (np.nan, np.nan)))
            else:
                for bin in bins[bins_idx]:
                    item_model_bin.append((m, dim, tuple(bin)))
    _converted = []
    for items, support in itemsets:
        _converted.append([])
        for item in items:
            _converted[-1].append((item_model_bin[item]))
        # sort items by key=(model, dim, bin)
        _converted[-1] = sorted(_converted[-1], key=lambda x: (x[0], x[1], -x[2][0]))
        _converted[-1].insert(0, support)
    # sort itemsets by support
    _converted = sorted(_converted, key=lambda x: -x[0])
    return _converted


def time_func(f, args):
    s1 = time()
    results = f(**args)
    elapsed = time() - s1
    return results, elapsed


def map_attr_sets_to_data_space(data, rules):
    '''
        given rules in explanation space and data each rule applies to, extract intervals of feature values covered by the points
        in data space.
        the actual values of the attribution scores are actually ignored

        this method only looks at marked dimensions in the support set of a rule and extracts min-max intervals.
        eg if there is a large gap between any two clusters in the support set, this method ignores it.
        maybe another hyperparameter that defines max distance between two clusters before breaking it up into two rules?
    '''
    rules_data_space = []
    for rule, _data in zip(rules, data):
        clauses = [(idx, start, end) for _, idx, (start, end) in rule]
        # collect all idxs
        dims = np.unique([c[0] for c in clauses])
        _picked_dims = _data[:, dims]

        _ranges = [(dim, (_min, _max)) for dim, _min, _max in
                   # zip(dims, np.min(_picked_dims, axis=0), np.max(_picked_dims, axis=0))]
                   # zip(dims, np.percentile(_picked_dims, axis=0, q=48), np.percentile(_picked_dims, axis=0, q=52))]
                   zip(dims, np.mean(_picked_dims, axis=0)-0.1, np.mean(_picked_dims, axis=0)+0.1)]

        rules_data_space.append(_ranges)
        # filter: if two rules share all idxs and the intervals overlap then merge them

    # to make np.unique work we append those dummy rules
    dummy_rules = [[(np.nan, (np.nan, np.nan))], [(np.nan, (np.nan, np.nan)), (np.nan, (np.nan, np.nan))]]
    rules_data_space.extend(dummy_rules)
    rules_data_space_arr = np.asarray(rules_data_space, dtype=object)
    unique_rules_data_space = np.unique(rules_data_space_arr).tolist()
    unique_rules_data_space = [r for r in unique_rules_data_space if r not in dummy_rules]

    # print(unique_rules_data_space)
    # merged_rules = []
    # lens = np.unique([len(r) for r in rules_data_space])
    # for l in lens:  # go through all rules of different length
    #     _rsl = [r for r in rules_data_space if len(r) == l]  # get rules of respective length
    #     # get all unique combinations of dimensions checked in rules of len l
    #     _dims_rs = list(dict.fromkeys([tuple(sorted(_r[0] for _r in r)) for r in _rsl]))
    #     # get all rules by those keys
    #     _dim_rsl_unique = []
    #     for _unique_dims in _dims_rs:
    #         _dim_rsl_unique.append([r for r in _rsl if tuple([_r[0] for _r in r]) == _unique_dims])
    #     for _potentially_equal_rules in _dim_rsl_unique:
    #         merged = merge_rules(_potentially_equal_rules)

    return unique_rules_data_space


def model_selection(task):

    _, n_classes = lxg.datasets._get_dim_classes(task)
    print(f'starting eval on {task} dataset')
    data_dir = _variables.get_data_dir(task)
    print(f'loading from {data_dir}')
    result_dir = _variables.get_result_dir(task)
    print(f'result dir: {result_dir}')


    fname = f"{task}_model_itemsets.txt"
    if Path(result_dir, fname).is_file():
        print("TODO, FILE FOUND, NOT LOADED")

    # model_seeds = get_all_model_seeds_sorted(data_dir)
    # print(f"found {len(model_seeds)} models")
    losses, _accuracies = load_losses(data_dir), load_accuracies(data_dir)

    selected_seed_fname = Path(result_dir, f"{task}_selected_models.csv")  # the rashomon set selected from all trained models
    with open(selected_seed_fname, 'r') as f:
        model_seeds = [str(int(seed)) for seed in f.readlines()]
    print(f"found selected seeds\ncontinuing with {len(model_seeds)} models\n\n")
    model_seeds = model_seeds[:int(len(model_seeds)/2)]

    # meta_data = load_meta_data(data_dir, just_one=False)
    meta_data = _load_meta_data_by_pid(data_dir)
    _data = [d['X'] for d in meta_data.values()]
    _targets = [d['Y'] for d in meta_data.values()]
    eq = []

    for i in range(len(_data)):
        for j in range(i, len(_data)):
            eq.append(_data[i] == _data[j])
    assert torch.all(torch.stack(eq))  # assert that all metadata in dir uses same testset 'X' <-> we don't have issue with randomzation


    print("loading models")
    # _models = load_idxs_from_multiple_models(data_dir, task, model_seeds, idxs=[-1], return_fns=True)
    _outputs = [o['output_distribution'] for o in load_idxs_from_multiple_outputs(data_dir, model_seeds, [-1])]
    _model_predictions = [torch.argmax(o, dim=1) for o in _outputs]

    # X_orig = _data[0]
    Y_true = _targets[0]
    # need to find 'a large set of models that agrees on a large set of points'
    # -> prioritize number of datapoints or number of models?
        # this sounds like set covering?
    B = [Y_true == mp for mp in _model_predictions]
    B = torch.vstack(B).detach().to(torch.int8).numpy()
    filter_min_one_correct = np.argwhere(np.sum(B, 0) > 0)  # keep only points where at least one model is correct
    # _labels_filtered = Y_true[filter_min_one_correct].squeeze()
    # _data_filtered = X_orig[filter_min_one_correct].squeeze()  # filter out data where all models are wrong

    B_filtered = B[:, filter_min_one_correct].squeeze()  # Filter points where all models are wrong
    n_one_correct = len(filter_min_one_correct)
    n_all_correct = np.sum(np.sum(B, 0) == len(B))
    _correct_ratio = 8/10  # magic number
    n_min_thresh_correct = int(_correct_ratio*(n_one_correct - n_all_correct) + n_all_correct )
    _fcis = []
    s1 = time()
    n_tries = 1
    while len(_fcis) == 0 and n_tries <= 10:
        print(f"support threshold: {n_min_thresh_correct}")
        _fcis = gely(B_filtered.T, threshold=n_min_thresh_correct, use_binary=False)
        if len(_fcis) == 0 or max([len(_f[0]) for _f in _fcis]) < 7:  # we either found none or too small imtemsets
            n_min_thresh_correct = np.ceil(n_min_thresh_correct*0.99)-1
            _fcis = []
            n_tries += 1
    if n_tries > 10:
        if len(_fcis) == 0:
            print(f"no fcis found with support of {n_min_thresh_correct}, abort")
        print(f"n_tries reached!")
        _largest_fcis_idx = np.argmax([len(_f[0]) for _f in _fcis])
        print(f"largest fcis has size: {len(_fcis[_largest_fcis_idx][0])}:")
        print(_fcis[_largest_fcis_idx])

    s2 = time()
    print(_fcis[:20])
    print(f"model selection for {task} took {n_tries} in {s2-s1} seconds")
    fname = f"{task}_model_itemsets.txt"
    print(f"saving to {fname}")
    with open(Path(result_dir, fname), 'w') as f:
        f.writelines(f"{model_seeds}\n")
        f.writelines("\n".join([str(_set) for _set in _fcis]))
        f.writelines('\n')
    print("done")


def load_model_itemset_seeds(task):
    from _variables_cfire import get_result_dir
    rdir = get_result_dir(task)
    fname = f"{task}_model_itemsets.txt"

    with open(Path(rdir, fname), 'r') as f:
        head = f.readline()
        seeds = eval(head[12:])
        sets = [eval(s) for s in f.readlines()]

    lens = [len(s[0]) for s in sets]
    selected_set_idxs = []
    target_len = 7
    while len(selected_set_idxs) == 0:
        for i, l in enumerate(lens):
            if l == target_len:
                selected_set_idxs.append(i)
        target_len -= 1
    candidates = [sets[i] for i in selected_set_idxs]
    largest_support_candidate = sorted(candidates, key=lambda x:x[1])[-1]
    selected_seeds = [seeds[i] for i in largest_support_candidate[0]]
    return selected_seeds


def rename(task, expl_method, model_set, gely_threshold, significance_threshold, k_means_max_bins):
    model_set_str = '['+'-'.join(model_set)+']'
    fname = f"{task}_{expl_method}_{gely_threshold}_{significance_threshold}_{k_means_max_bins}_{model_set_str}.pkl"
    fname_old = f"{task}_{expl_method}_{gely_threshold}_{significance_threshold}_{model_set_str}.pkl"

    result_dir = Path(_variables.get_result_dir(task), 'rule_models')
    _variables.__create_dir(result_dir)

    fname_new = f"t-{task}_ex-{expl_method}_gt-{gely_threshold}_st-{significance_threshold}_km-{k_means_max_bins}_s-{model_set_str}.pkl"

    if k_means_max_bins == 40:
        if Path(result_dir, fname_old).exists() and Path(result_dir, fname_old).is_file():
            os.rename(str(Path(result_dir, fname_old)), str(Path(result_dir, fname_new)))
            print(f"old file {fname_old} -> ", end='')
            print(f"renamed to {fname_new}")
    else: #  k_means_max_bins == 1
        if Path(result_dir, fname).exists() and Path(result_dir, fname).is_file():
            os.rename(str(Path(result_dir, fname)), str(Path(result_dir, fname_new)))
            print(f"old file {fname} -> ", end='')
            print(f"renamed to {fname_new}")
    return

def tree_baseline(task):
    import _variables_interpret as _variables
    _, n_classes = lxg.datasets._get_dim_classes(task)
    data_dir = _variables.get_data_dir(task)
    result_dir = Path(_variables.get_result_dir(task), 'rule_models')
    _variables.__create_dir(result_dir)
    meta_data = _load_meta_data_by_pid(data_dir)
    data_seed = next(iter(meta_data.values()))['data_seed']

    dataset_callable = lxg.datasets._get_dataset_callable(task)
    # yes naming is correct here
    (test_x, test_y), (trainx, trainy), dim, classes = dataset_callable(random_state=data_seed, as_torch=False)

    from sklearn.tree import DecisionTreeClassifier as DT

    sk_dt = DT(max_depth=5, max_leaf_nodes=10)

    sk_dt.fit(test_x, test_y)


    print(f"train acc sk DT: {sk_dt.score(test_x, test_y)}")
    print(f"test acc sk DT: {sk_dt.score(trainx, trainy)}")
    print(f"n_leaves: {sk_dt.get_n_leaves()}")
    print(f"depth: {sk_dt.get_depth()}")
    return

def __compute_rules_exp_space(explanations, data, gely_threshold, k_means_max_bins=2,
                              setcover_reduction=True, verbose=True, gely_sort_items=False) -> dict:
        binning_args = dict(n_jobs=4, n_bins=k_means_max_bins)
        bin_borders = compute_bins(explanations, binning_args=binning_args)
        expls_binarized = binarize_explanations(explanations, bin_borders)

        if gely_sort_items is not None and type(gely_sort_items) == bool:
            item_order = np.argsort(np.sum(expls_binarized, 0)).squeeze()
        elif gely_sort_items is not None and type(gely_sort_items) == np.ndarray:
            item_order = gely_sort_items
        else:
            item_order = np.arange(expls_binarized.shape[1])
        # (B, threshold, use_binary=False, remove_copmlete_transactions=True, targets=None, verbose=True):
        gely_args = {'B': expls_binarized,
                     'use_binary': False,
                     'threshold': gely_threshold,
                     'item_order': item_order
                     }
        # (B, threshold, X, Y, target_label, remove_copmlete_transactions=True)
        # gely_discriminatory_args = {
        #     'B': expls_binarized,
        #     'threshold': gely_threshold,
        #     'X': DATA ALL CLASSES,
        #     'Y': LABELS ALL CLASSES
        # }

        _gely_threshold = gely_threshold
        if verbose:
            print(f"compute fcis support = {gely_args['threshold']}")
        _fcis2 = []
        n_tries = 0
        max_tries = np.inf# 2 if gely_threshold >= 0.5 else 3
        while len(_fcis2) == 0 and n_tries < max_tries:
            # _fcis2, time2 = time_func(gely, gely_args)
            _fcis2, time2 = lxg.util.timed_task(gely, gely_args)
            # _fcis2, time2 = time_func(gely_discriminatory, gely_args)
            if len(_fcis2) == 0:
                if n_tries > max_tries:
                    print(f"gely: number of tries exceeded maximum of {max_tries}")
                    return []
                _gely_threshold -= 0.01  # if we don't find any, it terminates fast
                if _gely_threshold < 0.01:
                    print(f"no fcis in {n_tries} tries, last threshold:{_gely_threshold+0.01}")
                    return []
                gely_args['threshold'] = _gely_threshold
                n_tries += 1
            if verbose:
                print(f"computing fcis took: {time2}")
        if n_tries > max_tries:
            print(f"no fcis in {n_tries} tries, last threshold:{_gely_threshold}")
            print(f"\t aborted")
            return []
        ## 1. for each f in fcis obtain list of idxs of all samples that are covered by f
        __support = []
        __all_items = []
        for __fcis in _fcis2:
            _fcis_support = []
            for i, e in enumerate(expls_binarized):
                applicable = True
                for f in __fcis[0]:
                    if not e[f]:
                        applicable = False
                        break
                if applicable:
                    _fcis_support.append(i)
                    __all_items.append(i)
            __support.append(set(_fcis_support))

        if len(_fcis2) > 1 and setcover_reduction:
            _covered_set = set(np.unique(__all_items))
            _filtered_rules_idxs = greedy_set_cover(_covered_set, __support)
            _filtered_rules = [_fcis2[fr] for fr in _filtered_rules_idxs]
        else:
            # coverage = len(expls_binarized)/len(__support[0])
            _filtered_rules = _fcis2
        # _expl_coverage.append(coverage)
        del __all_items

        ## 2. compute set cover, returning list of indices of fcis to keep
        ## replace _fcis2 with reduced set of _fics,
        ## continue as before

        # convert fcis to model+attr space
        converted_fcis = item_to_model_bin(_filtered_rules, bin_borders,
                                           n_dims=data.shape[-1], n_models=1)  #len(model_seeds))
        if verbose:
            print(converted_fcis)

        # get the set of all datapoints a rule in model+attr space applies to
        _data_idx_for_rules = []
        _data_for_rules = []
        for fcis in converted_fcis:
            rules = fcis[1:]
            expl_idxs_matching = []
            _data_idx_for_rules.append([])
            _data_for_rules.append([])
            for i, expl in enumerate(explanations):
                matches = True
                for rule in rules:
                    model_idx = rule[0]
                    dim_idx = rule[1]
                    interval_start, interval_end = rule[2][0], rule[2][1]
                    _e_val = expl[data.shape[-1] * model_idx + dim_idx]
                    if not interval_start <= _e_val <= interval_end:
                        matches = False
                        break
                if matches:
                    _data_idx_for_rules[-1].append(i) # idxs support
                    _data_for_rules[-1].append(data[i]) # idxs
        _data_for_rules = [np.vstack(d) for d in _data_for_rules]
        # TODO rules[1:] deleted support information, add it back in again
        _supports = [len(d) for d in _data_for_rules]

        results = {
            'rules_expl': converted_fcis,
            'support_idxs_rules': _data_idx_for_rules,
            'support_set_rules': _data_for_rules,
            'supports': _supports,
            'final_gely_threshold': _gely_threshold,
        }
        return results

def compute_rules_exp_space(task: str, expl_method: str, model_set: list[int], gely_threshold: float,
                            significance_threshold: float, k_means_max_bins=40, n_samples_percent=1., seed=42,
                            setcover_reduction=True, verbose=False):
    # For one model only
    # ------------------------------------------------------------------------
    model_set_str = '['+'-'.join(model_set)+']'
    # if '314029' in model_set_str:
    #     print('sus model, skip')
    #     return

    # fname = f"{task}_{expl_method}_{gely_threshold}_{significance_threshold}_{k_means_max_bins}_{model_set_str}.pkl"
    # fname_old = f"{task}_{expl_method}_{gely_threshold}_{significance_threshold}_{model_set_str}.pkl"
    filetype = '.pkl'
    fname = f"t-{task}_ex-{expl_method}_gt-{gely_threshold}_st-{significance_threshold}_km-{k_means_max_bins}_" + \
             f"s-{model_set_str}_fs-{n_samples_percent}_sc-{setcover_reduction}"+filetype

    # ------------------------------------------------------------------------
    import _variables_cfire as _variables
    _, n_classes = lxg.datasets._get_dim_classes(task)
    if verbose:
        print(f'starting eval on {task} dataset')
    data_dir = _variables.get_data_dir(task)
    if verbose:
        print(f'loading from {data_dir}')
    expl_rule_dir = Path(_variables.expl_rule_dir, task)
    _variables.__create_dir(expl_rule_dir)
    if verbose:
        print(f'result dir: {expl_rule_dir}')
    # ------------------------------------------------------------------------

    # if (Path(result_dir, fname).exists() and Path(result_dir, fname).is_file()):
    #
    # sys.exit()

    print(f"starting {fname}")

    model_seeds = model_set

    meta_data = _load_meta_data_by_pid(data_dir)
    data_seed = next(iter(meta_data.values()))['data_seed']
    _data = [d['X'] for d in meta_data.values()]
    _targets = [d['Y'] for d in meta_data.values()]
    eq = []

    for i in range(len(_data)):
        for j in range(i, len(_data)):
            eq.append(_data[i] == _data[j])
    # assert that all metadata in dir uses same testset 'X' <-> we don't have issue with randomzation
    assert torch.all(torch.stack(eq))

    # _models = load_idxs_from_multiple_models(data_dir, task, model_seeds, idxs=[-1], return_fns=True)
    _outputs = [o['output_distribution'] for o in load_idxs_from_multiple_outputs(data_dir, model_seeds, [-1])]
    _model_predictions = [torch.argmax(o, dim=1).cpu() for o in _outputs]

    if verbose:
        print("loading models")
    X_orig = _data[0]
    Y_true = _targets[0]
    # need to find 'a large set of models that agrees on a large set of points'
    # -> prioritize number of datapoints or number of models?
        # this sounds like set covering?
    _B = [Y_true == mp for mp in _model_predictions]
    _nB = [torch.logical_not(Y_true) == mp for mp in _model_predictions]
    _nBi = [np.argwhere(nb.detach().numpy()).squeeze() for nb in _nB]
    np.array([len(np.setdiff1d(_nBi[i], _nBi[j])) for i in range(len(_nBi)) for j in range(i + 1, len(_nBi))])
    B = torch.vstack(_B).detach().to(torch.int8).numpy()
    data_idxs_all_models_agree = np.argwhere(np.sum(B, 0) == len(B))
    # now filter out all but majority class
    if verbose:
        _bin_count_classes = np.bincount(Y_true[data_idxs_all_models_agree].squeeze(), minlength=max(Y_true))
        print(f"class distribution: {_bin_count_classes}")
    # selected_class = np.argmax(_bin_count_classes)

    # LOAD EXPLANATIONS ---------------------------------------------------------------------------------------#
    expls_all_models = load_idxs_from_multiple_explanations(data_dir, model_seeds, idxs=[-1], explanation_method=expl_method)[0]
    # expls = [e[data_idxs_all_models_agree] for e in expls]

    expls_all_models = [e[expl_method] for e in expls_all_models]

    if 'ig' in expl_method:
        expls_all_models = [e for e, _ in expls_all_models]  # because ig is (attrs, delta)
    # ([explanations], epoch_batch) -> keep only explanations because we only look at last batch
    expls_all_models = [e.cpu().detach().numpy() for e in expls_all_models]

    if task in nlp_tasks and len(expls_all_models[0].shape) > 2:
        expls_all_models = [np.sum(e, -1) for e in expls_all_models]

    # expls = np.stack(expls)

    # COMPUTE RULES IN EXPLANATION AND DATASPACE FOR EACH CLASS ---------------------------------------------------#

    _all_classes_attr_rules = []
    _all_classes_rules = []
    _expl_coverage = []
    _all_results = []
    for idx_model, _model_num in enumerate(model_seeds):
        results_model = {}

        filetype = '.pkl'
        fname = (f"t-{task}_ex-{expl_method}_gt-{gely_threshold}_st-{significance_threshold}_km-{k_means_max_bins}_"
                 f"s-[{_model_num}]_fs-{n_samples_percent}_sc-{setcover_reduction}") + filetype
        # if Path(expl_rule_dir, fname).exists() and Path(expl_rule_dir, fname).is_file():
        #     print(f"found {str(Path(expl_rule_dir, fname))}")
        #     return


        start_time = time()
        expls_preproc = __preprocess_explanations(expls_all_models[idx_model], filtering=significance_threshold)

        for selected_class in range(n_classes):

            if verbose:
                print(f"choosing class #{selected_class}")
            # correct label
            _idxs_class = np.argwhere(Y_true == selected_class).squeeze()
            # correct label intersect with correctly classified by model
            _filtered_idxs_class = np.intersect1d(np.argwhere(_B[idx_model]), _idxs_class)
            ## apply filter to data and label vectors
            __data_filtered_class = X_orig[_filtered_idxs_class]
            __labels_filtered_class = Y_true[_filtered_idxs_class]

            if n_samples_percent < 1.:
                _n_samples_class = len(_filtered_idxs_class)
                _n_samples_remain = max(2, int(np.ceil(n_samples_percent * _n_samples_class)))
                with NumpyRandomSeed(seed):
                    np.random.shuffle(_filtered_idxs_class)
                    _filtered_idxs_class = _filtered_idxs_class[:_n_samples_remain]

            expls_filtered = expls_preproc[_filtered_idxs_class, :]

            results_model[selected_class] = __compute_rules_exp_space(expls_filtered, __data_filtered_class,
                                                                      gely_threshold, k_means_max_bins,
                                                                      setcover_reduction, verbose)
            pass
            # _rules_in_data_space = map_attr_sets_to_data_space(_data_for_rules, [rules[1:] for rules in converted_fcis])
        end_time = time()
        results_model['args'] = dict(task=task, expl_method=expl_method, gely_threshold=gely_threshold,
                                     model_seed=_model_num, significance_threshold=significance_threshold,
                                     k_means_max_bins=k_means_max_bins, n_samples_percent=n_samples_percent,
                                     setcover_reduction=setcover_reduction
                     )
        dump_pkl(results_model, Path(expl_rule_dir, fname))
        _all_results.append(results_model)
    return _all_results


def __compute_rules_discr(explanations, data_target, data_other, gely_threshold, k_means_max_bins=2,
        setcover_reduction=True, gely_sort_items=False, verbose=True, model_callable=None, compute_rules=True) -> dict:

    item_order = None
    if type(explanations) is list:
        # skip binning!
        _expls_counting = np.zeros_like(explanations[0])
        for e in explanations:
            e_ceiled = np.ceil(e) # ceil to set all values to 1. that are not 0. <-> binarizes e, accumulates expls_binarized
            _expls_counting += e_ceiled

        if len(explanations) == 2:
            expls_binarized = _expls_counting > 0
            item_order = np.argsort(np.sum(_expls_counting, 0)).squeeze()[::-1]
        elif len(explanations) > 2:
            # threshold for majority vote; require strict majority for even and uneven numbers of explanations
            if len(explanations) % 2 == 0:  # even
                th = (len(explanations) / 2.)-1  # eg, 4 methods -> require 3 for majority
            else:
                th = np.floor(len(explanations)/2.)   # 3 methods, require 2 for majority
            expls_binarized = (_expls_counting > th).astype(float)
            _expls_counting *= expls_binarized  # remove all counts below thresdhold
            item_order = np.argsort(np.sum(_expls_counting, 0)).squeeze()[::-1]

        else:
            raise ValueError

    else:

        # binning_args = dict(n_jobs=4, n_bins=k_means_max_bins)
        # bin_borders = compute_bins(explanations, binning_args=binning_args)
        # expls_binarized = binarize_explanations(explanations, bin_borders)
        expls_binarized = (explanations > 0).astype(int)

    if type(explanations) is list:
        assert item_order is not None
    elif gely_sort_items:
        # item_order = np.argsort(np.sum(expls_binarized, 0)).squeeze()[::-1]
        item_order = np.argsort(np.sum(explanations, 0)).squeeze()[::-1]
    else:
        item_order = np.arange(expls_binarized.shape[1])

    gely_args = {'B': expls_binarized,
                 'threshold': gely_threshold,
                 'X_target': data_target,
                 'X_other': data_other,
                 'item_order': item_order,
                 'model_callable': model_callable,
                 'compute_rules': compute_rules,
                 }
    _gely_threshold = gely_threshold
    if verbose:
        print(f"compute fcis support = {gely_args['threshold']}")
    _fcis2 = []
    n_tries = 0
    max_tries = np.inf  # 2 if gely_threshold >= 0.5 else 3
    assert len(expls_binarized) > 0
    if (m:=np.mean(expls_binarized)) < gely_threshold:
        if m == 0:
            return []
        gely_threshold = m*0.99
        gely_args['threshold'] = gely_threshold
    while len(_fcis2) == 0 and n_tries < max_tries:
        # _fcis2, time2 = time_func(gely_discriminatory, gely_args)
        gely_args['subset_test'] = True
        _fcis2, time2 = time_func(gely_discriminatory, gely_args)
        # _fcis2, time2 = time_func(gely_discriminatory, gely_args)
        if len(_fcis2) > 0:
            if len(_fcis2) == 1:
                print("what's up?")
            c = _fcis2[1][0].get_frequent_children()
            if len(c) == 0:
                print("no frequent children?")
            if compute_rules:
                r = [cc.dnf.rules for cc in c]
                assert all([len(rr)>0 for rr in r]) # assert that for all frequent nodes we also have a rule
                assert not any([rr == [(-1, (np.nan, np.nan))] for rr in r])
        else:# len(_fcis2) == 0:

            if n_tries > max_tries:
                print(f"gely: number of tries exceeded maximum of {max_tries}")
                return []
            _gely_threshold -= 0.01  # if we don't find any, it terminates fast
            if _gely_threshold < 0.01:
                print(f"no fcis in {n_tries} tries, last threshold:{_gely_threshold + 0.01}")
                return []
            gely_args['threshold'] = _gely_threshold
            n_tries += 1

        if verbose:
            print(f"computing fcis took: {time2}")
    return _fcis2


def cega_test(X_tr, Y_tr, attributions, X_te, Y_te, apriori_max_len, association_rule_mining_thresh,
              result_queue=None):
    import pandas as pd
    from multiprocessing import Pool
    
    import CEGA.helper_func as cega_help
    from CEGA.helper_func import intervals_dict, pos_queue, neg_queue  # wtf
    import CEGA.rules_model as cega_rulemodel
    from mlxtend.frequent_patterns import apriori, association_rules

    if len(intervals_dict) > 0:
        print(f"intervals dict not empty")
        intervals_dict.clear()
        print(f"cleared dict")


    start_time = time()
    '''
    from notebook:
    model_prediction = Y_tr
    # extract itemsets
    
    '''
    if type(Y_tr) == torch.Tensor:
        Y_tr = Y_tr.detach().cpu().numpy()
    if type(Y_te) == torch.Tensor:
        Y_te = Y_te.detach().cpu().numpy()

    pred = Y_tr# y_dev  # predictions from model to be explained
    pos_label = '1'
    neg_label = '0'

    def data_to_pd(x):
        if type(x) == torch.Tensor:
            x = x.cpu().detach().numpy()

        n_feat = x.shape[1]
        col_names = [f'V{i}' for i in range(1, n_feat+1)]
        x_pd = pd.DataFrame(x, columns=col_names)
        return x_pd

    def attr_to_pd(a):
        return pd.DataFrame(a)
        # return data_to_pd(a)

    X_train_pd = data_to_pd(X_tr)
    pd_attributions = attr_to_pd(attributions).T
    # pd_attributions = attr_to_pd(np.ones_like(X_dev)).T

    if len(intervals_dict) == 0:
        cega_help.compute_intervals(intervals_dict, X_train_pd)
    itemset = set()
    encoded_vals = []
    summed_values = {}
    shap_threshold = 0.001
    p = Pool(5)  # Pool(num_cores)

    for feature in X_train_pd.columns.to_list():
        if feature in intervals_dict:
            intervals = intervals_dict[feature]
            for interval in intervals:
                if interval != interval: continue
                left = interval.left
                right = interval.right
                name = f'{left}<{feature}<={right}'
                itemset.add(name)
        else:
            itemset.add(feature)

    itemset.add(pos_label)
    itemset.add(neg_label)
    feature_names = [f'V{i+1}' for i in range(X_train_pd.shape[1])]
    for indx in tqdm(range(len(X_train_pd))):
        pos_queue.put(pos_label)
        neg_queue.put(neg_label)

        zipped = zip(pd_attributions[indx].to_numpy().tolist(), pd_attributions[indx].array.tolist(),
        feature_names, [shap_threshold]*len(feature_names))

        # get_relevant_features accesses pos/neg_queue via global variable
        p.map(cega_help.get_relevant_features, zipped)

        cega_help.append_to_encoded_vals(pos_queue, itemset, encoded_vals)
        cega_help.append_to_encoded_vals(neg_queue, itemset, encoded_vals)
    # pos_queue.close()
    # neg_queue.close()
    p.close(); p.terminate(); p.join()
    pos_queue.close(); pos_queue.join_thread()
    neg_queue.close(); neg_queue.join_thread()
    print(f"len encoded vals: {len(encoded_vals)}")
    ohe_df = pd.DataFrame(encoded_vals)
    ##
    # freq_items = apriori(ohe_df, min_support=(10 / len(pred)), use_colnames=True, max_len=3)
    apriori_kwargs = dict(min_support=10/len(pred), use_colnames=True, max_len=apriori_max_len)
    # min_thresholds = [0.6, 0.7]
    # max_lens = [3, 5, 7, 10]
    arm_kwargs = dict(metric="confidence", min_threshold=association_rule_mining_thresh, support_only=False)


    freq_items = apriori(ohe_df, **apriori_kwargs)
    all_rules = association_rules(freq_items, **arm_kwargs)
    print(f"fi -> all_rules: {len(freq_items)} -> {len(all_rules)}")
    # assert len(all_rules) > 0
    if len(all_rules) == 0:
        print(f"NUM RULES WAS 0")
        results = dict(
            rules_discr=None,
            rules_chr=None,
            n_rules_discr=np.nan,
            n_rules_chr=np.nan,
            # discr_train=rules_train,
            # discr_test=rules_test,
            # chr_train=rules_train_chr,
            # chr_test=rules_test_chr,
            # eval_params=default_vals_from_code,
            discr_train=None,
            discr_test=None,
            chr_train=None,
            chr_test=None,
            times_discr=None,
            times_chr=None,
            eval_params=None,
            description='',
            time=time()-start_time
        )
        if result_queue is None:
            return results
        else:
            logging.debug(f"putting empty results in queue")
            result_queue.put(results)
            logging.debug(f".. finished")
        return results

    # save pos/neg rules for further processing
    freq_items_pos = apriori(ohe_df.loc[ohe_df[pos_label] == 1], **apriori_kwargs)
    pos_rules = association_rules(freq_items_pos,**arm_kwargs)
    print(f"fi -> pos_rules: {len(freq_items_pos)} -> {len(pos_rules)}")

    freq_items_neg = apriori(ohe_df.loc[ohe_df[neg_label] == 1], **apriori_kwargs)
    neg_rules = association_rules(freq_items_neg, **arm_kwargs)
    print(f"fi -> neg_rules: {len(freq_items_neg)} -> {len(neg_rules)}")

    positive = all_rules[all_rules['consequents'] == {pos_label}]
    positive = positive[positive['confidence'] == 1]
    positive = positive.sort_values(['confidence', 'support'], ascending=[False, False])
    # assert len(positive) > 0
    seen = set()
    dropped = set()
    indexes_to_drop = []

    positive = positive.reset_index(drop=True)
    print(len(positive))
    for i in positive.index:
        new_rule = positive.loc[[i]]['antecedents'].values[0]

        for seen_rule in seen:
            if seen_rule.issubset(new_rule):  # new_rule.issubset(seen_rule) or seen_rule.issubset(new_rule):
                indexes_to_drop.append(i)
                break
        else:
            seen.add(new_rule)

    positive.drop(positive.index[indexes_to_drop], inplace=True)
    print(len(positive))

    # confidence chr in example code was 1., 0.8, 0.7
    confidence_chr = [0.8, 0.75, 0.7, 0.65, 0.6, 0.5, 0.25, 0.1, 0.05]

    confidence_discr = [0.95, 0.9, 0.85, 0.7, 0.6, 0.5, 0.25, 0.1, 0.05]#, 0.75, 0.6, 0.5, 0.25]

    train_pred = Y_tr  # xgb_clf.predict(X_train)
    # X_test = data_to_pd(X_te)
    # test_pred = Y_te  # xgb_clf.predict(X_test)

    # Discriminative rules
    _rules_models_discr = []
    times_discr = []
    for c_discr in confidence_discr:
        start_time = time()
        discr_rules_c = _cega_discr_rules(deepcopy(all_rules), deepcopy(positive), neg_label, pos_label, confidence=c_discr)
        model_discr = cega_rulemodel.RulesModel(ohe_df, discr_rules_c, train_pred, pos_label, neg_label)
        end_time = time()
        elapsed = end_time - start_time
        times_discr.append(elapsed)
        _rules_models_discr.append(model_discr)

    _rules_models_chr = []
    times_chr = []
    for c_chr in confidence_chr:
        start_time = time()
        chr_rules_c = _cega_chr_rules(deepcopy(pos_rules), deepcopy(neg_rules), pos_label, neg_label, confidence=c_chr)
        model_chr = cega_rulemodel.RulesModel(ohe_df, chr_rules_c, train_pred, pos_label, neg_label)
        end_time = time()
        elapsed = end_time - start_time
        times_chr.append(elapsed)
        _rules_models_chr.append(model_chr)


    # rules_model = cega_rulemodel.RulesModel(ohe_df, discr_rules, train_pred, pos_label, neg_label)
    # Characteristic rules
    # rules_model_chr = cega_rulemodel.RulesModel(ohe_df, chr_rules, train_pred, pos_label, neg_label)

    discr_results = list(zip(confidence_discr, _rules_models_discr))
    chr_results = list(zip(confidence_chr, _rules_models_chr))

    results = dict(
        rules_discr=discr_results,
        rules_chr=chr_results,
        n_rules_discr=[len(r.rules) for r in _rules_models_discr],
        n_rules_chr=[len(r.rules) for r in _rules_models_chr],
        # discr_train=rules_train,
        # discr_test=rules_test,
        # chr_train=rules_train_chr,
        # chr_test=rules_test_chr,
        # eval_params=default_vals_from_code,
        discr_train=None,
        discr_test=None,
        chr_train=None,
        chr_test=None,
        times_discr=times_discr,
        times_chr=times_chr,
        eval_params=None,
        description=''
    )
    # return results
    end_time = time()
    results['time'] = end_time - start_time
    if result_queue is None:
        return results
    else:
        logging.debug(f"putting results in queue")
        result_queue.put(results)
        logging.debug(f".. finished")
    return results


def compute_cega(task: str, expl_method: str, model_set: list[int], gely_threshold: float,
                            significance_threshold: float, k_means_max_bins=40, n_samples_percent=1., seed=42,
                            setcover_reduction=True, item_order=None, gely_sort_items=None, modelclass=None, verbose=False,
                 extend_results=False):    # For one model only
    assert modelclass is not None
    # ------------------------------------------------------------------------
    model_set_str = '['+'-'.join(model_set)+']'
    # if '314029' in model_set_str:
    #     print('sus model, skip')
    #     return

    # fname = f"{task}_{expl_method}_{gely_threshold}_{significance_threshold}_{k_means_max_bins}_{model_set_str}.pkl"
    # fname_old = f"{task}_{expl_method}_{gely_threshold}_{significance_threshold}_{model_set_str}.pkl"
    filetype = '.pkl'
    fname = f"t-{task}_ex-{expl_method}_apriori-{gely_threshold}_st-{significance_threshold}_" + \
             f"s-{model_set_str}_fs-{n_samples_percent}"+filetype

    # ------------------------------------------------------------------------
    import cfire._variables_cfire as _variables
    _, n_classes = lxg.datasets._get_dim_classes(task)
    if verbose:
        print(f'starting eval on {task} dataset')
    data_dir = _variables.get_data_dir(task, modelclass)
    if verbose:
        print(f'loading from {data_dir}')
    expl_rule_dir = _variables.get_expl_rule_dir(task, modelclass)
    # cega_dir = _variables.get_cega_dir(task); _variables.__create_dir(cega_dir)
    _variables.__create_dir(expl_rule_dir)
    if verbose:
        print(f'result dir: {expl_rule_dir}')
    # ------------------------------------------------------------------------

    # if (Path(result_dir, fname).exists() and Path(result_dir, fname).is_file()):
    #
    # sys.exit()

    print(f"starting {fname}")

    model_seeds = model_set

    meta_data = _load_meta_data_by_pid(data_dir)
    data_seed = next(iter(meta_data.values()))['data_seed']
    _data = [d['X'] for d in meta_data.values()]
    _targets = [d['Y'] for d in meta_data.values()]
    eq = []

    for i in range(len(_data)):
        for j in range(i, len(_data)):
            eq.append(_data[i] == _data[j])
    # assert that all metadata in dir uses same testset 'X' <-> we don't have issue with randomzation
    # assert torch.all(torch.stack(eq))
    #
    # # _models = load_idxs_from_multiple_models(data_dir, task, model_seeds, idxs=[-1], return_fns=True)
    # _outputs = [o['output_distribution'] for o in load_idxs_from_multiple_outputs(data_dir, model_seeds, [-1])]
    # _model_predictions = [torch.argmax(o, dim=1).cpu() for o in _outputs]
    #
    # # get model outputs for validation set as test set for rule model later on;
    # # load models
    # if 'hypercube' in task:
    #     _data_val = [torch.Tensor(load_pkl(_variables.get_data_dir(task).__str__()+'/data.pkl')['X_val'])]
    # else:
    #     _data_val = [d['X_val'] for d in meta_data.values()]
    #     eq_val = []
    #     for i in range(len(_data_val)):
    #         for j in range(i, len(_data_val)):
    #             eq_val.append(_data_val[i] == _data_val[j])
    #     # assert that all metadata in dir uses same testset 'X' <-> we don't have issue with randomzation
    #     assert torch.all(torch.stack(eq_val))
    try:
        assert torch.all(torch.stack(eq))
    except TypeError:
        assert np.all(np.stack(eq))

    # _models = load_idxs_from_multiple_models(data_dir, task, model_seeds, idxs=[-1], return_fns=True)
    if modelclass == 'nn':
        _outputs = [o['output_distribution'] for o in load_idxs_from_multiple_outputs(data_dir, model_seeds, [-1])]
        _model_predictions = [torch.argmax(o, dim=1).cpu() for o in _outputs]
    else:
        _outputs = [torch.tensor(o['output_distribution']) for o in load_sklearn_outputs(data_dir, model_seeds)]
        _model_predictions = [torch.argmax(o, dim=1).cpu() for o in _outputs]


    # get model outputs for validation set as test set for rule model later on;
    # load models
    if 'hypercube' in task:
        _data_val = [torch.Tensor(load_pkl(_variables.get_data_dir(task).__str__()+'/data.pkl')['X_val'])]
    else:
        _data_val = [d['X_val'] for d in meta_data.values()]
        eq_val = []
        for i in range(len(_data_val)):
            for j in range(i, len(_data_val)):
                eq_val.append(_data_val[i] == _data_val[j])
        # assert that all metadata in dir uses same testset 'X' <-> we don't have issue with randomzation
    try:
        assert torch.all(torch.stack(eq_val))
    except TypeError:
        assert np.all(np.stack(eq_val))

    if verbose:
        print("loading models")
    X_orig = torch.tensor(_data[0])
    Y_true = torch.tensor(_targets[0])
    X_val = torch.tensor(_data_val[0])

    if modelclass == 'nn':
        _models = load_idxs_from_multiple_models(data_dir, task, model_seeds, [-1])
    else: # sklearn
        _models = load_sklearn_models(data_dir, model_seeds, wrap=True)
    _outputs_val = [_get_outputs(inference_fn=m.forward, data=X_val, model=m, device='cpu') for m in _models]
    _model_predictions_val = [torch.argmax(o, dim=1).cpu() for o in _outputs_val]
    # need to find 'a large set of models that agrees on a large set of points'
    # -> prioritize number of datapoints or number of models?
        # this sounds like set covering?
    _B = [Y_true == mp for mp in _model_predictions]
    _nB = [torch.logical_not(Y_true) == mp for mp in _model_predictions]
    _nBi = [np.argwhere(nb.detach().numpy()).squeeze() for nb in _nB]
    np.array([len(np.setdiff1d(_nBi[i], _nBi[j])) for i in range(len(_nBi)) for j in range(i + 1, len(_nBi))])
    B = torch.vstack(_B).detach().to(torch.int8).numpy()
    data_idxs_all_models_agree = np.argwhere(np.sum(B, 0) == len(B))
    # now filter out all but majority class
    if verbose:
        _bin_count_classes = np.bincount(Y_true[data_idxs_all_models_agree].squeeze(), minlength=max(Y_true))
        print(f"class distribution: {_bin_count_classes}")
    # selected_class = np.argmax(_bin_count_classes)

    # LOAD EXPLANATIONS ---------------------------------------------------------------------------------------#
    expls_all_models = load_idxs_from_multiple_explanations(data_dir, model_seeds, idxs=[-1], explanation_method=expl_method)[0]
    # expls = [e[data_idxs_all_models_agree] for e in expls]

    _model_predictions = [e['Y'] for e in expls_all_models]
    expls_all_models = [e[expl_method] for e in expls_all_models]

    # if expl_method == 'ig':
    #     expls_all_models = [e for e, _ in expls_all_models]  # because ig is (attrs, delta)
    # ([explanations], epoch_batch) -> keep only explanations because we only look at last batch
    expls_all_models = [e.cpu().detach().numpy() for e in expls_all_models]

    if task in nlp_tasks and len(expls_all_models[0].shape) > 2:
        expls_all_models = [np.sum(e, -1) for e in expls_all_models]

    # expls = np.stack(expls)

    # COMPUTE RULES IN EXPLANATION AND DATASPACE FOR EACH CLASS ---------------------------------------------------#

    _all_classes_attr_rules = []
    _all_classes_rules = []
    _expl_coverage = []
    _all_results = []
    # try except out of memory

    for idx_model, _model_num in enumerate(model_seeds):
        filetype = '.pkl'
        cega_fname = f"t-{task}_ex-{expl_method}_apriori-{gely_threshold}_st-{significance_threshold}_" + \
                     f"s-[{_model_num}]" + filetype
        print(f"start {cega_fname}")
        _cega_base_dir = _variables.get_cega_dir(task, modelclass); _variables.__create_dir(_cega_base_dir)
        _cega_pth = Path(_cega_base_dir, cega_fname)

        # if _cega_pth.exists() and _cega_pth.is_file():
        #     print(f"... result found, skipping")
        #     continue

        # print(task, expl_method, gely_threshold, significance_threshold, model_set_str)
        # fname = (f"t-{task}_ex-{expl_method}_gt-{gely_threshold}_st-{significance_threshold}_km-{k_means_max_bins}_"
        #          f"s-[{_model_num}]_fs-{n_samples_percent}_sc-{setcover_reduction}") + filetype
        # if Path(expl_rule_dir, fname).exists() and Path(expl_rule_dir, fname).is_file():
        #     print(f"found {str(Path(expl_rule_dir, fname))}")
        #     return

        # Running CEGA

        _args = dict(X_tr=X_orig,
                    Y_tr=_model_predictions[idx_model],
                    attributions=expls_all_models[idx_model],
                     X_te=X_val, Y_te=_model_predictions_val[idx_model],
                     apriori_max_len=gely_threshold,
                     association_rule_mining_thresh=significance_threshold)
        memory_threshold = 60
        start_time = time()
        cega_results = safe_call(cega_test, _args, memory_threshold*(1024**3))  # 1*(1024**3) == 120GB
        end_time = time()
        # cega_results = cega_test(**_args)
        if cega_results is None:
            # report this time in case cega doesnt finish in safe_call
            elapsed_time = end_time - start_time
            cega_results = dict(
                                rules_discr=None,
                                rules_chr=None,
                                n_rules_discr=None,
                                n_rules_chr=None,
                                discr_train=None,
                                discr_test=None,
                                chr_train=None,
                                chr_test=None,
                                eval_params=None,
                                finished=False,
                                description=f'No rules computed for parameter config; out of memory ({memory_threshold} GB)',
                                time=elapsed_time,
                            )
            print(f'no rules for {cega_fname}')
        else:
            cega_results['finished'] = True
        # print("\n\n")
        # continue
        # import sys;sys.exit()
        dump_pkl(cega_results, _cega_pth)


def _cega_discr_rules(all_rules, positive, neg_label, pos_label, confidence=1.):
    negative = all_rules[all_rules['consequents'] == {neg_label}]
    negative = negative[negative['confidence'] >= confidence ]

    negative = negative.sort_values(['confidence', 'support'], ascending=[False, False])
    # assert len(negative) > 0
    seen = set()
    dropped = set()
    indexes_to_drop = []

    negative = negative.reset_index(drop=True)
    print(len(negative))
    for i in negative.index:
        new_rule = negative.loc[[i]]['antecedents'].values[0]

        for seen_rule in seen:
            if seen_rule.issubset(new_rule):  # new_rule.issubset(seen_rule) or seen_rule.issubset(new_rule):
                indexes_to_drop.append(i)
                break
        else:
            seen.add(new_rule)

    negative.drop(negative.index[indexes_to_drop], inplace=True)
    print(len(negative))
    ##
    positive['num-items'] = positive['antecedents'].map(lambda x: len(x))
    negative['num-items'] = negative['antecedents'].map(lambda x: len(x))
    positive['consequents'] = positive['consequents'].map(lambda x: pos_label)
    negative['consequents'] = negative['consequents'].map(lambda x: neg_label)

    # both = positive.append(negative, ignore_index=True)  # deprecated
    both = pd.concat([positive, negative], ignore_index=True)

    discr_rules = both[
        ['antecedents', 'consequents', 'num-items', 'support', 'confidence', 'antecedent support']].sort_values(
        ['support', 'confidence', 'num-items'], ascending=[False, False, False])

    discr_rules = discr_rules.rename(columns={"antecedents": "itemset", "consequents": "label"})
    return deepcopy(discr_rules)

def _cega_chr_rules(pos_rules, neg_rules, pos_label, neg_label, confidence=0.8):
    # PREPARING RULES FOR CHARACTERISTIC MODE
    rev_positive = pos_rules[pos_rules['antecedents'] == {pos_label}]
    rev_positive = rev_positive[rev_positive['confidence'] >= confidence]
    rev_positive = rev_positive.sort_values(['confidence', 'support'], ascending=[False, False])

    seen = set()
    dropped = set()
    indexes_to_drop = []

    rev_positive = rev_positive.reset_index(drop=True)
    print(len(rev_positive))
    for i in rev_positive.index:
        new_rule = rev_positive.loc[[i]]['consequents'].values[0]

        for seen_rule, indx in seen:
            if seen_rule.issubset(new_rule):
                indexes_to_drop.append(i)
                break
        else:
            seen.add((new_rule, i))

    rev_positive.drop(rev_positive.index[indexes_to_drop], inplace=True)
    print(len(rev_positive))

    rev_negative = neg_rules[neg_rules['antecedents'] == {neg_label}]
    rev_negative = rev_negative[rev_negative['confidence'] >= 0.8]
    rev_negative = rev_negative.sort_values(['confidence', 'support'], ascending=[False, False])

    seen = set()
    dropped = set()
    indexes_to_drop = []

    rev_negative = rev_negative.reset_index(drop=True)
    print(len(rev_negative))
    for i in rev_negative.index:
        new_rule = rev_negative.loc[[i]]['consequents'].values[0]

        for seen_rule, indx in seen:
            if seen_rule.issubset(new_rule):
                indexes_to_drop.append(i)
                break
        else:
            seen.add((new_rule, i))

    rev_negative.drop(rev_negative.index[indexes_to_drop], inplace=True)
    print(len(rev_negative))

    ##

    rev_positive['num-items'] = rev_positive['consequents'].map(lambda x: len(x))
    rev_negative['num-items'] = rev_negative['consequents'].map(lambda x: len(x))
    rev_positive['antecedents'] = rev_positive['antecedents'].map(lambda x: pos_label)
    rev_negative['antecedents'] = rev_negative['antecedents'].map(lambda x: neg_label)

    # rev_both = rev_positive.append(rev_negative, ignore_index=True)
    rev_both = pd.concat([rev_positive, rev_negative], ignore_index=True)

    chr_rules = rev_both[['antecedents', 'consequents', 'num-items', 'support',
                          'confidence', 'consequent support']].sort_values(
        ['support', 'confidence', 'num-items'], ascending=[False, False, False])

    chr_rules = chr_rules.rename(columns={"antecedents": "label", "consequents": "itemset"})
    return deepcopy(chr_rules)

def compute_rules_exp_space_discr(task: str, expl_method: str, model_set: list[int], gely_threshold: float,
                            significance_threshold: float, k_means_max_bins=40, n_samples_percent=1., seed=42,
                            setcover_reduction=True, gely_sort_items=False, modelclass='NN', verbose=False):    # For one model only
    # ------------------------------------------------------------------------
    model_set_str = '['+'-'.join(model_set)+']'
    # if '314029' in model_set_str:
    #     print('sus model, skip')
    #     return

    # fname = f"{task}_{expl_method}_{gely_threshold}_{significance_threshold}_{k_means_max_bins}_{model_set_str}.pkl"
    # fname_old = f"{task}_{expl_method}_{gely_threshold}_{significance_threshold}_{model_set_str}.pkl"
    filetype = '.pkl'
    if type(expl_method) is list:
        expl_method_str = ''.join(sorted(expl_method))
    else:
        expl_method_str = expl_method
    fname = f"t-{task}_ex-{expl_method_str}_gt-{gely_threshold}_st-{significance_threshold}_km-{k_means_max_bins}_" + \
             f"s-{model_set_str}_fs-{n_samples_percent}_sc-{setcover_reduction}_discr"+filetype

    # ------------------------------------------------------------------------
    import cfire._variables_cfire as _variables
    _, n_classes = lxg.datasets._get_dim_classes(task)

    if verbose:
        print(f'starting eval on {task} dataset')
    data_dir = _variables.get_data_dir(task, modelclass=modelclass)
    modelclass = modelclass.lower()
    if verbose:
        print(f'loading from {data_dir}')
    expl_rule_dir = _variables.get_expl_rule_dir(task, modelclass)
    _variables.__create_dir(expl_rule_dir)
    if verbose:
        print(f'result dir: {expl_rule_dir}')
    # ------------------------------------------------------------------------

    # if (Path(result_dir, fname).exists() and Path(result_dir, fname).is_file()):
    #
    # sys.exit()

    print(f"starting {fname}")

    model_seeds = model_set

    meta_data = _load_meta_data_by_pid(data_dir)
    # data_seed = next(iter(meta_data.values()))['data_seed']
    _data = [d['X'] for d in meta_data.values()]
    _targets = [d['Y'] for d in meta_data.values()]

    # easy point to get hacky overview of random baselines, set n_jobs=1 in 01_calc_expl_rules
    # def _rnd_baseline(labels):
    #     h = np.bincount(labels)
    #     majority_class = np.argmax(h)
    #     _l_m = labels == majority_class
    #     return np.mean(_l_m)
    # print("\n\n\n\n")
    # print(task)
    # print(f"{_rnd_baseline(_targets[0]):.3f}")
    # print("\n\n\n\n")
    # return
    eq = []

    for i in range(len(_data)):
        for j in range(i, len(_data)):
            eq.append(_data[i] == _data[j])

    # assert that all metadata in dir uses same testset 'X' <-> we don't have issue with randomzation
    try:
        assert torch.all(torch.stack(eq))
    except TypeError:
        assert np.all(np.stack(eq))

    # _models = load_idxs_from_multiple_models(data_dir, task, model_seeds, idxs=[-1], return_fns=True)
    if modelclass == 'nn':
        _outputs = [o['output_distribution'] for o in load_idxs_from_multiple_outputs(data_dir, model_seeds, [-1])]
        _model_predictions = [torch.argmax(o, dim=1).cpu() for o in _outputs]
    else:
        _outputs = [torch.tensor(o['output_distribution']) for o in load_sklearn_outputs(data_dir, model_seeds)]
        _model_predictions = [torch.argmax(o, dim=1).cpu() for o in _outputs]


    # get model outputs for validation set as test set for rule model later on;
    # load models
    if 'hypercube' in task:
        _data_val = [torch.Tensor(load_pkl(_variables.get_data_dir(task).__str__()+'/data.pkl')['X_val'])]
    else:
        _data_val = [d['X_val'] for d in meta_data.values()]
        eq_val = []
        for i in range(len(_data_val)):
            for j in range(i, len(_data_val)):
                eq_val.append(_data_val[i] == _data_val[j])
        # assert that all metadata in dir uses same testset 'X' <-> we don't have issue with randomzation
    try:
        assert torch.all(torch.stack(eq_val))
    except TypeError:
        assert np.all(np.stack(eq_val))

    if verbose:
        print("loading models")
    if type(_data[0]) is torch.Tensor:
        X_orig = _data[0]
        Y_true = _targets[0]
        X_val = _data_val[0]
    else:
        X_orig = torch.tensor(_data[0])
        Y_true = torch.tensor(_targets[0])
        X_val = torch.tensor(_data_val[0])

    if modelclass == 'nn':
        _models = load_idxs_from_multiple_models(data_dir, task, model_seeds, [-1])
    else: # sklearn
        _models = load_sklearn_models(data_dir, model_seeds, wrap=True)

    _outputs_val = [_get_outputs(inference_fn=m.forward, data=X_val, model=m, device='cpu') for m in _models]
    _model_predictions_val = [torch.argmax(o, dim=1).cpu() for o in _outputs_val]
    _model_predictions = [_get_targets(inference_fn=m.forward, data=X_orig, model=m, device='cpu') for m in _models]


    # need to find 'a large set of models that agrees on a large set of points'
    # -> prioritize number of datapoints or number of models?
        # this sounds like set covering?
    _B = [Y_true == mp for mp in _model_predictions]
    _nB = [torch.logical_not(Y_true) == mp for mp in _model_predictions]
    _nBi = [np.argwhere(nb.detach().numpy()).squeeze() for nb in _nB]
    np.array([len(np.setdiff1d(_nBi[i], _nBi[j])) for i in range(len(_nBi)) for j in range(i + 1, len(_nBi))])
    B = torch.vstack(_B).detach().to(torch.int8).numpy()
    data_idxs_all_models_agree = np.argwhere(np.sum(B, 0) == len(B))
    # now filter out all but majority class
    if verbose:
        _bin_count_classes = np.bincount(Y_true[data_idxs_all_models_agree].squeeze(), minlength=max(Y_true))
        print(f"class distribution: {_bin_count_classes}")
    # selected_class = np.argmax(_bin_count_classes)

    # LOAD EXPLANATIONS ---------------------------------------------------------------------------------------#


    if type(expl_method) is list:
        expls_all_models = []
        for ex in expl_method:
            _ex_expls_all_models = load_idxs_from_multiple_explanations(data_dir, model_seeds, idxs=[-1], explanation_method=ex)[0]
            # expls = [e[data_idxs_all_models_agree] for e in expls]

            # _model_predictions = [e['Y'] for e in _ex_expls_all_models]
            _ex_expls_all_models = [e[ex] for e in _ex_expls_all_models]

            # if 'ig' in expl_method:
            #     expls_all_models = [e for e, _ in expls_all_models]  # because ig is (attrs, delta)
            # ([explanations], epoch_batch) -> keep only explanations because we only look at last batch
            _ex_expls_all_models = [e.cpu().detach().numpy() for e in _ex_expls_all_models]

            if task in nlp_tasks and len(_ex_expls_all_models[0].shape) > 2:
                _ex_expls_all_models = [np.sum(e, -1) for e in _ex_expls_all_models]
            expls_all_models.append(_ex_expls_all_models)


    else:
        expls_all_models = load_idxs_from_multiple_explanations(data_dir, model_seeds, idxs=[-1], explanation_method=expl_method)[0]
        # expls = [e[data_idxs_all_models_agree] for e in expls]

        # _model_predictions = [e['Y'] for e in expls_all_models]
        expls_all_models = [e[expl_method] for e in expls_all_models]

        # if 'ig' in expl_method:
        #     expls_all_models = [e for e, _ in expls_all_models]  # because ig is (attrs, delta)
        # ([explanations], epoch_batch) -> keep only explanations because we only look at last batch
        expls_all_models = [e.cpu().detach().numpy() for e in expls_all_models]

        if task in nlp_tasks and len(expls_all_models[0].shape) > 2:
            expls_all_models = [np.sum(e, -1) for e in expls_all_models]

    # expls = np.stack(expls)

    # COMPUTE RULES IN EXPLANATION AND DATASPACE FOR EACH CLASS ---------------------------------------------------#

    _all_classes_attr_rules = []
    _all_classes_rules = []
    _expl_coverage = []
    _all_results = []
    for idx_model, _model_num in enumerate(model_seeds):
        results_model = {}
        filetype = '.pkl'
        fname = (f"t-{task}_ex-{expl_method_str}_gt-{gely_threshold}_st-{significance_threshold}_km-{k_means_max_bins}_"
                 f"s-[{_model_num}]_fs-{n_samples_percent}_gsi-{gely_sort_items}") + filetype
        # if Path(expl_rule_dir, fname).exists() and Path(expl_rule_dir, fname).is_file():
        #     print(f"found {str(Path(expl_rule_dir, fname))}")
        #     return

        # # Running CEGA
        # cega_fname = (f"cega_t-{task}_ex-{expl_method}-s-[{_model_num}]") + filetype
        #
        # cega_results = cega_test(X_orig, _model_predictions[idx_model],
        #           expls_all_models[idx_model], X_val, _model_predictions_val[idx_model])
        # if cega_results is None:
        #     print(f'no rules for {cega_fname}')
        # else:
        #     dump_pkl(cega_results, Path(_variables.get_cega_dir(task), cega_fname))
        # continue
        # return
        # raise RuntimeError
        start_time = time()
        if type(expl_method) is list:
            expls_preproc = []
            for e in expls_all_models:
                expls_preproc.append(
                    __preprocess_explanations(e[idx_model], significance_threshold)
                )
        else:
            expls_preproc = __preprocess_explanations(expls_all_models[idx_model], significance_threshold)

        # expls_preproc = __preprocess_explanations(expls_all_models[idx_model], filtering='cega')

        classes = set(np.arange(n_classes))
        model = _models[idx_model]
        for selected_class in range(n_classes):

            # model callable need to take data and return True if sample is in target class, False otherwise
            # model_callable = lambda input: (_get_targets(inference_fn=model.forward, data=torch.tensor(input), model=model, device='cpu')
            #                                             == selected_class).detach().numpy()
            model_callable=None

            if verbose:
                print(f"choosing class #{selected_class}")
            if False:  # target ground truth label of data
                # correct label
                _idxs_class = np.argwhere(Y_true == selected_class).squeeze()
                # reshape to get "outer product"-like array of truth values, basically a "if Y_true in classes\selected_class"
                _idxs_others = np.argwhere(
                    np.sum(Y_true.numpy().reshape(-1, 1) == np.array(list(classes - set([selected_class]))).reshape(1, -1), 1)
                ).squeeze()

                # correct label intersect with correctly classified by model
                _filtered_idxs_class = np.intersect1d(np.argwhere(_B[idx_model]), _idxs_class)
                _filtered_idxs_others = np.intersect1d(np.argwhere(_B[idx_model]), _idxs_others)
                __data_filtered_class = X_orig[_filtered_idxs_class].numpy()
                __data_filtered_other = X_orig[_filtered_idxs_others].numpy()
            else:  # use model prediction as target label
                Y_pred = _model_predictions[idx_model]
                # targeting model output not data label, only filter wrt. model and ignore others
                _idxs_class_model = np.argwhere(Y_pred == selected_class).squeeze()
                _idxs_others_model = np.argwhere(
                    np.sum(Y_pred.numpy().reshape(-1, 1) == np.array(list(classes - set([selected_class]))).reshape(1, -1), 1)
                ).squeeze()
                _filtered_idxs_class_model = _idxs_class_model
                _filtered_idxs_others_model = _idxs_others_model.squeeze()

                ## apply filter to data and label vectors
                __data_filtered_class = X_orig[_filtered_idxs_class_model].numpy()
                __data_filtered_other = X_orig[_filtered_idxs_others_model].numpy()
                _filtered_idxs_class = _filtered_idxs_class_model

            # __labels_filtered_class = Y_true[_filtered_idxs_class]

            if n_samples_percent < 1.:
                raise NotImplementedError
                # TODO add subsampling of 'other' classes ?
                # _n_samples_class = len(_filtered_idxs_class)
                # _n_samples_remain = max(2, int(np.ceil(n_samples_percent * _n_samples_class)))
                # with NumpyRandomSeed(seed):
                #     np.random.shuffle(_filtered_idxs_class)
                #     _filtered_idxs_class = _filtered_idxs_class[:_n_samples_remain]

            if type(expl_method) is list:
                expls_filtered = []
                for e in expls_preproc:
                    expls_filtered.append(e[_filtered_idxs_class, :])
            else:
                expls_filtered = expls_preproc[_filtered_idxs_class, :]

            if not (len(__data_filtered_class) > 0 and len(__data_filtered_other) > 0):
                results_model[selected_class] = None
                continue

            results_model[selected_class] = __compute_rules_discr(expls_filtered, __data_filtered_class,
                                                                            __data_filtered_other, gely_threshold,
                                                                            k_means_max_bins, setcover_reduction,
                                                                            gely_sort_items, verbose, model_callable)
            # print(f"class {selected_class}")
            # _rules_in_data_space = map_attr_sets_to_data_space(_data_for_rules, [rules[1:] for rules in converted_fcis])

        # for i, rr in results_model.items():
        #     a = [r.accuracy for r in rr[1] if r.accuracy is not np.nan]
        #     print(i, len(a), np.round(np.mean(a), 4), np.round(np.std(a), 4))
        end_time = time()
        elapsed_time = end_time - start_time
        results_model['time'] = elapsed_time
        results_model['args'] = dict(task=task, expl_method=expl_method_str, gely_threshold=gely_threshold,
                                     model_seed=_model_num, significance_threshold=significance_threshold,
                                     k_means_max_bins=k_means_max_bins, n_samples_percent=n_samples_percent,
                                     setcover_reduction=setcover_reduction, gely_sort_items=gely_sort_items,
                                     modelclass=modelclass
                     )
        results_model['model_pred_val'] = _model_predictions_val[idx_model]
        results_model['model_pred_te'] = _model_predictions[idx_model]
        # quick_dirty_dnf_test(results_model, X_te=X_orig, Y_te=_model_predictions[idx_model],
        #                      X_val=X_val, Y_val=_model_predictions_val[idx_model])
        # return

        for c in range(n_classes):
            _rmc = results_model[c]
            if _rmc == []: # no rules found for class c - doesn't happen often but it can happen ..
                continue
            nodes = _rmc[1]
            fnodes = nodes[0].get_frequent_children()
            [fn.dnf.assert_no_infty() for fn in fnodes]

        dump_pkl(results_model, Path(expl_rule_dir, fname))
        _all_results.append(results_model)

    return _all_results



__key_to_idx = {k: v for (k,v) in
           zip(['task', 'expl_method', 'gely_threshold', 'significance_threshold',
                'kmeans_max_bins', 'n_models', 'setcover_reduction'], range(6))}

def split_results_by_key(results, key_param='n_models', key_result='train;accuracy'):
    # naming: f"{task}_{expl_method}_{gely_threshold}_{significance_threshold}_{n_models}
    _key_tr_te = None
    if ';' in key_result:
        _key_tr_te, key_result = key_result.split(';')

    idx = __key_to_idx[key_param]
    vals = [r[0].split('_')[idx] for r in results]
    keys = np.unique(vals)
    results_by_key = {k: [] for k in keys}
    for i, v in enumerate(vals):
        if _key_tr_te is not None:
            results_by_key[v].append(results[i][1][_key_tr_te][key_result])
        else:
            results_by_key[v].append(results[i][1][key_result])
    return results_by_key



def make_parser():
    """
    Defines options for the script and default values
    :return: parser object
    """
    def int_list(input: str) -> list[int]:
        # parse string of list "[1, 2, 3]" -> [1, 2, 3]; [1,] is an invalid input
        input = input.replace('[', '').replace(']', '').replace(' ', '')
        input = input.split(',')
        if len(input) == 0:
            return []
        else:
            return [int(i) for i in input]

    def str_list(input:str) -> list[str]:
        input = input.replace('[', '').replace(']', '').replace(' ', '')
        input = input.split(',')
        return input

    parser = argparse.ArgumentParser()
    parser.add_argument('--datarule', default=False, type=bool)
    parser.add_argument('--tasks', default=None, type=str_list)
    parser.add_argument('--resultsprefix', default=None, type=str_list)
    parser.add_argument('--sorting', default=None, type=str)

    return parser



if __name__ == '__main__':
    pass
    # '''
    # This script is an absolute mess :')
    # goal:
    # 1. given a variable set of tasks and params
    #     load indicated expl method
    #     compute expl rules according to params for all tasks
    # 2. #TODO filter expl rules
    # 3. given expl rules + supporting set, compute rules in dataspace
    # 4. evaluate rules
    # '''
    #
    # np.random.seed(42)
    #
    # import _variables_cfire as _variables
    # args = make_parser().parse_args()
    #
    # args.datarule = False
    # # if we want to compute rules for the data
    # if args.datarule:
    #     expl_rules_to_data_rules(); print(_variables.counter);
    #     sys.exit()
    # # else: we process the explanations to expl_rules to prepare for computing rules on data
    #
    # if args.tasks is not None:
    #     tasks = args.tasks
    # else:
    #     classif_root = Path('./data/cfire/')
    #     tasks = []
    #     for d in classif_root.iterdir():
    #         if d.is_dir() and 'classification' in str(d):
    #             tasks.append(str(d).split('/')[-1])
    #
    #
    #
    # tasks = _variables.make_classification_configs
    # tasks = ['beans']
    # print(tasks)
    #
    # model_sets = {task: lxg.util.get_all_model_seeds_sorted(Path(classif_root, task)) for task in tasks}
    # sizes_model_sets = [1, 2, 3, 5, 7, 10]
    # _max_runs_model_set_size = 15
    #
    # # model_combinations = {}
    # # for task in tasks:
    # #     model_combs = []
    # #     seeds = model_sets[task]
    # #     for i, size in enumerate(sizes_model_sets):
    # #         _combs_size = [c for c in combinations(seeds, size)]
    # #         np.random.shuffle(_combs_size)
    # #         model_combs.extend(_combs_size[:_max_runs_model_set_size])
    # #     model_combinations[task] = model_combs
    #
    #
    # '''
    # expl_methods_names = _variables.explanation_abbreviations
    # expl_methods_names = ['sg', 'ig']
    # # expl_methods_names = ['ig', 'ks']
    # # explanation_relevance_thresholds = [0.01, 0.05, 0.1, 0.2]
    # explanation_relevance_thresholds = [0.01, 'topk0.5', 'topk0.25', 'topk0.1', ]# 'topkabs0.1']
    # # explanation_relevance_thresholds = ['topk-0.1', 'topkabs-0.1']
    # # gely_threshold = [0.5, 0.7, 0.8, 0.9]
    # gely_threshold = [0.25, 0.4, 0.8]
    # # n_samples_percent = [0.1, 0.3, 0.5, 0.8]# 1.]
    # n_samples_percent = [1.]  # [0.5, 1.]
    # kmeans_max_bins = [1, 2, 40]
    # setcover_reduction = [False, True]
    # '''
    #
    # # PARAMS CALC RULES EXPL SPACE
    # expl_methods_names = _variables.explanation_abbreviations
    # expl_methods_names = ['sg', 'ig']
    # explanation_relevance_thresholds = [0.01, 'topk0.5', 'topk0.25']
    # gely_threshold = [0.05]
    # n_samples_percent = [1.]
    # kmeans_max_bins = [1]
    # setcover_reduction = [False]
    #
    # # PARAMS MAP RULES DATA SPACE
    #
    # # PARAMS EVAL RULES
    #
    #
    # arg_sets = []
    # # def compute_eval_rules(task, expl_method, model_set, gely_threshold, significance_threshold, kmeans, fraction):
    # for task in tasks:
    #     arg_sets.extend([
    #         argset for argset in product([task], expl_methods_names,
    #                                          [model_sets[task]],
    #                                          gely_threshold,
    #                                          explanation_relevance_thresholds,
    #                                          kmeans_max_bins,
    #                                          n_samples_percent,
    #                                          setcover_reduction
    #                                       )
    #     ])
    # ##### np.random.shuffle(arg_sets)
    # if args.sorting == 'reverse':
    #     print("\nreverse\n")
    #     arg_sets = arg_sets[::-1]
    # n_args = len(arg_sets)
    # # arg_sets = arg_sets[int(n_args/3):]
    # print(f"number of argument-combinations: {len(arg_sets)}")
    # # argnames = (task, expl_method, model_set, gely_threshold, significance_threshold):
    # # arg1 = dict(task='beans', expl_method='ig', model_set=model_set['beans'][:2], gely_threshold=0.8, significance_threshold=0.01, verbose=True)
    # # _, elapsed = time_func(compute_eval_rules, arg1)
    # # sys.exit()
    # # arg_sets = [arg_sets[0]]
    # # print(arg_sets)
    # # a = arg_sets[0]
    # # compute_rules_exp_space(task=a[0], expl_method=a[1], model_set=a[2], gely_threshold=a[3],
    # #                         significance_threshold=a[4], k_means_max_bins=a[5], n_samples_percent=a[6],
    # #                         setcover_reduction=a[7], verbose=False); sys.exit()
    #
    # # from multiprocessing import Pool
    #
    # # with Pool(14) as pool:
    # #     pool.starmap(compute_eval_rules,\
    # #             [(a[0], a[1], a[2], a[3], a[4], a[5], a[6]) for a in arg_sets],
    # #              chunksize=1)
    # # sys.exit()
    #
    # '''1. compute rules in expl space'''
    # with parallel_backend(backend='loky', n_jobs=10):  # 14
    #     # Parallel()(delayed(rename)(task=a[0], expl_method=a[1], model_set=a[2], gely_threshold=a[3],
    #     #                                         significance_threshold=a[4], k_means_max_bins=a[5]) for a in arg_sets)
    #     Parallel(verbose=10, batch_size=2)(delayed(compute_rules_exp_space)(
    #                                             task=a[0], expl_method=a[1], model_set=a[2], gely_threshold=a[3],
    #                                             significance_threshold=a[4], k_means_max_bins=a[5], n_samples_percent=a[6],
    #                                             setcover_reduction=a[7], verbose=False)
    #                                        for a in arg_sets)# 03 normal, 01[::-1]
    # '''2. TODO filter rules of expl space'''
    # '''3. map expl rules to data rules'''
    # # expl_rules_to_data_rules()
    # # print(_variables.counter)
    # '''4. evaluate data rules'''
    # sys.exit()
    #
    # # base_dir = './'
    # # results_prefix = 'rashomon'