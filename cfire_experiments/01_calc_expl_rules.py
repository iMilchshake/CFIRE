# standard packages
import os, argparse, sys
default_path = '../'
os.chdir(default_path)
print(f"chdir > curdir: {os.getcwd()}")

print(f"PYTHONPATH: {sys.path}")
# sys.exit()
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
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, silhouette_score

from scipy.spatial.distance import pdist

# local files
import lxg
from lxg.datasets import nlp_tasks, NumpyRandomSeed
from lxg.util import load_losses, load_accuracies, load_idxs_from_multiple_explanations, _load_meta_data_by_pid, \
    load_idxs_from_multiple_outputs,\
    dump_pkl, load_pkl


import cfire._variables_cfire as _variables
from cfire.gely import gely, gely_discriminatory
from cfire.util import greedy_set_cover, item_to_model_bin, __preprocess_explanations
from cfire.cfire import compute_rules_exp_space, compute_rules_exp_space_discr, compute_cega, compute_cfi_all_classes

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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
    _filtered_by_sign = [_e.T[i, mask] for i, mask in enumerate((_e > 0.).T)]
    if not positive:
        _filtered_by_sign = [-1 * f for f in _filtered_by_sign]
    return __kmeans_binning(_filtered_by_sign, n_bins, n_jobs)


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




def time_func(f, args):
    s1 = time()
    results = f(**args)
    elapsed = time() - s1
    return results, elapsed


def __compute_rules_exp_space(explanations, data, gely_threshold, k_means_max_bins=2, verbose=True) -> dict:
        binning_args = dict(n_jobs=4, n_bins=k_means_max_bins)
        bin_borders = compute_bins(explanations, binning_args=binning_args)
        expls_binarized = binarize_explanations(explanations, bin_borders)


        gely_args = {'B': expls_binarized,
                     'use_binary': False,
                     'threshold': gely_threshold}

        _gely_threshold = gely_threshold
        if verbose:
            print(f"compute fcis support = {gely_args['threshold']}")
        _fcis2 = []
        n_tries = 0
        max_tries = np.inf# 2 if gely_threshold >= 0.5 else 3
        while len(_fcis2) == 0 and n_tries < max_tries:
            _fcis2, time2 = time_func(gely, gely_args)
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


def _compare_tree_attrs(dt_paths, attributions):

    k = int(np.ceil(attributions.shape[1]*0.25))
    thresh = 0.1
    scores = []
    for (attr, path) in zip(attributions, dt_paths):
        _absmax = np.max(np.abs(attr))
        if _absmax != 0:
            attr = attr / _absmax
        _idxs_clearing_thresh = np.argwhere(attr >= thresh).squeeze()
        _topk = np.argsort(-attr)[:k]#[:len(path)]
        attr_idxs = np.intersect1d(_topk, _idxs_clearing_thresh)
        if len(attr_idxs) == 0:
            scores.append(0)
            continue
        # ignores if dims appear twice in path
        fa = len(np.intersect1d(attr_idxs, path))/len(path)#len(attr_idxs)#len(path)
        scores.append(fa)

    return np.mean(scores), np.std(scores)

def eval_expls_pgi(task, model_ids, attributions, data=None, k=0, n_noise_samples=100,
                   std=0.5, mask=None, parallelize=True):
    # compute pgi auc for all methods in expl_methods and models in model_ids for given task
    # adding attributions += ['random_attribution_baseline']
    # return average auc pgi and return
    # dict[k=expl_method] = _mean_and_variance(models, expl_method, attribution)

    if data is None:
        raise NotImplementedError

    if type(data) == np.ndarray:
        data = torch.from_numpy(data)

    models = [
        lxg.util.load_idxs_from_model
            (_variables.get_data_dir(task), task, model_id, [-1], return_fns=True)[0]
        for model_id in model_ids
    ]

    from lxg.evaluation import PGI

    if not parallelize:
        result_pgis = []
        for (_model, inference_fn, preprocess_fn), attrs in zip(models, attributions):
            _pgi_auc_model = PGI(_model, data, targets=None, attributions=attrs, k=k,
                                 inference_fn=inference_fn, min_init_prob=0.7,
                                 n_noise_samples=n_noise_samples, std=std, mask=mask  # if mask is not None,
                                 )

            result_pgis.append(_pgi_auc_model)
    else:
        with parallel_backend(backend='loky', n_jobs=min(12, len(models))):  # 14
            # Parallel()(delayed(rename)(task=a[0], expl_method=a[1], model_set=a[2], gely_threshold=a[3],
            #                                         significance_threshold=a[4], k_means_max_bins=a[5]) for a in arg_sets)
            result_pgis = Parallel(verbose=10, batch_size=1)(delayed(PGI)(
                            model=_model, data=data, targets=None, attributions=attrs, k=k,
                            inference_fn=inference_fn, n_noise_samples=n_noise_samples,
                            std=std, mask=mask, min_init_prob=0.7)
                                       for (_model, inference_fn, preprocess_fn), attrs in zip(models, attributions))

    # pgi_stats = []
    # for pgi in result_pgis:
    #     pgi_stats.append(
    #         (np.mean(pgi), np.std(pgi))
    #     )

    # coarse mean and std across all samples and models
    # return (np.mean([p[0] for p in pgi_stats]), np.mean([p[1] for p in pgi_stats]))
    return result_pgis

def compare_attributions_w_deicsionpaths(task: str, expl_method: str, model_set: list[int]):
    data_dir = _variables.get_data_dir(task)

    model_seeds = sorted(model_set)

    meta_data = _load_meta_data_by_pid(data_dir)
    md = next(iter(meta_data.values()))
    data_seed = md['data_seed']
    batch_sizes = md['batch_sizes']



    _data = [d['X'] for d in meta_data.values()]
    _targets = [d['Y'] for d in meta_data.values()]
    eq = []

    # load data with seed

    from lxg.datasets import _get_dataset_callable, _loader_to_numpy
    if str(task).startswith('classification'):
        _dataset = lxg.util.load_pkl(str(data_dir)+"/data.pkl")
        (X_tr, Y_tr), (X_te, Y_te), (X_val, Y_val) = _dataset['train'], _dataset['test'], _dataset['validation']
        X_te, Y_te = np.array(X_te, dtype=np.float32), np.array(Y_te, dtype=np.int32)

    else:
        dataset_callable = _get_dataset_callable(task)
        tr, te, _n_dim, _n_classes = dataset_callable(random_state=data_seed, batch_sizes=batch_sizes, as_torch=True)
        (X_tr, Y_tr) = _loader_to_numpy(tr)
        (X_te, Y_te) = next(iter(te))
        X_te, Y_te = X_te.detach().numpy(), Y_te.detach().numpy()

    from sklearn.tree import DecisionTreeClassifier as DT

    dt = DT(max_depth=5, random_state=42) # max_leaf_nodes might be interesting
    dt.fit(X_tr, Y_tr)
    dt_test_acc = dt.score(X_te, Y_te)
    print(f'Test accuracy tree {dt_test_acc}')

    def dt_node_idx_to_dim(dt):
        t = dt.tree_
        left = t.children_left
        right = t.children_right
        n_nodes = t.capacity
        _lr = ['l' if i in left else 'r' for i in range(n_nodes)]
        assert n_nodes == len(left)
        dims = [np.nan if left[i] == -1 or right[i] == -1 else t.feature[i] for i in range(n_nodes)]
        return np.array(dims)

    def dt_paths_to_dims(paths, dims):
        paths_by_dim = []
        for row in paths:
            idxs = row.indices
            # dims can appear twice in path
            paths_by_dim.append(dims[idxs][:-1])
        return paths_by_dim

    # def dt_path_to_unique_dim(paths, dims):
    #     paths_by_dim = []
    #     for row in paths:
    #         idxs = row.indices
    #         paths_by_dim.append(np.unique(dims[idxs][:-1]))
    #     return paths_by_dim

    dpaths = dt.decision_path(X_te)
    dt_node_dims = dt_node_idx_to_dim(dt)
    _path_by_dims = dt_paths_to_dims(dpaths, dt_node_dims)


    for i in range(len(_data)):
        for j in range(i, len(_data)):
            eq.append(_data[i] == _data[j])
    # assert that all metadata in dir uses same testset 'X' <-> we don't have issue with randomzation
    assert torch.all(torch.stack(eq))

    # _models = load_idxs_from_multiple_models(data_dir, task, model_seeds, idxs=[-1], return_fns=True)
    _outputs = [o['output_distribution'] for o in load_idxs_from_multiple_outputs(data_dir, model_seeds, [-1])]
    _model_predictions = [torch.argmax(o, dim=1).cpu() for o in _outputs]



    X_orig = _data[0]
    Y_true = _targets[0]
    # need to find 'a large set of models that agrees on a large set of points'
    # -> prioritize number of datapoints or number of models?
        # this sounds like set covering?
    _B = [Y_true == mp for mp in _model_predictions]
    _tree_pred = np.argwhere([Y_te == dt.predict(X_te)]).squeeze()
    B = torch.vstack(_B).detach().to(torch.int8).numpy()
    data_idxs_all_models_agree = np.argwhere(np.sum(B, 0) == len(B)).squeeze()
    tree_ann_agree = np.intersect1d(_tree_pred, data_idxs_all_models_agree)


    X_te, Y_te = X_te[tree_ann_agree], Y_te[tree_ann_agree]
    # LOAD EXPLANATIONS ---------------------------------------------------------------------------------------#
    # sorted by model seeds
    n_noise_samples = 1000
    std = 0.5
    _pgi_k = X_te.shape[1]#int(np.ceil(X_te.shape[1] * 0.1)) # * 0.25))
    n_reps = X_te.shape[1] * 20
    mask = 0.
    print(f"n_noise_samples: {n_noise_samples}\n"
          f"_pgi_k: {_pgi_k}\n"
          f"n_reps: {n_reps}\n")
    if expl_method == 'rnd':
        expls_all_models = [
                [np.random.uniform(low=-1.0, high=1.0, size=X_te.shape)  # actual distr does not matter for randomized ranks
                    for _ in range(len(model_set))
                ]
            for _ in range(n_reps)
        ]
        _rnd_pgis = []

        # with parallel_backend(backend='loky', n_jobs=12):  # 14
        #     # Parallel()(delayed(rename)(task=a[0], expl_method=a[1], model_set=a[2], gely_threshold=a[3],
        #     #                                         significance_threshold=a[4], k_means_max_bins=a[5]) for a in arg_sets)
        #     _rnd_pgis = Parallel(verbose=10, batch_size=2)(delayed(eval_expls_pgi)(
        #         task=task, model_ids=model_set, attributions=_exp_rep, data=X_te, k=int(np.ceil(X_te.shape[1] * 0.25)),
        #         n_noise_samples=100, std=1.)
        #                                        for _exp_rep in expls_all_models)

        from time import time
        with NumpyRandomSeed(int(time())):
            for _exp_rep in tqdm(expls_all_models):
                _pgi_rep = eval_expls_pgi(task, model_set, _exp_rep, k=_pgi_k,
                               data=X_te, n_noise_samples=n_noise_samples, std=std, mask=mask)
                _rnd_pgis.append(_pgi_rep)
        #TODO needs stacking, calc mean and std, hope std small else we need to increase n_reps
        _stacked = [[] for _ in range(len(_rnd_pgis[0]))]
        for r in _rnd_pgis:
            for i in range(len(r)):
                _stacked[i].append(r[i])
        pgis = [np.mean(np.stack(s), 0) for s in _stacked]


        # r = [[] for _ in range(len(model_set))]
        # for e in expls_all_models:
        #     for i in range(len(e)):
        #         r[i].append(_compare_tree_attrs(_path_by_dims, e[i]))
        r = []
        for e in expls_all_models:
            for rep in e:
                r.append(_compare_tree_attrs(_path_by_dims, rep))
        _means = [rr[0] for rr in r]
        _stds = [rr[1] for rr in r]
        mean = np.mean(_means)
        mean_stds = np.sqrt(np.sum(np.power(_stds, 2))/len(_stds))
        scores = (mean, mean_stds)

        return scores, pgis

    else:
        expls_all_models = load_idxs_from_multiple_explanations(data_dir, model_seeds, idxs=[-1], explanation_method=expl_method)[0]
        # expls = [e[data_idxs_all_models_agree] for e in expls]
        if expl_method == 'ig':
            expls_all_models = [e for e, _ in expls_all_models]  # because ig is (attrs, delta)
        # ([explanations], epoch_batch) -> keep only explanations because we only look at last batch
        expls_all_models = [e.cpu().detach().numpy() for e in expls_all_models]

        filtered_expls = [e[tree_ann_agree] for e in expls_all_models]
        pgis = eval_expls_pgi(task, model_set, filtered_expls, k=_pgi_k,
                       data=X_te, n_noise_samples=n_noise_samples, std=std, mask=mask)
    # compute pgis for each explanation and compare against random baseline;
    # remove explanations (samples) if explanation does not beat random.

    scores = []
    m, s = [], []


    for e in filtered_expls:
        _m, _s = _compare_tree_attrs(_path_by_dims, e)
        m.append(_m)
        s.append(_s)
    # scores = (np.mean(m), np.mean(s))
    scores = (m, s)
    print(f"{np.mean(m):.4f}, {np.mean(s):.4f}")
    print()
    return scores, pgis


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
    parser.add_argument('--cega', default=False, type=bool)
    parser.add_argument('--modelclass', default='xgb', type=str)
    parser.add_argument('--debug', default=False, type=bool)

    return parser

if __name__ == '__main__':

    '''
    This script is an absolute mess :')
    goal:
    1. given a variable set of tasks and params
        load indicated expl method
        compute expl rules according to params for all tasks
    2. #TODO filter expl rules
    3. given expl rules + supporting set, compute rules in dataspace
    4. evaluate rules
    '''

    np.random.seed(42)


    args = make_parser().parse_args()

    # else: we process the explanations to expl_rules to prepare for computing rules on data


    # tasks = [
    #     # TODO: which of tasks lead to iteration over 0-dim array?
    #     #      "hypercube-10000-
    #     #         'spf',3-0-0-0-0.01-0.01", # problematic? ValueError, listclosed on empty I (min(I)) error
    #     'spf',
    #     'spambase',
    #     'heloc',
    #     'btsc',
    #     'breastw',
    #     "hypercube-10000-3-3-0-0-0.01-0.01", # fails assertion check in dt_to_dnf regarding prediction equivalence
    #     "breastcancer",
    #     "beans",  # found the culprit!
    #     "ionosphere",
    #     "hypercube-10000-3-0-0-0-0.01-0.01",
    #      "hypercube-10000-4-4-0-0-0.01-0.01", # same as above  -------> REMOVED PRUNING FOR NOW
    #      "hypercube-10000-5-0-0-0-0.01-0.01", # ! no problem (yet)
    #      "hypercube-10000-5-5-0-0-0.01-0.01", # same assertion error in dt_to_dnf
    #          "hypercube-10000-4-0-0-0-0.01-0.01", # doesn't exist?
    #          ]
    if args.cega:
        print("CALC RULES FOR CEGA")
        tasks = ['ionosphere',
                'breastcancer',
                'btsc',
                'spf',
                'breastw',
                'heloc',
                'spambase']
    else:
        tasks = _variables.tasks

    modelclass = args.modelclass.lower()
    print(modelclass)

    # if args.tasks is not None or 'tasks' in locals():
    #     tasks = args.tasks
    # else:
    #     classif_root = Path('./data/cfire/')
    #     tasks = []
    #     for d in classif_root.iterdir():
    #         if d.is_dir() and 'classification' in str(d):
    #             tasks.append(str(d).split('/')[-1])
    # model_sets = {task:
    #                   lxg.util.get_all_model_seeds_sorted(Path(classif_root, task))
    #               for task in tasks if "classification" in task
    #               }

    # tasks = ['breastcancer', 'ionosphere', 'beans']
    print(tasks)

    model_id_acc = {task: lxg.util.get_top_k_models(_variables.get_data_dir(task, modelclass=modelclass), k=50)
                  for task in tasks}
    for task, acc in model_id_acc.items():
        print(f"task {task}\t\t -> mean acc {np.mean([a[1] for a in acc]):.2f}")
        # print(f"{np.mean([a[1] for a in acc]):.2f} ~ {np.std([a[1] for a in acc]):.2f}")
    # import sys;sys.exit()
    model_sets = {task: [x[0] for x in id_acc]
                  for task, id_acc in model_id_acc.items()}
    print(tasks)
    # sizes_model_sets = [1, 2, 3, 5, 7, 10]
    # _max_runs_model_set_size = 10

    '''
    expl_methods_names = _variables.explanation_abbreviations
    expl_methods_names = ['sg', 'ig']
    # expl_methods_names = ['ig', 'ks']
    # explanation_relevance_thresholds = [0.01, 0.05, 0.1, 0.2]
    explanation_relevance_thresholds = [0.01, 'topk0.5', 'topk0.25', 'topk0.1', ]# 'topkabs0.1']
    # explanation_relevance_thresholds = ['topk-0.1', 'topkabs-0.1']
    # gely_threshold = [0.5, 0.7, 0.8, 0.9]
    gely_threshold = [0.25, 0.4, 0.8]
    # n_samples_percent = [0.1, 0.3, 0.5, 0.8]# 1.]
    n_samples_percent = [1.]  # [0.5, 1.]
    kmeans_max_bins = [1, 2, 40]
    setcover_reduction = [False, True]
    '''
    
    # PARAMS CALC RULES EXPL SPACE
    # expl_methods_names = _variables.explanation_abbreviations
    # expl_methods_names = ['rnd', 'grdpgi', 'ks', 'li', 'vg', 'sg', 'ig']
    # SMOOTHGRAD CAUSES PROBLEMS IF ALL ATTR SCORES ARE NEGATIVE (why are they all negative)

    expl_methods_names = [ 'li', 'ks', 'ig',#'ds',#'li',# 'ks',#, 'ksub', 'ig', 'igub', 'li',
                            # ['ksub', 'li'], ['ksub', 'ks'], ['ksub', 'ig'],
                           # ['ksub', 'igub', 'liub'], ['ks', 'ig', 'li', ]
                          ]#'ks']#, 'sg', 'ig']  # ['grdpgi', 'ks', 'sg', 'ig', 'li']# 'ig']
    # if modelclass == 'nn':
    #     expl_methods_names.append('ig')
    # expl_methods_names = ['grdpgi', 'ks', 'li', 'vg', 'sg', 'ig']
    # explanation_relevance_thresholds = ['topk0.5', 'topk0.25', 0.01]
    explanation_relevance_thresholds = [0.01]#'topk0.5', 'topk0.25', 0.01]#, 'topk0.25']
    gely_threshold = [0.01]
    # print(gely_threshold)
    # import sys; sys.exit()
    n_samples_percent = [1.]
    kmeans_max_bins = [1]
    setcover_reduction = [False]
    gely_sort_items = [True]
    arg_sets = []

    if args.cega:
        gely_threshold = [3]#[3, 6]  # [3, 6, 10]? [3, 10, 15]  # None]  #nhyperparam aprioi 3=cege default, None=apriori default (leads to OOM)
        explanation_relevance_thresholds = [0.04]#[0.25, 0.5, 0.7]  # [0.25, 0.5, 0.7]
        gely_sort_items = [False]
        expl_methods_names = [e for e in expl_methods_names if type(e) is str]


    # def compute_eval_rules(task, expl_method, model_set, gely_threshold, significance_threshold, kmeans, fraction):
    for task in tasks:
        arg_sets.extend([
            argset for argset in product([task], expl_methods_names,
                                         [model_sets[task]],
                                         gely_threshold,
                                         explanation_relevance_thresholds,
                                         kmeans_max_bins,
                                         n_samples_percent,
                                         setcover_reduction,
                                         gely_sort_items, [modelclass]
                                         )
        ])
    if args.cega:
        # sort args
        arg_sets = sorted(arg_sets, key=lambda x: (np.argwhere(np.array(tasks)==x[0]).item(), x[3], -x[4]))

    ##### np.random.shuffle(arg_sets)_modelclasses
    if args.sorting == 'reverse':
        print("\nreverse\n")
        arg_sets = arg_sets[::-1]
    n_args = len(arg_sets)
    # arg_sets = arg_sets[int(n_args/3):]
    print(f"number of argument-combinations: {len(arg_sets)}")
    '''1. compute rules in expl space'''
    args.eike = False

    if args.cega:
        _callable = compute_cega
    # elif args.glocalx:
    #     _callable = compute_glocalx
    elif args.eike == True:
        _callable = compute_cfi_all_classes
    else:
        _callable = compute_rules_exp_space_discr
    # assert args.cega

    if args.debug:
        n_jobs = 1
    elif args.cega:
        n_jobs = 1
    else:
        n_jobs = min(12, max(1, int(n_args/2)))
    print(arg_sets)
    with parallel_backend(backend='loky', n_jobs=n_jobs):  # 14
        # Parallel()(delayed(rename)(task=a[0], expl_method=a[1], model_set=a[2], gely_threshold=a[3],
        #                                         significance_threshold=a[4], k_means_max_bins=a[5]) for a in arg_sets)
        Parallel(verbose=10, batch_size=2)(delayed(_callable)(
            task=a[0], expl_method=a[1], model_set=a[2], gely_threshold=a[3],
            significance_threshold=a[4], k_means_max_bins=a[5], n_samples_percent=a[6],
            setcover_reduction=a[7], gely_sort_items=a[8],  modelclass=a[9], verbose=False)
                                           for a in arg_sets)

# train_loader, test_loader, validation, input_size, n_classes = \
#                 _get_dataset_callable(task)(random_state=args.data_seed, batch_sizes=batch_sizes, split_validation=True)