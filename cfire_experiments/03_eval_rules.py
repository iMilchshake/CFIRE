from collections import Counter

import sklearn.metrics

default_path = '../'
import os
os.chdir(default_path)


import numpy as np
from pathlib import Path
from tqdm import tqdm
from joblib import parallel_backend, Parallel, delayed
from itertools import product

import torch
import numpy as np
import pandas as pd

import argparse

from lxg.datasets import __info_dim_classes
import lxg.models
from lxg.models import DNFClassifier
import cfire._variables_cfire as _variables
from cfire.util import (greedy_set_cover, __preprocess_explanations, DNFConsistencyReturnCode)

import cfire.nodeselection as nodeselection

from lxg.util import (load_pkl, load_accuracies, _load_meta_data_by_pid, dump_pkl,
                      load_sklearn_models, load_idxs_from_model, _get_outputs, _get_targets)

from cfire.gely import ItemsetNode

from copy import deepcopy

from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.model_selection import GridSearchCV



def compose_dnfs(gely_results: list[dict[ItemsetNode, dict]], scoring_weights, complexity_parameter=-1):

    # gely results contains one tuple of (supprt idxs, nodes) per class
    # and also [compute time (float), arguments (dict), model predictions for validation and test]
    n_classes = len(gely_results[0]) - 4
    orig_args = []
    _nodes = []
    for node in gely_results:
        orig_args.append(node['args'])
        _nodes.append([node[i] for i in range(n_classes)])

    # _split_data = []
    # for y in range(n_classes):
    #     _split_data.append(split_by_class(X_tr, Y_tr, y))

    def merge_single_class_dnfs_multiclass_dnf(dnfs):
        rules = [dnf.rules[0] for dnf in dnfs]  # 1 or 0?
        return DNFClassifier(rules, 'accuracy')

    DNFS = []
    _args = []

    for _iteration, (node, args) in enumerate(zip(_nodes, orig_args), 1):  # list of gely results
        print(f"{_iteration}/{len(_nodes)}: {args['model_seed']} - {args['expl_method']}")
        _DNFs_node = [] # will contain list[class1[strat 1, strat 2, strat 3], class2[strat 1 ..]]
        for c in range(n_classes):
            # nodes_c = tuple(support_idxs for frequent nodes, root+all children)
            nodes_c = node[c]
            if nodes_c is None or len(nodes_c) == 0:
                print(f"no nodes for {args['expl_method']}")
                l_scoring_weights = 0 if scoring_weights is None else len(scoring_weights)
                _DNFs_node.append([DNFClassifier(rules=[[[(-1,(np.nan, np.nan))]]]) for _ in range(l_scoring_weights+1)])
                continue
            supp, _all_nodes = nodes_c
            if len(_all_nodes) == 0:
                print(f"no nodes for {args['expl_method']}")
                l_scoring_weights = 0 if scoring_weights is None else len(scoring_weights)
                _DNFs_node.append([DNFClassifier(rules=[[[(-1,(np.nan, np.nan))]]]) for _ in range(l_scoring_weights+1)])
                continue
            root = _all_nodes[0]; assert root.parent is None
            freq_nodes = root.get_frequent_children()
            _f, _s = [], []
            for f, s in zip(freq_nodes, supp):
                if len(s) > 0:
                    _f.append(f); _s.append(s)
            freq_nodes, supp = _f, _s

            gc_dnf = nodeselection._comp_greedy_cover(supp, freq_nodes)
            score_dnfs = []
            if scoring_weights is not None and len(scoring_weights > 0):
                score_dnfs = [nodeselection._comp_score_cover(supp, freq_nodes,
                                                              acc_weight=l1, cx_weight=l2, cs_weight=l3)
                              for (l1, l2, l3) in scoring_weights
                              ]

            _DNFs_node.append([gc_dnf]+score_dnfs)

        # merge _DNFs_node st arg_dnf_strat1, arg_dnf_strat2
        merged = [merge_single_class_dnfs_multiclass_dnf(d) for d in list(zip(*_DNFs_node))]
        DNFS.extend(merged)
        # extend args list accordingly with stratnames
        _cover_args = deepcopy(args)
        _cover_args.update({'composition_strategy': 'cover',
                            'acc_weight': np.nan,
                            'cx_weight': np.nan,
                            'cs_weight': np.nan})
        _args.append(_cover_args)
        if scoring_weights is not None and len(scoring_weights) > 0:
            for (l1, l2, l3) in scoring_weights:
                __a = deepcopy(args)
                __a.update({'composition_strategy': 'score_cover',
                                'acc_weight': l1,
                                'cx_weight': l2,
                                'cs_weight': l3})
                _args.append(__a)

    # each element in DNFs has 3 models, corresponding to the composition three strats
    # each arg in _args is thus applicable to three models, maybe zip it to return it like that to make pandasDF also groupable by strat?

    return DNFS, _args

def load_dnfclassifiers(task, modelclass):
    pth = _variables.get_dnfclassifier_dir(task, modelclass)
    fname = f'{task}_dnfrules.pkl'
    return load_pkl(Path(pth, fname))


def load_all_explanations(task, seeds, methods, modelclass, ignore_missing_expls=True):
    from cfire.cfire import __preprocess_explanations
    pth = _variables.get_data_dir(task, modelclass)
    attrs_tr, attrs_te = {}, {}
    for seed in seeds:
        attrs_tr[seed] = {}
        attrs_te[seed] = {}
        for method in methods:
            expls = lxg.util.load_explanations(pth, seed, method, modelclass)[0][0]
            _atr = expls[method].numpy()
            attrs_tr[seed][method] = _atr  # __preprocess_explanations(a.numpy(), 0.01)
            attrs_tr[seed]['Y'] = expls['Y'].numpy()
            try:
                _ate = expls['val'][method].numpy()
                attrs_te[seed][method] = _ate  # __preprocess_explanations(_ate, 0.01)
                attrs_te[seed]['Y'] = expls['val']['Y'].numpy()
            except KeyError as e:
                if ignore_missing_expls: # validation expls
                    # print(f"warning: {e}")
                    # print(f"continue without")
                    continue
                else:
                    raise KeyError(e)


    return attrs_tr, attrs_te

def set_metrics(Y: set, P: set):
    true_positives = len(Y.intersection(P))
    false_positives = len(P - Y)
    false_negatives = len(Y - P)

    precision = true_positives / len(P) if len(P) > 0 else 0
    recall = true_positives / len(Y) if len(Y) > 0 else 0
    accuracy = true_positives / (true_positives + false_positives + false_negatives) \
        if (true_positives + false_positives + false_negatives) > 0 else 1


    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1_score,
    }

def comp_agreement_metrics(E_true, Y_true, E_pred, Y_pred) -> dict:
    # precision, recall, fpr, f1
    results = []

    for et, yt, ep, yp in zip(E_true, Y_true, E_pred, Y_pred):
        if yp == -1:
            continue
        if len(et) == 0:
            continue
        if yt != yp:
            continue

        et, ep = set(et), set(ep)
        r = set_metrics(et, ep)
        r = {'completeness_'+k: v for k, v in r.items()}
        results.append(r)

    results = pd.DataFrame(results).mean()
    return results.to_dict()

def get_comp_dummy_results():
    d =  [
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]
    r = {'completeness_' + k: np.nan for k in d}
    return r

def compt_agreement_topk(E_true, Y_true, E_pred, Y_pred, k) -> dict:
    # precision, recall, fpr, f1
    results = []

    for et, yt, ep, yp in zip(E_true, Y_true, E_pred, Y_pred):
        if yp == -1:
            continue
        if len(et) == 0:
            continue
        if yt != yp:
            continue
        et = et[:k]
        ep = ep[:k]
        et, ep = set(et), set(ep)
        r = set_metrics(et, ep)
        r = {f'completeness_top{k}_'+key: value for key, value in r.items()}
        results.append(r)

    results = pd.DataFrame(results).mean()
    return results.to_dict()

def dims_from_expl(E):
    # get all dims from explanation that have more than 0 importance
    # return dims in order from most to least important (ie sorted by score)
    _sorting = np.argsort(E, 1)[:, ::-1]
    results = []
    for e, s in zip(E, _sorting):
        r = [_si for _si in s if e[_si] > 0.01]
        results.append(r)
    return results

def dims_from_binarised_expl(E):
    results = []
    for r in E:
        results.append(list(np.argwhere(r).squeeze(-1)))
    return results

def sorted_unique_by_frequency(arr):
    unique, counts = np.unique(arr, return_counts=True)
    sorted_indices = np.lexsort((-counts, unique))
    sorted_unique_elements = unique[sorted_indices]
    return sorted_unique_elements

def dims_from_rules(rules: list):
    if rules is None:
        return np.empty(0)
    dims = []
    for r in rules:
        for t in r:
            dims.append(t[0])
    return sorted_unique_by_frequency(dims)

def comp_accuracy_metrics(Y_true, Y_pred):
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

    accuracy_include_rejected = np.mean(Y_true == Y_pred)

    rejection = np.mean(Y_pred == -1)
    coverage = 1. - rejection

    recall = recall_score(Y_true, Y_pred, average='micro', zero_division=0)

    _predicted_idxs = np.argwhere(Y_pred != -1).squeeze(-1)

    _Yt = Y_true[_predicted_idxs]
    _Yp = Y_pred[_predicted_idxs]

    if len(_Yp) == 0:
        accuracy_exclude_rejected = 0
        precision = 0
        f1 = 0
    else:
        accuracy_exclude_rejected = np.mean(_Yt == _Yp)
        precision = precision_score(_Yt, _Yp, average='micro', zero_division=0)

        f1 = 0
        if precision > 0 or recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)

    # _dms_roc_auc_tr = roc_auc_score(_Yt, _Yp, average='macro')

    return {
        'accuracy': accuracy_include_rejected,
        'accuracy_no_rejected': accuracy_exclude_rejected,
        'coverage': coverage,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

def compute_eval(Y_true, Y_pred, Attr_true, Attr_pred):
    r = {}
    _accuracies = comp_accuracy_metrics(Y_true, Y_pred)
    if Attr_true is None:
        _completeness = get_comp_dummy_results()
    else:
        _completeness = comp_agreement_metrics(Attr_true, Y_true, Attr_pred, Y_pred)
    # _comp_top5 = compt_agreement_topk(Attr_true, Y_true, Attr_pred, Y_pred, k=5)
    # _comp_top3 = compt_agreement_topk(Attr_true, Y_true, Attr_pred, Y_pred, k=3)
    r.update(_accuracies)
    r.update(_completeness)
    # r.update(_comp_top5)
    # r.update(_comp_top3)
    return r

def load_cega_models(task, modelclass, load_full=False):
    from CEGA.rules_model import RulesModel
    def get_args_cega_results(fname: str):
        # results_format = 't-STR_ex-STR_apriori-IN_-st-FLOAT_s-[INT].pkl'
        task = fname.split('_')[0].split('-')[1:]
        if len(task) > 1:
            task = '-'.join(task)
        else:
            task = task[0]
        expl = fname.split('_')[1].split('-')[1]
        apriori_max_len = int(fname.split('_')[2].split('-')[1])
        asrm_threshold = float(fname.split('_')[3].split('-')[1])
        seed = fname.split('[')[1].split(']')[0]
        return dict(task=task, expl_method=expl, model_seed=seed,
                    apriori_max_len=apriori_max_len, asrm_threshold=asrm_threshold)
    cega_resuts_dir = _variables.get_cega_dir(task, modelclass)
    _result_str_names = ['recall_micro', 'acc', 'recall', 'precision', 'f1', 'roc_auc']
    if not os.path.exists(cega_resuts_dir):
        print(f"no CEGA results found for task {task}")
        return pd.DataFrame(columns=_result_str_names+['n_rules']+['rule_type'])
    files = os.listdir(cega_resuts_dir)
    results = [load_pkl(os.path.join(cega_resuts_dir, f)) for f in files]
    if len(results) == 0:
        return pd.DataFrame(columns=_result_str_names+['n_rules']+['rule_type'])
    args = [get_args_cega_results(f) for f in files]
    _chr, _discr = [], []
    _loaded = []

    if load_full:
        min_rules_thresh = 0
    else:
        min_rules_thresh = 1

    for a, r in zip(args, results):

        if r['chr_test'] is not None:
            c = {k:v for v, k in zip(r['chr_test'][1:], _result_str_names)}
        else:
            c = {k:None for k in _result_str_names}

        if r['n_rules_chr'] != np.nan:
            try:
                n_rules_chr_thresh = min_rules_thresh if np.any(np.array(r['n_rules_chr']) >= min_rules_thresh) else max(r['n_rules_chr'])
            except TypeError as e:
                print(f"\n\n\n\n\n {task} {modelclass} {e} \n\n\n\n\n\n")
                assert False
        else:
            n_rules_chr_thresh = -1
        for (conf, rchr), nr in zip(r['rules_chr'], r['n_rules_chr']):
            if rchr is not None:
                if nr <= n_rules_chr_thresh:
                    continue
                _d = dict(rulemodel=rchr, rule_type='chr', confidence=conf,
                          n_rules=len(rchr.rules), n_unique_rules=nr,
                          time=r['time'])
                _d.update(a)
                # _d.update(c)

                _loaded.append(_d)


        if r['discr_test'] is not None:
            c = {k:v for v, k in zip(r['discr_test'][1:], _result_str_names)}
        else:
            c = {k:None for k in _result_str_names}

        if r['n_rules_discr'] != np.nan:
            n_rules_discr_thresh = min_rules_thresh if np.any(np.array(r['n_rules_discr']) >= min_rules_thresh) else max(r['n_rules_discr'])
        else:
            n_rules_discr_thresh = -1
        for (conf, rdiscr), nr in zip(r['rules_discr'], r['n_rules_discr']):
            if rdiscr is not None:
                if nr <= n_rules_discr_thresh:
                    continue
                _d = dict(rulemodel=rdiscr, rule_type='discr',
                          confidence=conf, n_unique_rules=nr,
                          time=r['time'])
                _d.update(a)
                # _d.update(c)
                _loaded.append(_d)

    # unq_itemsets = [l['rulemodel'].rules.itemset.unique() for l in _loaded]
    # mask = []
    # for i in unq_itemsets:
    #     for _iset in i:
    #         if len(_iset) > 1:
    #             mask.append(True)
    #             break
    #         mask.append(False)
    # __filtered = [l for i, l in zip(mask, _loaded) if i]
    # loaded_df = pd.DataFrame(__filtered)
    loaded_df = pd.DataFrame(_loaded)
    return loaded_df

def cega_rule_get_dim(r):
    # r.itemset = "frozenset({'-0.243<V28<=1.328'})"
    # -> split('<') -> ["frozenset({'-0.243", 'V28', "=1.328'})"]
    # print(f"r: {r.itemset}")
    _l = list(iter(r.itemset))
    # print(f"l = {_l}")
    dims = []
    for i in r.itemset:
        d = i.split('<')[1]
        d = int(d[1:]) - 1  # cega dims are offset by 1
        dims.append(d)
    return dims

def cega_dims_from_rules(rules):
    results = []
    for r in rules:
        results.extend(cega_rule_get_dim(r))
    results = sorted_unique_by_frequency(results)
    return results


# def _row_exists(r_dict, df):
#     relevant_keys = ['expl_method', 'model_seed', 'apriori_max_len', 'asrm_threshold', 'confidence']

__LOADED_MODELS = {}
import torch


def flip_signs_with_probability(matrix, p=0.5):
    random_matrix = torch.rand_like(matrix, dtype=torch.float)
    flip_mask = random_matrix < p
    matrix_float = matrix.float()
    flipped_matrix = torch.where(flip_mask, -matrix_float, matrix_float)

    return flipped_matrix.to(matrix.dtype)

def __comp_pgi_perturb(samples, relevant_dims, model, inference_fn, n_perturbations, std=1):
    # PGI, see Openxai paper and cited sources;
    # og definition perturbs sample 'with small amount of gaussian noise'

    _mean = 0
    _min, _max = torch.min(samples), torch.max(samples)
    _span = _max - _min
    _sampling_range = np.abs(0.1 * _span)
    # print(f"_sampling_range={_sampling_range}")

    _out_orig = _get_outputs(inference_fn, samples, model, 'cpu')
    _pred = torch.argmax(_out_orig, dim=1)
    _pgi_scores = []
    for i, (s, rd) in enumerate(zip(samples, relevant_dims)):
        if len(rd) == 0:
            _pgi_scores.append(np.nan)
            continue
        # _perturbations = torch.normal(_mean, std, size=(n_perturbations, len(rd)))

        # uniform sampling [0.01, _sampling_range+0.01]
        _perturbations = torch.rand((n_perturbations, len(rd))) * _sampling_range + 0.05*_span
        # add sign
        _perturbations = flip_signs_with_probability(_perturbations, 0.5)

        _s = s.repeat((n_perturbations, 1))
        for j, d in enumerate(rd):
            _s[:, int(d)] += _perturbations[:, j]
        _s_out = _get_outputs(inference_fn, _s, model, 'cpu')
        score_orig = _out_orig[i, _pred[i]].item()
        diff = (score_orig - _s_out[:, _pred[i]])/score_orig
        score = torch.mean(diff).item()
        _pgi_scores.append(score)

    return np.array(_pgi_scores)


def __comp_pgi_mask(samples, relevant_dims, model, inference_fn,
                    masking_val=0.):
    # PGI, see Openxai paper and cited sources;
    # og definition perturbs sample 'with small amount of gaussian noise'

    if type(masking_val) in [float, int]:
        masking_val = np.full_like(samples, masking_val)
    elif type(masking_val) == torch.Tensor and len(masking_val) == 1:
        masking_val = masking_val.expand(len(samples), -1)

    assert masking_val.shape == samples.shape

    _out_orig = _get_outputs(inference_fn, samples, model, 'cpu')
    _pred = torch.argmax(_out_orig, dim=1).unsqueeze(1)
    _out_orig = torch.gather(_out_orig, 1, _pred).squeeze()
    _pgi_scores = []

    for i, rd in enumerate(relevant_dims):
        if len(rd) > 0:
            _rd = np.array(rd, dtype=int)
            samples[i, _rd] = masking_val[i, _rd]
    _out_masked = _get_outputs(inference_fn, samples, model, 'cpu')
    _out_masked = torch.gather(_out_masked, 1, _pred).squeeze()

    diff = ((_out_orig - _out_masked)/_out_orig).detach().numpy()
    for i, rd in enumerate(relevant_dims):
        if len(rd) == 0:
            diff[i] = np.nan

    return diff

from lxg.attribution import cmaes_baseline
def zero_uniform_prediction_baseline(inference_fn, n_classes, n_dims):
    cmaes_solution, _ = cmaes_baseline(inference_fn, initial_baseline=0.,
                                       n_dims=n_dims, n_classes=n_classes)
    torch_cmaes_baseline = torch.tensor(cmaes_solution, dtype=torch.float)
    _bl = torch_cmaes_baseline.unsqueeze(0)
    return _bl

def comp_pgi_scores(samples, relevant_dimensions, task, modelclass, modelseed,
                    perturbation_mode='sampling', n_samples=1000, baselines=['zeroup'], data_seed=None):
    assert perturbation_mode in ['sampling', 'masked']

    if type(samples) is torch.Tensor:
        samples = samples.clone().detach()
    else:
        samples = torch.tensor(samples).clone().detach()

    # retrieve model, load if not in __LOADED_MODELS yet
    try:
        model = __LOADED_MODELS[modelclass][modelseed]
    except KeyError:
        if modelclass not in __LOADED_MODELS.keys():
            __LOADED_MODELS[modelclass] = {}
        _data_dir = _variables.get_data_dir(task, modelclass)
        if modelclass == 'nn':
            __LOADED_MODELS[modelclass][modelseed] = load_idxs_from_model(_data_dir, task,
                                                                          modelseed, idxs=[-1], return_fns=False)[0]
        else:
            # sklearn
            __LOADED_MODELS[modelclass][modelseed] = load_sklearn_models(_data_dir, [modelseed], True)[0]
        model = __LOADED_MODELS[modelclass][modelseed]
    inference_fn = model.predict_batch_softmax if modelclass == 'nn' else model.forward

    if perturbation_mode == 'masked':
        pgis = np.zeros(samples.shape[0])
        dp = _variables.get_data_dir(task, modelclass)
        bl = load_pkl(Path(dp, 'baselines', f'baselines_{modelseed}_{data_seed}.pkl'))
        for b in baselines:
            _bl = bl[b][0].expand(len(samples), -1)
            pgi = __comp_pgi_mask(samples, relevant_dimensions, model, inference_fn, masking_val=_bl)
            pgis += pgi
        pgis /= len(baselines)
        return pgis

    elif perturbation_mode == 'sampling':
        return __comp_pgi_perturb(samples, relevant_dimensions, model, inference_fn, n_samples)
    else:
        raise NotImplementedError

def comp_suff_scores(samples, relevant_dimensions, task, modelclass, modelseed,
                    perturbation_mode='sampling', n_samples=1000, baselines=['zeroup'], data_seed=None):
    _dims = set(np.arange(samples.shape[1]))
    irrelevant_dims = []
    for r in relevant_dimensions:
        irrelevant_dims.append(list(_dims - set(r)))
    return comp_pgi_scores(samples, irrelevant_dims, task, modelclass, modelseed,
                    perturbation_mode, n_samples, baselines, data_seed)


def row_exists(df, value_dict, relevant_keys):
    # relevant_keys = ['expl_method', 'model_seed', 'apriori_max_len', 'asrm_threshold', 'confidence']
    r_dict = {k: value_dict[k] for k in relevant_keys}
    query = ' and '.join(f'{k} == {repr(v)}' for k, v in r_dict.items())
    return df.query(query).shape[0] > 0

def eval_cega(task, modelclass='nn', expl_methods=['ks'], extend_results=False, test_only=False):
    print(f"CEGA on {task} - {modelclass}")

    if len(__LOADED_MODELS) > 0:
        __LOADED_MODELS.clear()

    data_dir = _variables.get_data_dir(task, modelclass=modelclass)
    _pth = Path(data_dir, 'explanations')
    cega_results = load_cega_models(task, modelclass)


    if cega_results is None:
        raise ValueError(f"no CEGA results found for task {task} and modelclass {modelclass}")


    result_pth = Path(f'./plots/cfire/{modelclass}/{task}/')
    result_pth.mkdir(exist_ok=True, parents=True)
    if extend_results:
        try:
            CEGA_te_eval = load_pkl(str(Path(result_pth, 'CEGA_te_eval.pkl')))
            # CEGA_tr_eval = load_pkl(str(Path(result_pth, 'CEGA_tr_eval.pkl')))
        except FileNotFoundError:
            extend_results = False
    # df_DNFS: pd.DataFrame = load_dnfclassifiers(task)

    meta_data = _load_meta_data_by_pid(data_dir)
    md = meta_data[next(iter(meta_data))]

    X_train = md['X']  # test data from NN
    X_test = md['X_val']  # val data from NN becomes test data for rules

    if type(X_train) is not torch.Tensor:
        X_train = torch.from_numpy(X_train)
    if type(X_test) is not torch.Tensor:
        X_test = torch.from_numpy(X_test)

    col_names = [f'V{i}' for i in range(1, X_train.shape[1] + 1)]  # see cega_test called from 01_calc_expl_rules
    X_train_df = pd.DataFrame(X_train, columns=col_names)
    X_test_df = pd.DataFrame(X_test, columns=col_names)
    # contains attributions and labels
    if expl_methods is None:
        expl_methods = cega_results['expl_method'].unique()
    if 'model_seed' not in cega_results.columns:
        raise ValueError(f"sth wrong with cega on {task} - {modelclass}")

    # attrs_tr, attrs_te = load_all_explanations(task,
    #                                            cega_results['model_seed'].unique(),
    #                                            expl_methods,
    #                                            modelclass)
    _eval_test = []
    _eval_train = []
    _cega_eval_collected = []
    for _count, row in tqdm(cega_results.iterrows(), total=len(cega_results), desc=f'{task} - {modelclass}'):
        print(f"{task}: {_count}/{len(cega_results)}")
        # model_seed, expl_method, labels


        _rd = row.to_dict()

        # if extend_results:
        #     if row_exists(value_dict=_rd,
        #                   df=CEGA_te_eval,
        #                   relevant_keys=['expl_method', 'model_seed', 'apriori_max_len', 'asrm_threshold', 'confidence']):
        #         continue

        dnf = row.rulemodel
        if dnf is None:
            # DataFrame will fill in non-existing keys here with NaN
            tr_eval = _rd
            _eval_train.append(tr_eval)
            te_eval = _rd
            _eval_test.append(te_eval)
            continue

        if dnf.intervals_dict is None:
            dnf.compute_intervals_dict(X_train_df)
            row.dnf = dnf
            _rd = row.to_dict()

        _expl_method = row.expl_method
        seed = row.model_seed

        _att_tr_dims = None
        _att_te_dims = None
        try:
            model = __LOADED_MODELS[modelclass][seed]
        except KeyError:
            if modelclass not in __LOADED_MODELS.keys():
                __LOADED_MODELS[modelclass] = {}
            _data_dir = _variables.get_data_dir(task, modelclass)
            if modelclass == 'nn':
                __LOADED_MODELS[modelclass][seed] = load_idxs_from_model(_data_dir, task,
                                                                              seed, idxs=[-1], return_fns=False)[0]
            else:
                # sklearn
                __LOADED_MODELS[modelclass][seed] = load_sklearn_models(_data_dir, [seed], True)[0]
            model = __LOADED_MODELS[modelclass][seed]
        inference_fn = model.predict_batch_softmax if modelclass == 'nn' else model.forward

        ytr = _get_targets (inference_fn, X_train, model, 'cpu').detach().numpy()
        yte = _get_targets(inference_fn, X_test, model, 'cpu').detach().numpy()
        # if not test_only:
        #     _attr = attrs_tr[seed][_expl_method]
        #     _att_tr_dims = dims_from_expl(_attr)
        #
        # _atte = attrs_te[seed][_expl_method]
        # _att_te_dims = dims_from_expl(_atte)
        #
        # ytr = attrs_tr[seed]['Y']
        # yte = attrs_te[seed]['Y']

        _random_baselines_tr = _rnd_baseline(ytr)
        _random_baselines_te = _rnd_baseline(yte)

        # _cega_eval_collected.append(dnf.eval_rules(X_train_df, ytr))
        # if not test_only:
        _y_tr_pred, _tr_rules = dnf.predict_explain(X_train_df)#, explain=True)
        _tr_dims = [cega_dims_from_rules(r) for r in _tr_rules]
        _y_te_pred, _te_rules = dnf.predict_explain(X_test_df)
        _te_dims = [cega_dims_from_rules(r) for r in _te_rules]

        _tr_avg_n, _tr_avg_len = _compute_expl_len_statistics(list(zip(_y_tr_pred, _tr_rules)))
        _te_avg_n, _te_avg_len = _compute_expl_len_statistics(list(zip(_y_te_pred, _te_rules)))

        dnf.X = None  # delete data from model so we dont save it
        # if not test_only:
        tr_eval = compute_eval(ytr, np.array(_y_tr_pred), _att_tr_dims, _tr_dims)
        tr_eval.update(deepcopy(_rd))
        _tr_eval_copy = deepcopy(tr_eval)
        tr_eval['rnd_tr'] = _random_baselines_tr
        tr_eval['empirical_avg_n_rules'] = _tr_avg_n
        tr_eval['empirical_avg_len'] = _tr_avg_len
        _eval_train.append(tr_eval)

        te_eval = compute_eval(yte, np.array(_y_te_pred), _att_te_dims, _te_dims)
        for k, v in _tr_eval_copy.items():
            te_eval[f'tr_{k}'] = v
        te_eval.update(deepcopy(_rd))
        te_eval['rnd_te'] = _random_baselines_te
        te_eval['empirical_avg_n_rules'] = _te_avg_n
        te_eval['empirical_avg_len'] = _te_avg_len


        norm_pgi = lambda pgi, dims: [score / len(d) if score != np.nan else np.nan for score, d in
                                 zip(pgi, dims)]
        pbm = 'masked'
        pgi_masked_cega_te = comp_pgi_scores(X_test, _te_dims, task, modelclass, seed, perturbation_mode=pbm,
                                           data_seed=md['data_seed'])
        pgi_masked_dnf_te_normalized = norm_pgi(pgi_masked_cega_te, _te_dims)
        te_eval['pgi_masked'] = pgi_masked_cega_te
        te_eval['pgi_masked_normalized'] = pgi_masked_dnf_te_normalized
        te_eval['sufficiency_masked'] = comp_suff_scores(X_test, _te_dims, task, modelclass, seed, perturbation_mode=pbm,
                                           data_seed=md['data_seed'])
        te_eval['pgi_sampling'] = None
        te_eval['pgi_sampling_normalized'] = None


        _eval_test.append(te_eval)

    _cega_eval_collected_df = pd.DataFrame(_cega_eval_collected)
    _eval_test_df = pd.DataFrame(_eval_test)
    if not test_only:
        _eval_train_df = pd.DataFrame(_eval_train)

    if extend_results:
        _eval_test_df = pd.concat([_eval_test_df, CEGA_te_eval], axis=0)
        # if not test_only:
        #     _eval_train_df = pd.concat([_eval_train_df, CEGA_tr_eval], axis=0)

    dump_pkl(_eval_test_df, str(Path(result_pth, 'CEGA_te_eval.pkl')))
    if not test_only:
        dump_pkl(_eval_train_df, str(Path(result_pth, 'CEGA_tr_eval.pkl')))


def tree_predictions_paths(tree, X):
    y = tree.predict(X)

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

    dpaths = tree.decision_path(X)
    dt_node_dims = dt_node_idx_to_dim(tree)
    paths = dt_paths_to_dims(dpaths, dt_node_dims)

    return y, paths

def tree_count_unique_internal_nodes(dt):
    # thanks claude :3
    # Get the tree structure
    tree = dt.tree_

    # Function to create a unique representation of a node
    def node_representation(node_id):
        return (tree.feature[node_id], tree.threshold[node_id])

    # Get all non-leaf nodes
    internal_nodes = np.where(tree.feature != -2)[0]

    # Get unique node representations
    unique_nodes = set(node_representation(node) for node in internal_nodes)

    return len(unique_nodes)

def _compute_expl_len_statistics(rule_explanations):
    # given explanations (applicable rules) for multiple samples, compute:
    # 1) avg. how many rules were applicable per sample
    # 2) avg. long was each rule
    avg_n_rules = np.mean([len(r[1]) for r in rule_explanations if r[0]!=-1])
    avg_len_terms = np.mean([np.mean([len(r) for r in rr[1]]) for rr in rule_explanations if rr[0]!=-1])
    return avg_n_rules, avg_len_terms


def _eval_ensembling_expl_methods(descriptions, row, expl_methods=['ks', 'li', 'ig']):

    def get_ensemble_description(y, e):

        _attrs, descs = [], []
        for a, d in e:
            _attrs.append(a)
            descs.append(d[y])
        _statuses = []
        for d in descs:
            if len(d) == 0 or d == [[]]:
                _statuses.append('uncovered')
            elif len(d) == 1:
                _statuses.append('single')
            else:
                _dim_sets = []
                for clause in d:
                    _dim_sets.append(set([t[0] for t in clause]))
                longest_set = max(_dim_sets, key=len)
                _multi = True
                for s in _dim_sets:
                    if not s <= longest_set:
                        _multi = False
                _statuses.append('multi' if _multi else 'inconsistent')
        _counts = Counter(_statuses)

        if _counts['uncovered'] == len(e):
            return [], [], DNFConsistencyReturnCode.NOT_COVERED
        if _counts['single'] + _counts['multi'] > 1:
            idxs = list(filter(lambda x: _statuses[x] in ['single', 'multi'], range(len(_statuses))))
            # pick with best attribution precision
            _precisions = []
            for i in idxs:
                itemset = []  # Extract sets of dimensions that were used in each conjunction
                for conjunction in descs[i]:
                    itemset.extend([t[0] for t in conjunction])
                itemset = set(itemset)  # remove duplicates
                common = _attrs[i].intersection(itemset)
                try:
                    precision = len(common) / len(itemset)
                except ZeroDivisionError:
                    print("stop")
                _precisions.append(precision)
            _mi = np.argmax(_precisions)
            max_idx = idxs[_mi]
            s = DNFConsistencyReturnCode.CORRECT_MULTI if _statuses[max_idx] == 'multi' else DNFConsistencyReturnCode.CORRECT_SINGLE
            return (_attrs[max_idx], descs[max_idx], s)
        if _counts['single'] == 1:
            idx = _statuses.index('single')
            return _attrs[idx], descs[idx], DNFConsistencyReturnCode.CORRECT_SINGLE
        if _counts['multi'] == 1:
            idx = _statuses.index('multi')
            return _attrs[idx], descs[idx], DNFConsistencyReturnCode.CORRECT_MULTI
        if _counts['inconsistent'] != 0:
            idxs = list(filter(lambda x: _statuses[x] in ['inconsistent'], range(len(_statuses))))
            # pick with best attribution precision
            _precisions = []
            for i in idxs:
                itemset = []  # Extract sets of dimensions that were used in each conjunction
                for conjunction in descs[i]:
                    itemset.extend([t[0] for t in conjunction])
                itemset = set(itemset)  # remove duplicates
                common = _attrs[i].intersection(itemset)
                try:
                    precision = len(common) / len(itemset)
                except ZeroDivisionError:
                    print("stop")
                _precisions.append(precision)
            _mi = np.argmax(_precisions)
            max_idx = idxs[_mi]
            return (_attrs[max_idx], descs[max_idx], DNFConsistencyReturnCode.CORRECT_INCONSISTENT)
        raise RuntimeError # we should never arrive here

    def _agreement_prec(attr_set, rules):
        if len(rules) == 0:
            return np.nan
        items = []
        for term in rules:
            items.extend([c[0] for c in term])
        items = set(items)
        common = attr_set.intersection(items)
        return len(common) / len(items)


    def _add_keys(_d, source_keys):
        return {k: _d.get(k, 0) for k in source_keys}

    __local_expl_eval_keys = [DNFConsistencyReturnCode.NOT_COVERED.name,
                              DNFConsistencyReturnCode.CORRECT_MULTI.name,
                              DNFConsistencyReturnCode.CORRECT_INCONSISTENT.name,
                              DNFConsistencyReturnCode.CORRECT_SINGLE.name, ]

    # get all results for same model seed with different explanations
    row['expl_method'] = 'ensemble'
    # test
    _te = descriptions['te']
    yte = _te['y']
    eval_te = []
    # compute ensemble description output
    for y, e in zip(yte, list(zip(*(_te[ex] for ex in expl_methods)))):
        # y=int, e=tuple(set, list)
        # attribution set, description, status['uncovered', 'single', 'multi']
        _ensemble_description = get_ensemble_description(y, e)
        eval_te.append(_ensemble_description)
    # compute performance
    te_codes = [e[2] for e in eval_te]
    te_local_eval = {k.name: v for k,v in Counter(te_codes).items()}
    te_local_eval = _add_keys(te_local_eval, __local_expl_eval_keys)
    te_prec_agreement = np.nanmean([_agreement_prec(e[0], e[1]) for e in eval_te])
    row['te_local_eval'] = te_local_eval
    row['te_prec_agreement'] = te_prec_agreement

    _tr = descriptions['tr']
    ytr = _tr['y']
    eval_tr = []
    # compute ensemble description output
    for y, e in zip(ytr, list(zip(*(_tr[ex] for ex in expl_methods)))):
        # y=int, e=tuple(set, list)
        # attribution set, description, status['uncovered', 'single', 'multi']
        _ensemble_description = get_ensemble_description(y, e)
        eval_tr.append(_ensemble_description)
    # compute performance
    tr_codes = [e[2] for e in eval_tr]
    tr_local_eval = {k.name: v for k,v in Counter(tr_codes).items()}
    tr_local_eval = _add_keys(tr_local_eval, __local_expl_eval_keys)
    tr_prec_agreement = np.nanmean([_agreement_prec(e[0], e[1]) for e in eval_tr])
    row['tr_local_eval'] = tr_local_eval
    row['tr_prec_agreement'] = tr_prec_agreement
    row['te_consistency'] = None
    row['tr_consistency'] = None

    return row


def eval_dnfclassifiers_consistency(task, modelclass):
    # related to discussion on Jan 28th, we want to find out how often a sample is classified by a box
    # that "should" not cover it according to the local explanation.
    # if this doesn't happen we are good. if it does we have more to write about and formalize. we suspect it
    # might happen regularly.

    print(f"eval dnf {task}-{modelclass}")

    if len(__LOADED_MODELS) > 0:
        __LOADED_MODELS.clear()

    data_dir = _variables.get_data_dir(task, modelclass)
    df_DNFS: pd.DataFrame = load_dnfclassifiers(task, modelclass)
    _chosen_models_accuracy = pd.read_csv('./plots/cfire/nn/chosen_models.csv')
    if "Unnamed: 0" in _chosen_models_accuracy.columns:
        _chosen_models_accuracy = _chosen_models_accuracy.drop(columns=["Unnamed: 0"]).reset_index(drop=True)
        _chosen_models_accuracy = _chosen_models_accuracy.astype(str)

    df_DNFS['chosen'] = (df_DNFS.set_index(['model_seed', 'task', 'expl_method']).
                         index.isin(_chosen_models_accuracy.set_index(['model_seed', 'task', 'expl_method']).index))

    print(f"{task} {df_DNFS['expl_method'].unique()}")

    meta_data = _load_meta_data_by_pid(data_dir)
    md = meta_data[next(iter(meta_data))]

    X_train = md['X']  # test data from NN
    X_test = md['X_val']  # val data from NN becomes test data for rules
    if type(X_train) is not torch.Tensor:
        X_train = torch.from_numpy(X_train)
    if type(X_test) is not torch.Tensor:
        X_test = torch.from_numpy(X_test)
    # print(f"X size     = {len(X_train)}")
    # print(f"X_val size = {len(X_test)}")
    # continue

    # all_expl_methods = df_DNFS['expl_method'].unique()
    all_expl_methods = ['ks', 'li']
    if modelclass=='nn':
        all_expl_methods.append('ig')#D, 'ksub', 'ig', 'igub', 'li', 'liub',]
    _expl_methods = [e for e in all_expl_methods if len(e) == 2 or ('ub' in e and len(e) == 4)]
    # contains attributions and labels
    print(f"{task} - {modelclass} looking for: {_expl_methods}")

    attrs_tr, attrs_te = load_all_explanations(task,
                                               df_DNFS['model_seed'].unique(),
                                               _expl_methods,
                                               modelclass)
    # preload all models

    _eval = []
    # for _, row in tqdm(df_DNFS.iterrows(), total=len(df_DNFS), desc=f'{task} - {modelclass}'):
    grouped_df = df_DNFS.groupby('model_seed')
    for _seed, group in tqdm(grouped_df, total=len(grouped_df), desc=f'{task} - {modelclass}'):
        _collected_seed_descriptions = {}
        _collected_seed_descriptions['te'] = {}
        _collected_seed_descriptions['tr'] = {}
        for _, row in group.iterrows():
            # model_seed, expl_method, labels
            dnf = row.dnf
            _chosen_model = row['chosen']
            dnf.assert_no_infty()
            dnf.tie_break = 'accuracy'
            _expl_method = row.expl_method
            seed = row.model_seed
            is_single_expl_method = len(_expl_method) == 2 or ('ub' in _expl_method and len(_expl_method) == 4)
            if is_single_expl_method:
                _attr = attrs_tr[seed][_expl_method]
                attrs_tr_binarized = np.array(__preprocess_explanations(_attr,
                                                                        filtering=float(row.significance_threshold)) > 0, dtype=int)
                _att_tr_dims = list([set(np.argwhere(e).reshape(-1)) for e in attrs_tr_binarized])
                _atte = attrs_te[seed][_expl_method]
                attrs_te_binarized = np.array(__preprocess_explanations(_atte,
                                                                        filtering=float(row.significance_threshold)) > 0, dtype=int)
                _att_te_dims = list([set(np.argwhere(e).reshape(-1)) for e in attrs_te_binarized])
            else:
                _att_tr_dims = None
                _att_te_dims = None

            try:
                model = __LOADED_MODELS[modelclass][seed]
            except KeyError:
                if modelclass not in __LOADED_MODELS.keys():
                    __LOADED_MODELS[modelclass] = {}
                _data_dir = _variables.get_data_dir(task, modelclass)
                if modelclass == 'nn':
                    __LOADED_MODELS[modelclass][seed] = load_idxs_from_model(_data_dir, task,
                                                                                  seed, idxs=[-1], return_fns=False)[0]
                else:
                    # sklearn
                    __LOADED_MODELS[modelclass][seed] = load_sklearn_models(_data_dir, [seed], True)[0]
                model = __LOADED_MODELS[modelclass][seed]
            inference_fn = model.predict_batch_softmax if modelclass == 'nn' else model.forward

            def check_description_concistency(desc, y_true):
                _class_has_description = [len(d)>0 for d in desc]
                if sum(_class_has_description) > 1:
                    return DNFConsistencyReturnCode.AMBIGUOUS

                try:
                    predicted_label = np.argwhere(_class_has_description).item()
                except ValueError as e:
                    if not any(_class_has_description):
                        return DNFConsistencyReturnCode.NOT_COVERED

                prediction_correct = predicted_label == y_true
                class_desc = desc[predicted_label]
                if len(class_desc) == 1:  # len == number of conjunctions that were applicable in class-DNF
                    if prediction_correct:
                        return DNFConsistencyReturnCode.CORRECT_SINGLE
                    else:
                        return DNFConsistencyReturnCode.WRONG_SINGLE
                else:
                    itemsets = []  # Extract sets of dimensions that were used in each conjunction
                    for conjunction in class_desc:
                        itemsets.append(set([t[0] for t in conjunction]))
                    largest_itemset = max(itemsets, key=len)
                    # check for all itemsets if they are subsets of the largest_itemset
                    consistent = np.all([i <= largest_itemset for i in itemsets])

                    if prediction_correct:
                        if consistent:
                            return DNFConsistencyReturnCode.CORRECT_MULTI
                        else:
                            return DNFConsistencyReturnCode.CORRECT_INCONSISTENT
                    else:
                        if consistent:
                            return DNFConsistencyReturnCode.WRONG_MULTI
                        else:
                            return DNFConsistencyReturnCode.WRONG_INCONSISTENT

            def eval_as_local_explanation(desc, y_true):
                desc_y = desc[y_true]
                if len(desc_y) == 0 or desc_y == [[]]:
                    return DNFConsistencyReturnCode.NOT_COVERED
                if len(desc_y) == 1:
                    return DNFConsistencyReturnCode.CORRECT_SINGLE

                itemsets = []  # Extract sets of dimensions that were used in each conjunction
                for conjunction in desc_y:
                    itemsets.append(set([t[0] for t in conjunction]))
                largest_itemset = max(itemsets, key=len)
                # check for all itemsets if they are subsets of the largest_itemset
                consistent = np.all([i <= largest_itemset for i in itemsets])

                if consistent:
                    return DNFConsistencyReturnCode.CORRECT_MULTI
                else:
                    return DNFConsistencyReturnCode.CORRECT_INCONSISTENT


            def eval_precision_agreement_local_explanation(desc, y_true, e):
                desc_y = desc[y_true]
                if len(desc_y) == 0 or desc_y == [[]]:
                    return np.nan

                itemset = []  # Extract sets of dimensions that were used in each conjunction
                for conjunction in desc_y:
                    itemset.extend([t[0] for t in conjunction])
                itemset = set(itemset) # remove duplicates
                common = e.intersection(itemset)
                try:
                    precision = len(common) / len(itemset)
                except ZeroDivisionError:
                    precision = np.nan
                return precision


            def _add_keys(_d, source_keys):
                return {k: _d.get(k, 0) for k in source_keys}

            __local_expl_eval_keys = [DNFConsistencyReturnCode.NOT_COVERED.name,
                                                             DNFConsistencyReturnCode.CORRECT_MULTI.name,
                                                             DNFConsistencyReturnCode.CORRECT_INCONSISTENT.name,
                                                             DNFConsistencyReturnCode.CORRECT_SINGLE.name,]

            ytr = _get_targets (inference_fn, X_train, model, 'cpu').detach().numpy()
            dnf.compute_rule_performance(X_train, ytr)
            y_tr_pred = dnf.predict(X_train)
            tr_descriptions = dnf.describe(X_train)
            tr_descriptions_consistencies = [check_description_concistency(d, y) for d,y in zip(tr_descriptions, ytr)]
            tr_descriptions_consistencies_stats = {k.name: v for k, v in Counter(tr_descriptions_consistencies).items()}
            tr_descriptions_consistencies_stats = _add_keys(tr_descriptions_consistencies_stats, [k.name for k in DNFConsistencyReturnCode])
            # tr_descriptions_consistencies_stats_normalized = {k: v / sum(tr_descriptions_consistencies_stats.values())
            #                                                   for k, v in tr_descriptions_consistencies_stats.items()}
            tr_local_expls = [eval_as_local_explanation(d, y) for d,y in zip(tr_descriptions, ytr)]
            tr_local_expls_stats = {k.name: v for k,v in Counter(tr_local_expls).items()}
            tr_local_expls_stats = _add_keys(tr_local_expls_stats, __local_expl_eval_keys)
            tr_prec_agreement = np.nanmean([eval_precision_agreement_local_explanation(d, y, e)
                                  for d, y, e in zip(tr_descriptions, ytr, _att_tr_dims)])
            _y_tr_matches = y_tr_pred == ytr
            tr_prec_agreement_correct_prediction = []
            for _i, _matches in enumerate(_y_tr_matches):
                if _matches:
                    tr_prec_agreement_correct_prediction.append(
                        eval_precision_agreement_local_explanation(tr_descriptions[_i], ytr[_i], _att_tr_dims[_i])
                    )
            tr_prec_agreement_correct_prediction = np.nanmean(tr_prec_agreement_correct_prediction)

            # compute prec_agreement for cases where model gives a true label


            _collected_seed_descriptions['tr']['y'] = ytr
            _collected_seed_descriptions['tr'][_expl_method] = list(zip(_att_tr_dims, tr_descriptions))

            row['tr_consistency'] = tr_descriptions_consistencies_stats
            row['tr_local_eval'] = tr_local_expls_stats
            row['tr_prec_agreement'] = tr_prec_agreement
            row['tr_prec_agreement_correct_prediction'] = tr_prec_agreement_correct_prediction

            yte = _get_targets(inference_fn, X_test, model, 'cpu').detach().numpy()
            y_te_pred = dnf.predict(X_test)
            te_descriptions = dnf.describe(X_test)
            te_descriptions_consistencies = [check_description_concistency(d, y) for d,y in zip(te_descriptions, yte)]
            te_descriptions_consistencies_stats = {k.name: v for k,v in Counter(te_descriptions_consistencies).items()}
            te_descriptions_consistencies_stats = _add_keys(te_descriptions_consistencies_stats, [k.name for k in DNFConsistencyReturnCode])
            # te_descriptions_consistencies_stats_normalized = {k: v / sum(te_descriptions_consistencies_stats.values())
            #                                                   for k, v in te_descriptions_consistencies_stats.items()}

            te_local_expls = [eval_as_local_explanation(d, y) for d,y in zip(te_descriptions, yte)]
            te_local_expls_stats = {k.name: v for k,v in Counter(te_local_expls).items()}
            te_local_expls_stats = _add_keys(te_local_expls_stats, __local_expl_eval_keys)
            te_prec_agreement = np.nanmean([eval_precision_agreement_local_explanation(d, y, e)
                                  for d, y, e in zip(te_descriptions, yte, _att_te_dims)])

            _y_te_matches = y_te_pred == yte
            tr_prec_agreement_correct_prediction = []
            for _i, _matches in enumerate(_y_te_matches):
                if _matches:
                    tr_prec_agreement_correct_prediction.append(
                        eval_precision_agreement_local_explanation(te_descriptions[_i], yte[_i], _att_te_dims[_i])
                    )
            te_prec_agreement_correct_prediction = np.nanmean(tr_prec_agreement_correct_prediction)

            _collected_seed_descriptions['te']['y'] = yte
            _collected_seed_descriptions['te'][_expl_method] = list(zip(_att_te_dims, te_descriptions))

            row['te_consistency'] = te_descriptions_consistencies_stats
            row['te_local_eval'] = te_local_expls_stats
            row['te_prec_agreement'] = te_prec_agreement
            row['te_prec_agreement_correct_predictions'] = te_prec_agreement_correct_prediction
            row = row.to_dict()
            _eval.append(row)

        _ensemble_eval = _eval_ensembling_expl_methods(_collected_seed_descriptions, deepcopy(row))
        _eval.append(_ensemble_eval)


    eval_df = pd.DataFrame(_eval)


    pth = f'./plots/cfire/{modelclass}/{task}'
    dump_pkl(eval_df, str(Path(pth, 'consistency.pkl')))
    return


def eval_dnfclassifiers(task, modelclass):

    print(f"eval dnf {task}-{modelclass}")

    if len(__LOADED_MODELS) > 0:
        __LOADED_MODELS.clear()

    data_dir = _variables.get_data_dir(task, modelclass)
    df_DNFS: pd.DataFrame = load_dnfclassifiers(task, modelclass)
    print(f"{task} {df_DNFS['expl_method'].unique()}")

    meta_data = _load_meta_data_by_pid(data_dir)
    md = meta_data[next(iter(meta_data))]

    X_train = md['X']  # test data from NN
    X_test = md['X_val']  # val data from NN becomes test data for rules
    if type(X_train) is not torch.Tensor:
        X_train = torch.from_numpy(X_train)
    if type(X_test) is not torch.Tensor:
        X_test = torch.from_numpy(X_test)
    # print(f"X size     = {len(X_train)}")
    # print(f"X_val size = {len(X_test)}")
    # continue

    # all_expl_methods = df_DNFS['expl_method'].unique()
    all_expl_methods = ['ks', 'li']
    if modelclass=='nn':
        all_expl_methods.append('ig')#D, 'ksub', 'ig', 'igub', 'li', 'liub',]
    _expl_methods = [e for e in all_expl_methods if len(e) == 2 or ('ub' in e and len(e) == 4)]
    # contains attributions and labels
    print(f"{task} - {modelclass} looking for: {_expl_methods}")

    attrs_tr, attrs_te = load_all_explanations(task,
                                               df_DNFS['model_seed'].unique(),
                                               _expl_methods,
                                               modelclass)
    # preload all models

    _eval_test = []
    _eval_train = []
    for _, row in tqdm(df_DNFS.iterrows(), total=len(df_DNFS), desc=f'{task} - {modelclass}'):
        # model_seed, expl_method, labels
        dnf = row.dnf
        dnf.assert_no_infty()
        dnf.tie_break = 'accuracy'
        _expl_method = row.expl_method
        seed = row.model_seed
        # is_single_expl_method = len(_expl_method) == 2 or ('ub' in _expl_method and len(_expl_method) == 4)
        # if is_single_expl_method:
        #     _attr = attrs_tr[seed][_expl_method]
        #     _att_tr_dims = dims_from_expl(_attr)
        #     _atte = attrs_te[seed][_expl_method]
        #     _att_te_dims = dims_from_expl(_atte)
        # else:
        #     _att_tr_dims = None
        #     _att_te_dims = None
        #
        # ytr = attrs_tr[seed]['Y']
        # yte = attrs_te[seed]['Y']

        _att_tr_dims = None
        _att_te_dims = None
        try:
            model = __LOADED_MODELS[modelclass][seed]
        except KeyError:
            if modelclass not in __LOADED_MODELS.keys():
                __LOADED_MODELS[modelclass] = {}
            _data_dir = _variables.get_data_dir(task, modelclass)
            if modelclass == 'nn':
                __LOADED_MODELS[modelclass][seed] = load_idxs_from_model(_data_dir, task,
                                                                              seed, idxs=[-1], return_fns=False)[0]
            else:
                # sklearn
                __LOADED_MODELS[modelclass][seed] = load_sklearn_models(_data_dir, [seed], True)[0]
            model = __LOADED_MODELS[modelclass][seed]
        inference_fn = model.predict_batch_softmax if modelclass == 'nn' else model.forward

        ytr = _get_targets (inference_fn, X_train, model, 'cpu').detach().numpy()
        yte = _get_targets(inference_fn, X_test, model, 'cpu').detach().numpy()

        dnf.compute_rule_performance(X_train, ytr)
        _tr_explain = dnf.predict(X_train, explain=True)
        _y_tr_pred, _tr_rules = np.array([r[0] for r in _tr_explain]), [r[1] for r in _tr_explain]
        _tr_avg_n, _tr_avg_len = _compute_expl_len_statistics(_tr_explain)
        _tr_dims = [dims_from_rules(r) for r in _tr_rules]

        _te_explain = dnf.predict(X_test, explain=True)
        _te_avg_n, _te_avg_len = _compute_expl_len_statistics(_tr_explain)
        _y_te_pred, _te_rules = np.array([r[0] for r in _te_explain]), [r[1] for r in _te_explain]
        _te_dims = [dims_from_rules(r) for r in _te_rules]

        # tree baseline
        param_grid = {'max_depth': range(1, 10)}
        grid_search = GridSearchCV(DT(), param_grid, cv=int(4), scoring='accuracy')
        # Fit the GridSearchCV object to the data
        grid_search.fit(X_train, ytr)
        dt = grid_search.best_estimator_
        dt_n_unique_nodes = tree_count_unique_internal_nodes(dt)

        _rd = row.to_dict()
        tr_eval = compute_eval(ytr, _y_tr_pred, _att_tr_dims, _tr_dims)
        _tr_eval_copy = deepcopy(tr_eval)
        tr_eval.update(deepcopy(_rd))

        _y_tr_dt, _dt_dims_tr = tree_predictions_paths(dt, X_train,)
        _dt_tr_eval = compute_eval(ytr, _y_tr_dt, _att_tr_dims, _dt_dims_tr)
        _dt_tr_eval = {'tree_'+k:v for k, v in _dt_tr_eval.items()}
        _dt_tr_eval['tree_n_unique_literals'] = dt_n_unique_nodes
        tr_eval.update(_dt_tr_eval)

        tr_eval['empirical_avg_n_rules'] = _tr_avg_n
        tr_eval['empirical_avg_len'] = _tr_avg_len

        _eval_train.append(tr_eval)


        te_eval = compute_eval(yte, _y_te_pred, _att_te_dims, _te_dims)
        te_eval.update(deepcopy(_rd))

        _y_te_dt, _dt_dims_te = tree_predictions_paths(dt, X_test,)
        common_pairs = [(a, b) for a, b in zip(_te_dims, _dt_dims_te) if len(a) > 0]
        dnf_dt_dim_agreement = np.mean([len(set(a).intersection(set(b)))/max(len(a), len(b)) for (a, b) in common_pairs])
        _dt_te_eval = compute_eval(yte, _y_te_dt, _att_te_dims, _dt_dims_te)
        _dt_te_eval = {'tree_'+k:v for k, v in _dt_te_eval.items()}
        _dt_te_eval['tree_n_unique_literals'] = dt_n_unique_nodes
        _dt_te_eval['dnf_dt_dim_agreement'] = dnf_dt_dim_agreement
        te_eval.update(_dt_te_eval)

        # COMP PGI SCORES, expensive much
        # pgi score is nan if no dims are provided (ie sample rejected)

        norm_pgi = lambda pgi, dims: [score / len(d) if score != np.nan else np.nan for score, d in
                                 zip(pgi, dims)]
        pbm = 'masked'

        pgi_masked_dnf_te = comp_pgi_scores(X_test, _te_dims, task, modelclass, row.model_seed, perturbation_mode=pbm,
                                           data_seed=md['data_seed'])
        pgi_masked_dnf_te_normalized = norm_pgi(pgi_masked_dnf_te, _te_dims)
        pgi_masked_dt_te = comp_pgi_scores(X_test, _dt_dims_te, task, modelclass, row.model_seed, perturbation_mode=pbm,
                                           data_seed=md['data_seed'])
        pgi_masked_dt_te_normalized = norm_pgi(pgi_masked_dt_te, _dt_dims_te)
        te_eval['pgi_masked'] = pgi_masked_dnf_te
        te_eval['pgi_masked_normalized'] = pgi_masked_dnf_te_normalized
        te_eval['tree_pgi_masked'] = pgi_masked_dt_te
        te_eval['tree_pgi_masked_normalized'] = pgi_masked_dt_te_normalized
        te_eval['sufficiency_masked'] = comp_suff_scores(X_test, _te_dims, task, modelclass, seed, perturbation_mode=pbm,
                                           data_seed=md['data_seed'])
        for k, v in _tr_eval_copy.items():
            te_eval[f'tr_{k}'] = v

        # pbm = 'sampling'
        # pgi_sampling_dnf_te = comp_pgi_scores(X_test, _te_dims, task, modelclass, row.model_seed, perturbation_mode=pbm,
        #                                            data_seed=md['data_seed'])
        # pgi_sampling_dnf_te_normalized = norm_pgi(pgi_sampling_dnf_te, _te_dims)
        # pgi_sampling_dt_te = comp_pgi_scores(X_test, _dt_dims_te, task, modelclass, row.model_seed, perturbation_mode=pbm,
        #                                            data_seed=md['data_seed'])
        # pgi_sampling_dt_te_normalized = norm_pgi(pgi_sampling_dt_te, _dt_dims_te)
        # te_eval['pgi_sampling'] = pgi_sampling_dnf_te
        # te_eval['pgi_sampling_normalized'] = pgi_sampling_dnf_te_normalized
        # te_eval['tree_pgi_sampling'] = pgi_sampling_dt_te
        # te_eval['tree_pgi_sampling_normalized'] = pgi_sampling_dt_te_normalized

        te_eval['empirical_avg_n_rules'] = _te_avg_n
        te_eval['empirical_avg_len'] = _te_avg_len
        _eval_test.append(te_eval)

    _eval_test_df = pd.DataFrame(_eval_test)
    _eval_train_df = pd.DataFrame(_eval_train)
    # assert np.all([e in all_expl_methods for e in _eval_test_df['expl_method'].unique() ])
    pth = Path(f'./plots/cfire/{modelclass}/{task}/')
    pth.mkdir(exist_ok=True, parents=True)
    dump_pkl(_eval_test_df, str(Path(pth, 'te_eval.pkl')))
    dump_pkl(_eval_train_df, str(Path(pth, 'tr_eval.pkl')))


def load_anchors(data_dir):
    anchors_dir = Path(data_dir, "anchors")
    fnames = os.listdir(anchors_dir)

    results = {}
    for fname in fnames:
        model_seed = fname.split('_')[1]
        full_path = Path(anchors_dir, fname)
        anchors = load_pkl(full_path)
        results[model_seed] = anchors

    return results



def anchor_to_rule(a):
    predicted_class = int(a['prediction'])
    rule = []
    for t in a['names']:
        _t_split = t.split(' ')
        if len(_t_split) == 3:
            dim_str, relation_str, val_str = _t_split
            dim = int(dim_str[1:]) # eg "V5", STARTING AT **ZERO**
            val = float(val_str)

            left = val if relation_str[0] == '>' else -np.inf
            right = val if relation_str[0] == '<' else np.inf
        else: # len == 5, eg "-0.58 < V6 <= -0.17".split(" ")
            left_str, _, dim_str, _, right_str = _t_split
            dim = int(dim_str[1:])
            left = float(left_str)
            right = float(right_str)

        r = (dim, (left, right))
        rule.append(r)
    return (predicted_class, rule)


def check_applicability(data, rules):
    # Start with an array of True values with the same length as X
    mask = np.ones(data.shape[0], dtype=bool)

    # Iterate over the conditions
    for dimension, (left, right) in rules:
        # Apply the rule to the specified dimension
        condition = np.logical_and((left < data[:, dimension]), (data[:, dimension] <= right))
        # Combine with the previous result using a logical AND
        mask = np.logical_and(mask, condition)

    return np.argwhere(mask)


def anchors_to_rule_model(anchors, n_classes, tie_break):
    rules = [anchor_to_rule(a) for a in anchors]
    rules_classes = []
    for c in range(n_classes):
        rules_classes.append([])
        rc = [r[1] for r in rules if r[0] == c]
        rules_classes[-1].extend(rc)
    dnf = DNFClassifier(rules_classes, tie_break=tie_break)
    dnf._meta_information = dict(time=sum([a['time'] for a in anchors]))
    return dnf

def greedy_coverage_anchor(explanations, data, n_classes, tie_break='accuracy'):
    # pick all anchors until all samples are covered
    rules = [anchor_to_rule(e) for e in explanations]
    rules = [r[1] for r in rules]
    applicability = []
    for r in rules:
        a = check_applicability(data, r)
        applicability.append(a)
    applicability = [a.squeeze() for a in applicability]
    applicability = [set() if a.ndim == 0 else set(a) for a in applicability]
    # for some reason we sometimes lose a sample!? like an anchor is not applicable to any datapoint? sounds wrong!
    items = set.union(*applicability)
    if len(items) != len(explanations):
        print(f"lost items: {set(np.arange(len(explanations))) - items}")
        # for i, a in enumerate(applicability):
        #     if len(a) == 0:
        #         print("anchor not applicable: ")
        #         for e, r in zip(explanations[i]['names'], rules[i]):
        #             print(f"\t {e}\n \t\t -> {r}")
        #         print()
        # raise ValueError
    rule_idxs = greedy_set_cover(items, applicability)
    picked_anchors = [explanations[i] for i in rule_idxs]
    return anchors_to_rule_model(picked_anchors, n_classes, tie_break)

def greedy_precision_anchor(anchors, data, n_classes, tie_break='accuracy', k=None):
    # pick all anchors until all samples are covered
    rules = [anchor_to_rule(e) for e in anchors]
    rules = [r[1] for r in rules]
    applicability = []
    for r in rules:
        a = check_applicability(data, r)
        applicability.append(a)
    applicability = [a.squeeze() for a in applicability]
    applicability = [set() if a.ndim == 0 else set(a) for a in applicability]
    # for some reason we sometimes lose a sample!? like an anchor is not applicable to any datapoint? sounds wrong!
    items = set.union(*applicability)
    if len(items) != len(anchors):
        print(f"lost items: {set(np.arange(len(anchors))) - items}")
        # for i, a in enumerate(applicability):
        #     if len(a) == 0:
        #         print("anchor not applicable: ")
        #         for e, r in zip(explanations[i]['names'], rules[i]):
        #             print(f"\t {e}\n \t\t -> {r}")
        #         print()
        # raise ValueError
    def greedy_precision_cover(X, F, k=None):
        U = X.copy()
        C = []
        if k is None:
            k = np.inf
        while len(U) > 0 and len(C) < k:
            _f = sorted(F, key=lambda x: -x[1]['precision'])[0]
            F.remove(_f)

            U_new = U - _f[0]
            # check if _f increased coverage, else discard
            if len(U_new) < len(U):
                C.append(_f[1])
            U = U_new
        return C
    applicability = list(zip(applicability, anchors))
    picked_anchors = greedy_precision_cover(items, applicability, k=k)
    dnf = anchors_to_rule_model(picked_anchors, n_classes, tie_break)
    return dnf




from anchor.anchor.utils import greedy_pick_anchor
# def choose_anchors_cv(anchors, data, max_k=100, nfolds=5):
#     # choose set of anchors for k == 1 .. max_k
#     # that performs best on average over nfolds
#
#     for k in range(0, max_k+1):


def __eval_anchor_dnf(dnf, X, X_te, Y, Y_te, task, seed, metadata, modelclass):
    dnf.compute_rule_performance(X, Y)

    _random_baselines_tr = _rnd_baseline(Y)
    _random_baselines_te = _rnd_baseline(Y_te)

    rd = dict(model_seed=seed, task=task, dnf=dnf, rnd_te=_random_baselines_te, rnd_tr=_random_baselines_tr)

    _tr_explain = dnf.predict(X, explain=True)
    _y_tr_pred, _tr_rules = np.array([r[0] for r in _tr_explain]), [r[1] for r in _tr_explain]
    _tr_dims = [dims_from_rules(r) for r in _tr_rules]
    _te_explain = dnf.predict(X_te, explain=True)
    _y_te_pred, _te_rules = np.array([r[0] for r in _te_explain]), [r[1] for r in _te_explain]
    _te_dims = [dims_from_rules(r) for r in _te_rules]

    _tr_avg_n, _tr_avg_len = _compute_expl_len_statistics(_tr_explain)
    _te_avg_n, _te_avg_len = _compute_expl_len_statistics(_tr_explain)

    tr_eval = compute_eval(Y, _y_tr_pred, None, _tr_dims)
    _tr_eval_copy = deepcopy(tr_eval)
    tr_eval.update(deepcopy(rd))


    te_eval = compute_eval(Y_te, _y_te_pred, None, _te_dims)
    te_eval.update(deepcopy(rd))
    for k, v in _tr_eval_copy.items():
        te_eval[f'tr_{k}'] = v

    # COMP PGI SCORES, expensive much
    # pgi score is nan if no dims are provided (ie sample rejected)

    norm_pgi = lambda pgi, dims: [score / len(d) if score != np.nan else np.nan for score, d in
                                  zip(pgi, dims)]
    pbm = 'masked'
    pgi_masked_dnf_te = comp_pgi_scores(X_te, _te_dims, task, modelclass, seed, perturbation_mode=pbm,
                                        data_seed=metadata['data_seed'])
    pgi_masked_dnf_te_normalized = norm_pgi(pgi_masked_dnf_te, _te_dims)
    te_eval['pgi_masked'] = pgi_masked_dnf_te
    te_eval['pgi_masked_normalized'] = pgi_masked_dnf_te_normalized
    te_eval['sufficiency_masked'] = comp_suff_scores(X_te, _te_dims, task, modelclass, seed,
                                                     perturbation_mode=pbm,
                                                     data_seed=metadata['data_seed'])
    # pbm = 'sampling'
    # pgi_sampling_dnf_te = comp_pgi_scores(X_test, _te_dims, task, modelclass, seed, perturbation_mode=pbm,
    #                                            data_seed=md['data_seed'])
    # pgi_sampling_dnf_te_normalized = norm_pgi(pgi_sampling_dnf_te, _te_dims)
    # te_eval['pgi_sampling'] = pgi_sampling_dnf_te
    # te_eval['pgi_sampling_normalized'] = pgi_sampling_dnf_te_normalized
    te_eval['pgi_sampling'] = None
    te_eval['pgi_sampling_normalized'] = None

    tr_eval['empirical_avg_n_rules'] = _tr_avg_n
    tr_eval['empirical_avg_len'] = _tr_avg_len
    te_eval['empirical_avg_n_rules'] = _te_avg_n
    te_eval['empirical_avg_len'] = _te_avg_len
    return te_eval, tr_eval

def eval_anchors(task, modelclass):

    # load all anchors available for modelclass + task
    # run submodular pick to select best performing anchors set
    # computestuff?


    print(f"eval ANCHORS {task}-{modelclass}")

    if len(__LOADED_MODELS) > 0:
        __LOADED_MODELS.clear()

    data_dir = _variables.get_data_dir(task, modelclass)
    anchors = load_anchors(data_dir)

    # md = load_meta_data(data_dir, task)
    meta_data = _load_meta_data_by_pid(data_dir)
    md = meta_data[next(iter(meta_data))]

    X_train = md['X']  # test data from NN
    X_test = md['X_val']  # val data from NN becomes test data for rules
    if type(X_train) is not torch.Tensor:
        X_train = torch.from_numpy(X_train)
    if type(X_test) is not torch.Tensor:
        X_test = torch.from_numpy(X_test)
    # print(f"X size     = {len(X_train)}")
    # print(f"X_val size = {len(X_test)}")
    # continue

    _eval_test = []
    _eval_train = []

    import pandas as pd
    n_rules_df = pd.read_csv('./n_rules_dnfclassifiers.csv')

    for seed, _anchors in anchors.items():

        try:
            model = __LOADED_MODELS[modelclass][seed]
        except KeyError:
            if modelclass not in __LOADED_MODELS.keys():
                __LOADED_MODELS[modelclass] = {}
            _data_dir = _variables.get_data_dir(task, modelclass)
            if modelclass == 'nn':
                __LOADED_MODELS[modelclass][seed] = load_idxs_from_model(_data_dir, task,
                                                                              seed, idxs=[-1], return_fns=False)[0]
            else:
                # sklearn
                __LOADED_MODELS[modelclass][seed] = load_sklearn_models(_data_dir, [seed], True)[0]
            model = __LOADED_MODELS[modelclass][seed]
        inference_fn = model.predict_batch_softmax if modelclass == 'nn' else model.forward

        ytr = _get_targets (inference_fn, X_train, model, 'cpu').detach().numpy()
        yte = _get_targets(inference_fn, X_test, model, 'cpu').detach().numpy()

        def anchors_compute_precision(anchors, X, Y, n_classes):
            for a in anchors:
                rule = anchor_to_rule(a)
                c, r = rule
                app = check_applicability(X, r).squeeze()
                yapp = np.zeros_like(Y)
                yapp[app] = c
                a['precision'] = sklearn.metrics.precision_score(Y, yapp, zero_division=0, average='micro')
            pass

        n_classes = __info_dim_classes[task][1]
        # manipulate in place
        anchors_compute_precision(_anchors, np.array(X_train), np.array(ytr), n_classes)

        # dnf_cover = greedy_coverage_anchor(deepcopy(_anchors), np.array(X_train), n_classes)
        k = n_rules_df[
            (n_rules_df['task'] == task) &
            (n_rules_df['modelclass'] == modelclass) &
            (n_rules_df['model_seed'] == int(seed))
        ]['n_rules'].values
        assert len(k) == 1
        k = max(1, k[0])
        dnf_cover = greedy_precision_anchor(deepcopy(_anchors), np.array(X_train), n_classes, k=k)
        dnf_precision = greedy_precision_anchor(deepcopy(_anchors), np.array(X_train), n_classes)

        te_eval_cover, tr_eval_cover = (
            __eval_anchor_dnf(dnf_cover, X_train, X_test, ytr, yte, task, seed, md, modelclass))
        te_eval_cover['mode'] = 'cover'
        tr_eval_cover['mode'] = 'cover'
        _eval_test.append(te_eval_cover)
        _eval_train.append(tr_eval_cover)
        te_eval_prec, tr_eval_prec = (
            __eval_anchor_dnf(dnf_precision, X_train, X_test, ytr, yte, task, seed, md, modelclass))
        # te_eval_prec_k, tr_eval_prec_k = (
        #     __eval_anchor_dnf(dnf_precision_k, X_train, X_test, ytr, yte, task, seed, md, modelclass))
        te_eval_prec['mode'] = 'precision'
        tr_eval_prec['mode'] = 'precision'
        _full_time = sum([a['time'] for a in _anchors])
        te_eval_prec['time'] = _full_time
        te_eval_prec['optimal_time'] = te_eval_prec['dnf']._meta_information['time']
        te_eval_cover['time'] = _full_time
        te_eval_cover['optimal_time'] = te_eval_cover['dnf']._meta_information['time']
        tr_eval_cover['time'] = _full_time
        tr_eval_cover['optimal_time'] = tr_eval_cover['dnf']._meta_information['time']
        tr_eval_prec['time'] = _full_time
        tr_eval_prec['optimal_time'] = tr_eval_prec['dnf']._meta_information['time']
        _eval_test.append(te_eval_prec)
        _eval_train.append(tr_eval_prec)


    _eval_test_df = pd.DataFrame(_eval_test)
    _eval_train_df = pd.DataFrame(_eval_train)

    pth = Path(f'./plots/cfire/{modelclass}/{task}')
    pth.mkdir(exist_ok=True, parents=True)
    dump_pkl(_eval_test_df, str(Path(pth, 'anchors_te_eval.pkl')))
    dump_pkl(_eval_train_df, str(Path(pth, 'anchors_tr_eval.pkl')))


def composednf(task, modelclass):

    print()
    print(f"compose dnfs for {modelclass}s on {task}")
    data_dir = _variables.get_data_dir(task, modelclass)
    # _ex = [['ks'], ['li']]
    _ex = [['ks'], ['li']]#, ['ds']]
    if modelclass == 'nn':
        _ex.append(['ig'])#['ks'], ['ksub'], ['ig'], ['igub'], ['li'], ['liub'],  # single methods
         # ['igksub'], ['ksubli'], ['ksksub'], ['igubksubliub'], ['igksli'],  # methods combined at itemset mining stage
     # ['ksub', 'li'], ['ksub', 'ks'], ['ksub', 'ig'],  # methods combined at composition stage
     # ['ksub', 'igub', 'liub'], ['ks', 'ig', 'li', ]   # more methods combined at composition stage


    selection_params = dict(
        gt=None,  # gely_threshold=0.8,
        ex=_ex, #[['ig'], ['igub'], ['ks'], ['ksub'], ['li'],  # results for base methods
        #     ['liub'], ['igksub'], ['ksubli'], ['ksksub'],  # some attributions combined at itemsetorder-stage
        #     ['ig', 'ksub'], ['ks', 'ksub'], ['ksub', 'li']  # combine results for some diff attr methods at composition stage
        #     ],  # expl_method='ig', can be single string or list
        km=None,  # k_means_max_bins=2,
        s=None,  # model_seed=None,
        fs=None,  # n_sam ples_percent=1.0,
        sc=None,  # setcover_reduction=True,
        st=0.01,  #'topk0.5',  # significance_threshold='top0.5',
        # giso=None
    )
    # print(selection_params)
    # import sys;sys.exit()
    accs = load_accuracies(data_dir)
    accs = [v[-1] for v in accs.values()]
    # model_id_accs = lxg.util.get_top_k_models(data_dir, k=20)
    # model_id = model_id_accs[0][0]

    # load labels according to model output, obtain model output for validation set to test rules on

    if 'classification' in task:
        data_fname = Path(data_dir, 'data.pkl')
        data = load_pkl(data_fname)
        X_train, Y_train = data['validation']
        X_te, Y_te = data['test']
    elif 'hypercube' in task:
        data_fname = Path(data_dir, 'data.pkl')
        data = load_pkl(data_fname)
        X_train = data['X_te']
        Y_train = data['Y_te']
        X_te = data['X_val']
        Y_te = data['Y_val']
        meta_data = _load_meta_data_by_pid(data_dir)
        md = next(iter(meta_data.values()))
        assert np.all(np.isclose(md['X'], X_train))
    else:
        meta_data = _load_meta_data_by_pid(data_dir)
        md = next(iter(meta_data.values()))
        # X is test data for NN that the explanations are computed for; this becomes the train set for the DNF
        # in turn the validation set is used to assess generalization of DNF
        X_train = md['X']
        Y_train = md['Y']
        X_te = md['X_val']
        Y_te = md['Y_val']

        print(np.mean(np.array(Y_te)))

    if type(X_train) is torch.Tensor:
        X_train = X_train.detach().cpu().numpy()
    if type(Y_train) is torch.Tensor:
        Y_train = Y_train.detach().cpu().numpy()
    if type(X_te) is torch.Tensor:
        X_te = X_te.detach().cpu().numpy()
    if type(Y_te) is torch.Tensor:
        Y_te = Y_te.detach().cpu().numpy()

    # ---

    # load gely stuff
    # for each complexity level do sth
    # for each selection strategy build a dnf
    # eval

    # ---

    # load all dnfs for all tasks, matching selection_params (param None -> return all)
    # path eg:
    # data/cfire/NN_dnf/classification-20000-4-4-0-0-16-1-0.01/DNF_t-classification-20000-4-4-0-0-16-1-0.01_ex-sg_gt-0.05_st-0.01_km-1_s-[300804]_fs-1.0_sc-False.pkl
    # -> data_path/nn_postfix/task/DNF_{config}.pkl
    # DNFmodels = load_nn_dnfs(task, selection_params)[:1]
    def dicts_differ_by_specific_key(dict1, dict2, key):
        return dict1.keys() == dict2.keys() and all(dict1[k] == dict2[k] for k in dict1 if k != key) and dict1[key] != dict2[key]

    def get_matching_idxs(nl, node):
        idxs = []
        for i in range(len(nl)):
            n2 = nl[i]
            if node != n2 and dicts_differ_by_specific_key(node['args'], n2['args'], 'expl_method'):
                idxs.append(i)
        return idxs

    def merge_nodelist_dicts(node, to_merge):
        merged = deepcopy(node)
        args = merged['args']
        exs = [args['expl_method']] + [t['args']['expl_method'] for t in to_merge]
        exs = 'COMP'+''.join(sorted(exs))
        merged['args']['expl_method'] = exs
        for t in to_merge:
            for k, v in t.items():
                if type(k) is int:
                    merged[k][0].extend(v[0])
                    merged[k][1].extend(v[1])
        return merged


    def merge_nodelists_across_expls(nl):
        merged_nl = []
        n_iter, n_max_iter = 0, len(nl)
        while len(nl) > 0:
            if n_iter > n_max_iter:
                raise ValueError
            n = nl[0]
            idxs_to_merge = get_matching_idxs(nl, n)
            if len(idxs_to_merge) > 0:
                to_merge = [nl[i] for i in idxs_to_merge]
                merged = merge_nodelist_dicts(n, to_merge)
                assert type(merged['args']['expl_method']) is not list
                merged_nl.append(merged)
                nl = [nn for i, nn in enumerate(nl) if i not in [0]+idxs_to_merge]
            else:
                merged_nl.append(n)
                nl = nl[1:]
            n_iter += 1
        return merged_nl

    from cfire.util import load_expl_rules

    NodeLists = []
    if type(selection_params['ex']) is list and len(selection_params['ex']) > 1:
        for ex in selection_params['ex']:
            print(f"{task} - {modelclass} merging nodes for {ex}")
            s = deepcopy(selection_params)
            if len(ex) == 1:
                s['ex'] = ex[0]
                _nodelist = load_expl_rules(task, s, modelclass=modelclass)
                NodeLists.extend(_nodelist)
            else:
                s['ex'] = ex
                ex_NodeLists = load_expl_rules(task, s, modelclass=modelclass)
                _merged_nodelists = merge_nodelists_across_expls(ex_NodeLists)
                NodeLists.extend(_merged_nodelists)
    else:
        NodeLists = load_expl_rules(task, selection_params, modelclass=modelclass)


    for l in NodeLists:
        for _c in range(1000): # 1000 classes
            try:
                _n = l[_c]
                if _n == []:
                    continue
                _n = _n[1][0].get_frequent_children()
                for node in _n:
                    node.dnf.assert_no_infty()
            except KeyError:
                break


    from cfire.nodeselection import comp_variants_dnfs
    c_max = 10
    # Nodelists[models][class][(support sets of frequent nodes, full gely_tree)]
    # some_frequent_node = NodeLists[0][0][1][0].get_frequent_children()[0]
    _all_dnfs = []
    _all_args = []
    # complexity_min, complexity_max = int(np.ceil(c_max * 0.1)), c_max
    n_complexities = 1
    _model_seeds = np.unique([n['args']['model_seed'] for n in NodeLists])
    _expl_methods = np.unique([n['args']['expl_method'] for n in NodeLists])
    # attributions_tr, attributions_te = load_all_attributions(_model_seeds, _expl_methods)
    Y_tr = [n['model_pred_te'].detach().numpy() for n in NodeLists]
    Y_te = [n['model_pred_val'].detach().numpy() for n in NodeLists]


    _rnd_bl_tr = [_rnd_baseline(y) for y in Y_tr]
    _rnd_bl_te = [_rnd_baseline(y) for y in Y_te]
    for i, n in enumerate(NodeLists):
        n['args']['rnd_tr'] = _rnd_bl_tr[i]
        n['args']['rnd_te'] = _rnd_bl_te[i]
        try:

            n['args']['time'] = n['time']
        except KeyError:
            print(n['args'])
            assert False

    def generate_weight_combinations(n):
        """
        Generate equidistant weight combinations on a 2-simplex (triangle).
        Each combination consists of three weights that sum to 1.

        :param n: Number of divisions along each edge of the simplex
        :return: Array of weight combinations
        """
        weights = []
        for i in range(n + 1):
            for j in range(n + 1 - i):
                k = n - i - j
                weights.append([i / n, j / n, k / n])
        return np.array(weights)


    # Generate weight combinations
    n = 4  # Number of divisions along each edge
    # score_weights = generate_weight_combinations(n)
    # score_weights = np.array([[1., 0., 0.],
    #                           [0.5, 0.5, 0.],
                              # [0., 0., 1.]])
    score_weights = None
    # print(score_weights)

    # def score_fn(node: ItemsetNode, acc_weight, complex_weight, complete_weight):
    #     return node.score(acc_weight=acc_weight, cx_weight=complex_weight, cs_weight=complete_weight)


    # for c in range(complexity_min, complexity_max):
    #     _composed_dnfs, _args = compose_dnfs(NodeLists, scoring_weights=score_weights)
    #     for a in _args:
    #         a.update({'complexity_parameter':-1})
    _composed_dnfs, _args = compose_dnfs(NodeLists, scoring_weights=score_weights)
    for a in _args:
        a.update({'complexity_parameter':-1})
        a.update({'modelclass': modelclass})
    [_dnf.assert_no_infty() for _dnf in _composed_dnfs]
    _all_dnfs.extend(_composed_dnfs)
    _all_args.extend(_args)
    n_strats = len(_all_dnfs)/n_complexities / len(NodeLists)
    assert n_strats == int(n_strats); n_strats = int(n_strats)
    # Y_tr = Y_tr * (n_complexities * n_strats)
    # Y_te = Y_te * (n_complexities * n_strats)

    # Y_preds = [d(X_train) for d in _all_dnfs]
    # __accs_before_pruning = np.array([np.mean(yp == yt) for yp, yt in zip(Y_preds, Y_tr)])
    print("pruning DNFs")
    # _n_literals_bincount_before = np.bincount([a.n_literals for a in _all_dnfs])
    _cover_only_simplified = []
    _cover_only_simplified_args = []
    for a, d in tqdm(zip(_all_args, _all_dnfs), total=len(_all_dnfs), desc=f'{task} - {modelclass}'):

        # min_complexity -> if rules are redundant, keep shorter/longer if min_complexity=True/False
        # target is maximise cover with minimal num rules
        if a['composition_strategy'] == 'cover':
            # _d_simplified: DNFClassifier = deepcopy(d)
            # _d_simplified.remove_empirically_redundant_rules(X_train, min_complexity=True)
            # _a_simpl = deepcopy(a)
            # _a_simpl['composition_strategy'] = 'cover_mincomp'
            # _cover_only_simplified_args.append(_a_simpl)
            # _cover_only_simplified.append(_d_simplified)
            d.remove_empirically_redundant_rules(X_train, min_complexity=False)# min_complexity=a['cx_weight'] >= a['cs_weight'])
        else:  # we may want to consider accyarcy, complexity or completeness
            # accuracy
            if a['acc_weight'] == 1.:
                # remove redundant and keep longest
                d.remove_empirically_redundant_rules(X_train, min_complexity=False)
            # completeness
            elif ['cx_weight'] == 1.:
                # remove redundant rules and keep shortest
                d.remove_empirically_redundant_rules(X_train, min_complexity=True)
            elif a['cs_weight'] == 1.:  # if we target completeness, dont remove empiricall redundant rules
                # keep redundant rules, only merge
                d.simplify_merge_rules()
            else:
                raise NotImplementedError


        seed = a['model_seed']
        ytr_seed = None
        for n in NodeLists:
            if n['args']['model_seed'] == seed:
                ytr_seed = n['model_pred_te'].detach().numpy()
                break
        d.compute_rule_performance(X_train, ytr_seed)
        d.assert_no_infty()
    import pandas as pd

    # _all_dnfs.extend(_cover_only_simplified)
    # _all_args.extend(_cover_only_simplified_args)

    _all_rules_df = pd.DataFrame(_all_args)
    _all_rules_df['dnf'] = _all_dnfs
    _all_rules_df['modelclass'] = modelclass
    pth = _variables.get_dnfclassifier_dir(task, modelclass=modelclass)
    _variables.__create_dir(pth)
    print(f"saving rules {task} {modelclass} with expl method combinations {_all_rules_df['expl_method'].unique()}")
    dump_pkl(_all_rules_df, os.path.join(pth, f'{task}_dnfrules.pkl'))

def _rnd_baseline(labels):
    h = np.bincount(labels)
    majority_class = np.argmax(h)
    _l_m = labels == majority_class
    return np.mean(_l_m)

def make_parser():


    parser = argparse.ArgumentParser()

    parser.add_argument('--evalanchors', default=False, type=bool)
    parser.add_argument('--evalcega', default=False, type=bool)
    parser.add_argument('--evaldnf', default=False, type=bool)
    parser.add_argument('--composednf', default=False, type=bool)
    parser.add_argument('--evaldnfconsistency', default=False, type=bool)

    parser.add_argument('--modelclass', default='nn', type=str)
    parser.add_argument('--debug', default=False, type=bool)
    parser.add_argument('--cega-extend-results', type=bool, default=True)

    return parser


if __name__ == '__main__':


    args = make_parser().parse_args()

    modelclass = args.modelclass

    tasks = _variables.tasks

    if args.evalanchors is True:
        fun = eval_anchors
    elif args.evalcega:
        print("eval cega")
        fun = eval_cega
        if args.cega_extend_results:
            fun = lambda t, m: eval_cega(t, m, extend_results=False)
        tasks = [
            "breastcancer",
            "ionosphere",
            'btsc',
            'spf',
            'breastw',  # slow eval
            'heloc',  # slow eval
            'spambase',  # slow eval
        ]

    elif args.evaldnf:
        print("eval dnf")
        fun = eval_dnfclassifiers
    elif args.evaldnfconsistency:
        print("eval dnf consistency")
        fun = eval_dnfclassifiers_consistency
    elif args.composednf:
        print(f"compose dnfs")
        fun = composednf
    else:
        raise ValueError("one of args --evalcega, --evaldnf or --composednf has to be selected")

    modelclasses = ['nn']
    arg_sets = list(product(tasks, modelclasses))
    arg_sets = sorted(arg_sets, key=lambda x: np.argwhere(x[0]==np.array(tasks)).item())


    if args.debug:
        n_jobs = 1
    elif not args.evalcega:
        n_jobs = min(12, len(arg_sets))
    else:
        n_jobs = min(7, len(arg_sets))
        # n_jobs = 2
    print(f"starting with {n_jobs} jobs")
    with parallel_backend(backend='loky', n_jobs=n_jobs):
        Parallel(verbose=10, batch_size=2)(delayed(fun)(a[0], a[1])for a in arg_sets)
