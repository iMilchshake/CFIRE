from multiprocessing.managers import Value

import lxg.util

default_path = '../'
import os
os.chdir(default_path)
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm


from lxg.util import load_pkl, dump_pkl, load_meta_data
from lxg.models import DNFClassifier
import cfire._variables_cfire as _variables
from cfire._variables_cfire import __create_dir
from cfire.cfire import __preprocess_explanations

import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm
from tabulate import tabulate

# sns.set_palette('colorblind')

from CEGA.rules_model import RulesModel

from scipy import stats


def print_table_dnf_cega_anchors(df, compact=True):
    cega_applicable_tasks = [
        "breastcancer", "ionosphere", 'btsc', 'spf', 'breastw', 'heloc', 'spambase'
    ]
    grouped = df.groupby(['task', 'modelclass', 'alg'])
    stats = grouped.agg({
        'accuracy_no_rejected': ['mean', 'std'],
        'f1': ['mean', 'std'],
        'coverage': ['mean', 'std'],
        'suff': ['mean', 'std'],
        'time_short': ['mean', 'std'],
        'time_long': ['mean', 'std'],
        'le_rnd': 'sum',  # count models that have WORSE than random test performance
        'class_missing': 'sum',
        'n_rules': ['mean', 'std'],
        'n_avg_literals': ['mean', 'std'],
        'empirical_avg_n_rules': ['mean', 'std'],
        'fail': 'sum'
    })
    all_tasks = _variables.tasks_plotting_order
    if not compact:
        metrics = ['le_rnd', 'class_missing', 'f1',
                   'accuracy_no_rejected', 'coverage', 'n_rules',
                   'empirical_avg_n_rules',
                   'n_avg_literals',
                   # 'time_long',
                   # 'time_short',
                   ]  #, 'suff']
        metric_names = ['\#Models', 'Incomplete', 'F1',
                        'Precision', 'Coverage', '\#Rules',
                        '\\o ExLength',
                        '\\o Literals$',
                        # 'Time*',
                        # 'Time',
                        ]  #, 'RS']
    else:
        metrics = ['fail', 'f1', 'accuracy_no_rejected', 'n_rules']
        metric_names = ['\\xmark', 'F1', 'Precision', '\#Rules',
                        # 'Time*',
                        # 'Time',
                        ]  #, 'RS']
    # Determine unique model classes present in the DataFrame
    modelclasses = sorted(df['modelclass'].unique().tolist())
    def format_cell(task, modelclass, alg, metric):
        try:
            if metric == 'model_seed':
                value = stats.loc[(task, modelclass, alg), (metric, 'nunique')]
                return f"{value:.0f}" if pd.notnull(value) else '-'
            if metric in ['le_rnd', 'class_missing', 'fail']:
                value = stats.loc[(task, modelclass, alg), (metric, 'sum')]
                return f"{value:.0f}" if pd.notnull(value) else '-'
            mean = stats.loc[(task, modelclass, alg), (metric, 'mean')]
            std = stats.loc[(task, modelclass, alg), (metric, 'std')]
            if pd.isnull(std):
                if pd.notnull(mean):
                    return f"{mean:.2f}"
                else:
                    return '-'
            if pd.isnull(mean) or pd.isnull(std):
                return '-'
            return f"{mean:.2f}\\tiny{{$\\pm${std:.2f}}}"
        except KeyError:
            return '-'
    def create_table(tasks, algs, include_cega=True):
        num_cols = len(modelclasses) * len(algs)
        col_spec = "@{}r@{}@{}c@{}" + "|".join(["*{" + str(len(algs)) + "}{c}" for _ in modelclasses]) + "@{}"
        latex_table = f"""
                \\begin{{table}}[t]
                \\centering
                \\caption{{{"Binary Tasks" if include_cega else "Multi-Class Tasks"}}}
                \\footnotesize
                \\begin{{tabular}}{{{col_spec}}}
                & & """ + " & ".join([f"\\multicolumn{{{len(algs)}}}{{c|}}{{{mc.upper()}}}" for mc in
                                      modelclasses[
                                      :-1]]) + f"& \\multicolumn{{{len(algs)}}}{{c}}{{{modelclasses[-1].upper()}}}" + r"""\\
        \rotatebox{90}{\raisebox{1mm}{Task}} & Metric & """ + " & ".join([" & ".join(algs) for _ in modelclasses]) + r"""\\
        \midrule
        """
        for task in tasks:
            first_metric = True
            for metric, metric_name in zip(metrics, metric_names):
                task_cell = r'\multirow{7}{*}{\rotatebox{90}{\raisebox{1mm}{' + task + r'}}}' if first_metric else ''
                row = [task_cell, metric_name]
                for modelclass in modelclasses:
                    for alg in algs:
                        row.append(format_cell(task, modelclass, alg, metric))
                latex_table += " & ".join(row) + r" \\" + "\n"
                if first_metric:
                    first_metric = False
            latex_table += r"\midrule" + "\n" if task != tasks[-1] else r"\bottomrule" + "\n"
        latex_table += r"""
        \end{tabular}
        \end{table}
        """
        return latex_table
    algs = df['alg'].unique().tolist()
    cega_table = create_table([task for task in all_tasks if task in cega_applicable_tasks], include_cega=True, algs=algs)
    algs = [a for a in algs if a != 'cega']
    non_cega_table = create_table([task for task in all_tasks if task not in cega_applicable_tasks], include_cega=False, algs=algs)
    full_latex_output = cega_table + "\n" + non_cega_table
    print(full_latex_output)
    return full_latex_output


# ---

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

def comp_agreement(E_true, Y_true, E_pred, Y_pred) -> dict:
    # precision, recall, fpr, f1
    results = []

    for et, yt, ep, yp in zip(E_true, Y_true, E_pred, Y_pred):
        if yp == -1 or len(E_pred) == 0:
            continue
        if len(et) == 0:
            continue
        if yt != yp:
            continue
        et, ep = set(et), set(ep)
        r = set_metrics(et, ep)
        r = {'agreement_'+k: v for k, v in r.items()}
        results.append(r)

    results = pd.DataFrame(results).mean()
    return results.to_dict()

def dims_from_expl(E):
    # get all dims from explanation that have more than 0 importance
    # return dims in order from most to least important (ie sorted by score)
    _sorting = np.argsort(E, 1)[:, ::-1]
    results = []
    for e, s in zip(E, _sorting):
        r = [_si for _si in s if e[_si] > 0]
        results.append(r)
    return results

def dims_from_rules(rules: list):
    if rules is None:
        return np.empty(0)
    dims = []
    for r in rules:
        for t in r:
            dims.append(t[0])
    return np.unique(dims)



def update_cega_eval(r: RulesModel, data, labels, explanations):
    r = RulesModel(r.ohe_df, r.rules, None, r.pos_label, r.neg_label, prb_pos=r.prb_pos)

    # eval rules returns eval scores + rules applicable to prediction so we can compute completeness/ agreement here
    eval = r.eval_rules(data, labels)
    # get test data again
    _used_rules = eval['applied_rules']


def cega_redo_eval(calc_for_test=True):
    data_seed = 11880
    data_path = _variables.data_path
    for task in _variables.tasks:
        print(task)
        if task == 'beans': continue
        # load data
        meta_data = load_meta_data(Path(data_path, task))
        meta_data = meta_data[str(data_seed)]
        if calc_for_test:
            X, Y = meta_data['X_val'].numpy(), meta_data['Y_val'].numpy()
        else:
            X, Y = meta_data['X'].numpy(), meta_data['Y'].numpy()

        _pth = Path(data_path, task, 'explanations')
        cega_results = load_all_cega_results(task, filter=False)

        for i in tqdm(range(len(cega_results))):
            r = cega_results.iloc[i]

            # get explanations
            fname = f"{task}_{r['model_seed']}_{data_seed}_EPOCH-BATCH_{r['expl_method']}.pkl"
            fpath = Path(_pth, fname)
            e = load_pkl(fpath)
            if calc_for_test:
                e = e['val']

            e = e[r['expl_method']].numpy()
            expl_filtering = r['significance_threshold']
            try:
                # can be float 0.01 or string topk0.XX, latter raises ValueError
                expl_filtering = float(expl_filtering)
            except ValueError:
                pass

            E = __preprocess_explanations(e, expl_filtering)
            important_dims = dims_from_expl(E)
            r_updated = update_cega_eval(r, X, Y, E)

            _applied_rules = r_updated['applied_rules']
            Ep = []
            for a in _applied_rules:
                if len(a) == 0:
                    Ep.append([])
                else:
                    ad = [cega_rule_get_dim(aa) for aa in a]
                    Ep.append(ad)



def calc_attribution_agreement(calc_for_test=True):
    data_seed = 11880
    data_path = _variables.data_path
    for task in _variables.tasks:
        print(task)
        # load data
        meta_data = load_meta_data(Path(data_path, task))
        meta_data = meta_data[str(data_seed)]
        if calc_for_test:
            X, Y = meta_data['X_val'].numpy(), meta_data['Y_val'].numpy()
        else:
            X, Y = meta_data['X'].numpy(), meta_data['Y'].numpy()

        _results: pd.DataFrame = load_eval_results(task, filter_acc=False,
                                     filter_expl=None, filter_st=None,
                                     load_test=calc_for_test)
        _pth = Path(data_path, task, 'explanations')
        _results['agreement_accuracy'] = pd.NA
        _results['agreement_precision'] = pd.NA
        _results['agreement_recall'] = pd.NA
        _results['agreement_f1'] = pd.NA

        for i in tqdm(range(len(_results))):
            r = _results.iloc[i]

            # get explanations
            fname = f"{task}_{r['model_seed']}_{data_seed}_EPOCH-BATCH_{r['expl_method']}.pkl"
            fpath = Path(_pth, fname)
            e = load_pkl(fpath)
            if calc_for_test:
                e = e['val']

            e = e[r['expl_method']].numpy()
            expl_filtering = r['significance_threshold']
            try:
                # can be float 0.01 or string topk0.XX, latter raises ValueError
                expl_filtering = float(expl_filtering)
            except ValueError:
                pass

            E = __preprocess_explanations(e, expl_filtering)
            important_dims = dims_from_expl(E)

            dnf = r['dnf']
            dnf = lxg.models.DNFClassifier(dnf.rules, tie_break="first")

            rule_predictions = dnf.predict(X, explain=True)
            _labels, _rules = np.array([r[0] for r in rule_predictions]), [r[1] for r in rule_predictions]
            _dims = [dims_from_rules(r) for r in _rules]
            _agreement = comp_agreement(important_dims, Y, _dims, _labels)

            for k, v in _agreement.items():
                _results.iloc[i, _results.columns.get_loc(k)] = v

        # dump results again
        pth = './plots/cfire/'
        _pth = os.path.join(pth, task)
        fname = 'te_eval.pkl' if calc_for_test else 'tr_eval.pkl'
        fpath = Path(_pth, fname)
        dump_pkl(_results, fpath)



# ---


'''

CEGA results
    results = dict(
        rules_discr=rules_model,
        rules_chr=rules_model_chr,
        n_rules_discr=len(rules_model.rules),
        n_rules_chr=len(rules_model_chr.rules),
        discr_train=rules_train,
        discr_test=rules_test,
        chr_train=rules_train_chr,
        chr_test=rules_test_chr,
        eval_params=default_vals_from_code,
        description='eval results: (accuracy, recall weighted, recall amcro, precision macro, f1 macro, roc_auc)'
    )
    -> *_discr/ *_chr -> discriminative and characteristic rule models
    -> n_rules* -> n rules reported by model (this does not equal the number of literals though)
    -> *_train, *_test -> eval on train and test sets
    -> eval results contain are predictions, rules_rec_micro, rules_acc, rules_rec, rules_prec, rules_f1, rules_roc_auc
'''

task = None
cega_dir = './data/cfire/cega_rules'

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


def plt_cega_results(r: pd.DataFrame, pth, task, title_suffix):
    # Create boxplots grouped by the 'method' column
    targets = ['acc', 'precision', 'recall', 'n_rules']
    groupby_columns = ['expl_method', 'model_seed', 'rule type',
                       'apriori_max_len', 'asrm_threshold']

    r.drop(columns=['recall_micro', 'f1', 'roc_auc'], inplace=True)
    __create_dir(Path(pth))
    print(groupby_columns)
    nrows = int(np.ceil(len(targets)/2))+1
    for t in targets:
        plt.figure(figsize=(12, 6))
        plt.suptitle(f'{task} {title_suffix}')
        for i, param in enumerate(groupby_columns, 1):
            print(i, param)
            plt.subplot(nrows, 2, i)
            sns.boxplot(x=param, y=t, data=r)
            plt.title(f'{t} by {param}')
            # plt.xlabel(param)
            # plt.ylabel(str(t))
            plt.xticks(ticks=range(len(r[param].unique())), labels=r[param].unique())
        plt.tight_layout()
        plt.savefig(str(Path(pth, f'{task}-{t}.pdf')), format='pdf')
        # plt.show()

    # not so helpful
    # drop some columns for now
    dummy_encoded = pd.get_dummies(r)
    pd_correlation_matrix = dummy_encoded.corr()
    # Display correlation matrix
    print(pd_correlation_matrix)
    # Visualize correlation matrix using a heatmap without annotations
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(pd_correlation_matrix, cmap='coolwarm', annot=False)
    # Set x and y axis labels
    heatmap.set_xticklabels(pd_correlation_matrix.columns, rotation=45, horizontalalignment='right')
    heatmap.set_yticklabels(pd_correlation_matrix.index, rotation=0)
    plt.title(f'Correlation Matrix {task}')
    plt.tight_layout()
    plt.savefig(str(Path(pth, f'corr.pdf')), format='pdf');plt.close()


def cega_rule_get_dim(r):
    # r.itemset = "frozenset({'-0.243<V28<=1.328'})"
    # -> split('<') -> ["frozenset({'-0.243", 'V28', "=1.328'})"]
    d = str(r.itemset).split('<')[1]
    d = int(d[1:])
    return d

def load_all_cega_results(task, filter=True, modelclasses=['xgb', 'rf', 'svm', 'nn']):

    all_results = []
    for modelclass in modelclasses:
        _result_str_names = ['recall_micro', 'acc', 'recall', 'precision', 'f1', 'roc_auc']
        _pth_task = os.path.join(cega_dir, modelclass, task)
        if not os.path.exists(_pth_task):
            print(f"no CEGA results found for task {task}")
            return pd.DataFrame(columns=_result_str_names+['n_rules']+['rule_type'])
        files = os.listdir(_pth_task)
        results = [load_pkl(os.path.join(cega_dir, task, f)) for f in files]
        if len(results) == 0:
            return pd.DataFrame(columns=_result_str_names+['n_rules']+['rule_type'])
        args = [get_args_cega_results(f) for f in files]
        _chr, _discr = [], []
        for a, r in zip(args, results):
            if r['chr_test'] is not None:
                c = {k:v for v, k in zip(r['chr_test'][1:], _result_str_names)}
                c.update({'n_rules': r['n_rules_chr']})
            else:
                c = {k:None for k in _result_str_names}
                c.update({'n_rules': None})

            c.update(a)
            if (filter and r['chr_test'] is not None and
                    # throw all models away that don't have at least one rule per class
                    len(r['rules_chr'].rules['label'].unique()) < 2):
                pass
            else:
                _chr.append(c)

            if r['discr_test'] is not None:
                d = {k:v for v, k in zip(r['discr_test'][1:], _result_str_names)}
                d.update({'n_rules': r['n_rules_discr']})
            else:
                d = {k:None for k in _result_str_names}
                d.update({'n_rules': None})

            d.update(a)

            if (filter and r['discr_test'] is not None and
                    len(r['rules_discr'].rules['label'].unique()) < 2):
                pass
            else:
                _discr.append(d)


        _chr_df = pd.DataFrame(_chr)
        _discr_df = pd.DataFrame(_discr)
        _chr_df['rule type'] = 'chr'
        _discr_df['rule type'] = 'discr'

        merged = pd.concat([_chr_df, _discr_df])
        merged.reset_index(drop=True, inplace=True)

        if filter:  # finaly filter accuracy
            merged = merged[merged['acc'] > _random_baseline_acc[task]]

        merged['modelclass'] = modelclass

        all_results.append(merged)

    all_results_df = pd.concat(all_results, ignore_index=True)
    return all_results_df


def get_n_unique_literals(dnf):
    r = dnf.rules
    if r is None or len(r) == 0:
        return []
    literals = []
    for c_rule in r:
        for term in c_rule:
            for l in term:
                if l not in literals: literals.append(l)
    return len(literals)


def get_n_avg_literals(dnf):
    rules = dnf.rules
    lens = []
    for c in rules:
        for r in c:
            lens.append(len(r))
    if len(lens) == 0:
        return np.nan
    return np.mean(lens)


def load_anchors_results(task, filter_acc=True, load_test=True, modelclasses=['nn']):
    full_eval = []
    for modelclass in modelclasses:
        pth = './plots/cfire/'
        _pth = os.path.join(pth, modelclass, task)
        fname = 'anchors_te_eval.pkl' if load_test else 'anchors_tr_eval.pkl'
        _rnd = 'rnd_te' if load_test else 'rnd_tr'
        if not os.path.exists(os.path.join(_pth, fname)):
            continue
        eval = lxg.util.load_pkl(Path(_pth, fname))
        eval['n_rules'] = [d.n_rules for d in eval['dnf']]
        eval['n_literals'] = [d.n_literals for d in eval['dnf']]
        eval['n_unique_literals'] = [get_n_unique_literals(d) for d in eval['dnf']]
        eval['n_avg_literals'] = [get_n_avg_literals(d) for d in eval['dnf']]
        eval['time_short'] = [d._meta_information['time'] for d in eval['dnf']]
        eval['task'] = task

        eval['gr_rnd'] = eval['accuracy_no_rejected'] > eval[_rnd]
        eval['class_missing'] = [is_class_missing(d) for d in eval['dnf']]
        if filter_acc:
            eval = eval[eval['accuracy_no_rejected'] > eval[_rnd]]

        eval['modelclass'] = modelclass

        full_eval.append(eval)

    full_eval_df = pd.concat(full_eval, ignore_index=True)
    return full_eval_df

def is_class_missing(dnf):
    return sum([len(r) == 0 or (len(r) == 1 and len(r[0])==0) for r in dnf.rules])

def load_eval_results(task, filter_acc=True,
                      filter_expl=['ks', 'sg'],
                      filter_st=[0.01],
                      filter_gely=True,
                      filter_composition_strat=None,
                      load_test=True, modelclasses=['xgb', 'rf', 'svm', 'nn']):
    full_eval = []
    for modelclass in modelclasses:
        pth = './plots/cfire/'
        _pth = os.path.join(pth, modelclass, task)
        fname = 'te_eval.pkl' if load_test else 'tr_eval.pkl'
        _rnd = 'rnd_te' if load_test else 'rnd_tr'
        if not os.path.exists(os.path.join(_pth, fname)):
            print(f"did not find {fname} in {_pth}")
            continue
        eval = lxg.util.load_pkl(Path(_pth, fname))
        if 'time' not in eval.columns:
            # assert 'time' in eval.columns, (
            ValueError(f"column time not found for {modelclass} on {task}")
            continue
        [d.assert_no_infty() for d in eval['dnf']]
        eval['n_rules'] = [d.n_rules for d in eval['dnf']]
        eval['n_literals'] = [d.n_literals for d in eval['dnf']]
        eval['n_unique_literals'] = [get_n_unique_literals(d) for d in eval['dnf']]
        eval['n_avg_literals'] = [get_n_avg_literals(d) for d in eval['dnf']]  # avg num literals per rule
        eval['task'] = task
        eval['gr_rnd'] = eval['accuracy_no_rejected'] > eval[_rnd]
        eval['class_missing'] = [is_class_missing(d) for d in eval['dnf']]
        if filter_expl is not None and len(filter_expl) > 0:
            eval = eval[eval['expl_method'].isin(filter_expl)]
        if filter_st is not None and len(filter_st) > 0:
            eval = eval[eval['significance_threshold'].isin(filter_st)]
        if filter_gely is not None:
            eval = eval[eval['gely_sort_items'] == filter_gely]
        if filter_composition_strat is not None:
            eval = eval[eval['composition_strategy'].isin(filter_composition_strat)]
        if filter_acc:
            eval = eval[eval['accuracy_no_rejected'] > eval[_rnd]]

        eval['modelclass'] = modelclass

        full_eval.append(eval)

    if len(full_eval) == 0:
        return None

    full_eval_df = pd.concat(full_eval, ignore_index=True)
    return full_eval_df


def cega_n_avg_literals(rulemodel):
    rules = rulemodel.rules.itemset
    l = []
    for rule in rules:
        l.append(len(rule))
    if len(l) == 0:
        return np.nan
    return np.mean(l)

def load_new_cega_results(task, filter_acc=True,
                      filter_expl=['ks', 'ig', 'li'],
                      load_test=True, modelclasses=['xgb', 'rf', 'svm', 'nn']):

    all_eval = []
    for modelclass in modelclasses:
        pth = './plots/cfire/'
        _pth = os.path.join(pth, modelclass, task)
        fname = 'CEGA_te_eval.pkl' if load_test else 'CEGA_tr_eval.pkl'
        _rnd = 'rnd_te' if load_test else 'rnd_tr'
        if not os.path.exists(Path(_pth, fname)):
            all_eval.append(None)
            print(f"did not find {fname} in {_pth}")
            continue
        eval = lxg.util.load_pkl(Path(_pth, fname))
        if 'time' not in eval.columns:
            # assert 'time' in eval.columns, (
            print(f"column time not found for {modelclass} on {task}")
            continue
        eval['n_rules'] = [len(d.rules.itemset) if d is not None else np.nan for d in eval['rulemodel']]
        eval['n_literals'] = [np.sum([len(dd) for dd in d.rules.itemset]) if d is not None else np.nan for d in eval['rulemodel']]
        eval['n_avg_literals'] = [np.mean([len(dd) for dd in d.rules.itemset]) if d is not None else np.nan for d in eval['rulemodel']]
        eval['n_unique_literals'] = eval['n_literals']
        eval['task'] = task

        # only keep best performing cega model according to [metric]
        # eval = eval.loc[eval.groupby(['model_seed'])['accuracy_no_rejected'].idxmax()]]
        # eval = eval.loc[eval.groupby(['model_seed', 'expl_method'])['tr_accuracy_no_rejected'].idxmax()]
        eval['gr_rnd'] = eval['accuracy_no_rejected'] > eval[_rnd]
        eval['class_missing'] = [2-len(rm.rules['label'].unique()) for rm in eval['rulemodel']]
        if filter_acc:
            eval = eval[eval['accuracy_no_rejected'] > eval[_rnd]]
        if filter_expl is not None and len(filter_expl) > 0:
            eval = eval[eval['expl_method'].isin(filter_expl)]

        eval['modelclass'] = modelclass
        all_eval.append(eval)
    try:
        all_eval_df = pd.concat(all_eval, ignore_index=True)
    except ValueError:
        all_eval_df = None

    return all_eval_df

def print_table(df):
    grouped = df.groupby(['task', 'alg'])
    stats = grouped.agg({
        'accuracy_no_rejected': ['mean', 'std'],
        'f1': ['mean', 'std'],
        'coverage': ['mean', 'std'],
        'n_unique_literals': ['mean', 'std'],
        'completeness_accuracy': ['mean', 'std'],
        'completeness_f1': ['mean', 'std'],
        'model_seed': 'nunique'
    })

    # Reshape the data for easier LaTeX table generation
    tasks = df['task'].unique()
    tasks = _variables.tasks_plotting_order
    metrics = ['accuracy_no_rejected', 'f1', 'coverage', 'n_unique_literals']
    completeness_metrics = ['completeness_accuracy', 'completeness_f1']
    algs = ['us', 'DT', 'cega']

    def format_cell(task, alg, metric):
        try:
            mean = stats.loc[(task, alg), (metric, 'mean')]
            std = stats.loc[(task, alg), (metric, 'std')]
            return f"{mean:.2f}\\tiny{{$\\pm${std:.2f}}}"
        except KeyError:
            return '-'

    def generate_rows(metrics_list):
        rows = []
        for task in tasks:
            row = [task]
            for alg in algs:
                try:
                    n_models = stats.loc[(task, alg), ('model_seed', 'nunique')]
                    row.append(f"{n_models}")
                except KeyError:
                    row.append('-')
            for metric in metrics_list:
                for alg in algs:
                    row.append(format_cell(task, alg, metric))
            rows.append(" & ".join(row) + " \\\\")
        return "\n".join(rows)

    # Generate LaTeX tables
    latex_tables = r"""
        \begin{table*}[t]
        \centering
        \caption{Accuracy and Complexity Results}
        \footnotesize
        \begin{tabular}{l|ccc|ccc|ccc|ccc|ccc}
        \hline\hline
        & \multicolumn{3}{c|}{\# models} & \multicolumn{3}{c|}{Acc} & \multicolumn{3}{c|}{F1} & \multicolumn{3}{c|}{Cover} & \multicolumn{3}{c}{Complexity} \\
        Task & Ours & DT & CEGA & Ours & DT & CEGA & Ours & DT & CEGA & Ours & DT & CEGA & Ours & DT & CEGA \\
        \hline
        """

    latex_tables += generate_rows(metrics)

    latex_tables += r"""
        \hline\hline
        \end{tabular}
        \end{table*}

        \begin{table*}[t]
        \centering
        \caption{Completeness Results}
        \footnotesize
        \begin{tabular}{l|ccc|ccc|ccc}
        \hline\hline
        & \multicolumn{3}{c|}{\# models} & \multicolumn{3}{c|}{Acc} & \multicolumn{3}{c}{F1} \\
        Task & Ours & DT & CEGA & Ours & DT & CEGA & Ours & DT & CEGA \\
        \hline
        """

    latex_tables += generate_rows(completeness_metrics)

    latex_tables += r"""
        \hline\hline
        \end{tabular}
        \end{table*}
        """

    print(latex_tables)

def print_table_everything(df):
    cega_applicable_tasks = [
        "breastcancer", "ionosphere", 'btsc', 'spf', 'breastw', 'heloc', 'spambase'
    ]
    grouped = df.groupby(['task', 'modelclass', 'alg'])
    stats = grouped.agg({
        'accuracy_no_rejected': ['mean', 'std'],
        'f1': ['mean', 'std'],
        'coverage': ['mean', 'std'],
        'n_unique_literals': ['mean', 'std'],
        'completeness_accuracy': ['mean', 'std'],
        'completeness_f1': ['mean', 'std'],
        'model_seed': 'nunique'
    })
    metrics = ['model_seed', 'accuracy_no_rejected', 'f1', 'coverage', 'n_unique_literals', 'completeness_accuracy',
               'completeness_f1']
    metric_names = ['\#Models', 'Accuracy', 'F1', 'Coverage', 'Complexity', 'RC-Precision', 'RC-F1']
    all_tasks = _variables.tasks_plotting_order
    modelclasses = ['xgb', 'rf', 'svm', 'nn']

    def format_cell(task, modelclass, alg, metric):
        try:
            if metric == 'model_seed':
                value = stats.loc[(task, modelclass, alg), (metric, 'nunique')]
                return f"{value:.0f}" if pd.notnull(value) else '-'

            mean = stats.loc[(task, modelclass, alg), (metric, 'mean')]
            std = stats.loc[(task, modelclass, alg), (metric, 'std')]

            # Handle single-value case
            if pd.isnull(std):
                if pd.notnull(mean):
                    return f"{mean:.2f}"
                else:
                    return '-'

            if pd.isnull(mean) or pd.isnull(std):
                return '-'

            return f"{mean:.2f}\\tiny{{$\\pm${std:.2f}}}"
        except KeyError:
            return '-'

    def create_table(tasks, include_cega=True):

        algs = ['us', 'cega', 'DT'] if include_cega else ['us', 'DT']
        num_cols = len(modelclasses) * len(algs)
        col_spec = "@{}r@{}|@{}c@{}|" + "|".join(["*{" + str(len(algs)) + "}{c}" for _ in modelclasses]) + "@{}"
        latex_table = f"""
                \\begin{{table*}}[t]
                \\centering
                \\caption{{{"Results for Datasets with CEGA" if include_cega else "Results for Datasets without CEGA"}}}
                \\footnotesize
                \\begin{{tabular}}{{{col_spec}}}
                \\hline\\hline
                & & """+ " & ".join([f"\\multicolumn{{{len(algs)}}}{{c|}}{{{mc.upper()}}}" for mc in
                              modelclasses[:-1]]) + f"& \\multicolumn{{{len(algs)}}}{{c}}{{{modelclasses[-1].upper()}}}" + r"""\\
        \rotatebox{90}{Task} & Metric & """ + " & ".join([" & ".join(algs) for _ in modelclasses]) + r"""\\
        \hline
        """
        for task in tasks:
            first_metric = True
            for metric, metric_name in zip(metrics, metric_names):
                task_cell = r'\multirow{7}{*}{\rotatebox{90}{' + task + r'}}' if first_metric else ''
                row = [task_cell, metric_name]
                for modelclass in modelclasses:
                    for alg in algs:
                        row.append(format_cell(task, modelclass, alg, metric))
                latex_table += " & ".join(row) + r" \\" + "\n"
                if first_metric:
                    first_metric = False
                    # latex_table += r"\cline{2-" + str(num_cols + 2) + "}" + "\n"
            latex_table += r"\hline" + "\n" if task != tasks[-1] else r"\hline\hline" + "\n"
        latex_table += r"""
        \end{tabular}
        \end{table*}
        """
        return latex_table
    # Create two tables
    cega_table = create_table([task for task in all_tasks if task in cega_applicable_tasks], include_cega=True)
    non_cega_table = create_table([task for task in all_tasks if task not in cega_applicable_tasks], include_cega=False)
    # Combine both tables
    full_latex_output = cega_table + "\n" + non_cega_table
    print(full_latex_output)



def nanmean_arrays(arrays):
    # Compute mean of each array, ignoring NaNs, then compute mean of those means
    return np.nanmean([np.nanmean(arr) for arr in arrays if not np.isnan(arr).all()])


def get_pgi_means(df):
    # zero_agg = df['pgi_sampling'].apply(np.nanmean) TODO
    # First-level aggregation: Mean of arrays for each task and modelclass, ignoring NaNs
    first_agg = df.groupby(['task', 'modelclass', 'model_seed'])['pgi_masked'].agg('mean').reset_index()

    # Second-level aggregation: Mean across modelclasses for each task, ignoring NaNs
    second_agg = first_agg.groupby(['task', 'modelclass'])['pgi_masked'].agg(np.nanmean).reset_index()

    return second_agg

def cgpt_giant_scatter(df):
    agg_data = []

    for (task, expl_method), group in df.groupby(['task', 'expl_method']):
        agg_data.append({
            'task': task,
            'expl_method': expl_method,
            'f1_mean': group['f1'].mean(),
            'f1_std': group['f1'].std(),
            'f1_min': group['f1'].min(),
            'f1_max': group['f1'].max(),
            'pgi_mean': group['pgi'].mean(),
            'pgi_std': group['pgi'].std(),
            'pgi_min': group['pgi'].min(),
            'pgi_max': group['pgi'].max(),
            'n_unique_literals_mean': group['n_unique_literals'].mean(),
            'n_unique_literals_std': group['n_unique_literals'].std(),
            'n_unique_literals_min': group['n_unique_literals'].min(),
            'n_unique_literals_max': group['n_unique_literals'].max()
        })

    agg_df = pd.DataFrame(agg_data)

    # Print out the column names to ensure correctness
    print("Column names in agg_df:", agg_df.columns)

    # Define a palette and markers for the different explanation methods
    palette = sns.color_palette("tab20", n_colors=len(agg_df['expl_method'].unique()))
    markers = ['o', 's', 'D', 'X', 'P', '*', 'h', 'H', '+', 'x', '^', '<', '>', 'v', 'p', '|', '_']

    # Step 2: Plotting using FacetGrid with col_wrap
    def scatter_with_error_bars(x, y, **kwargs):
        data = kwargs.pop("data")
        ax = plt.gca()
        x_mean = data[f'{x}_mean'].to_numpy()
        y_mean = data[f'{y}_mean'].to_numpy()
        x_err = [x_mean - data[f'{x}_min'].to_numpy(), data[f'{x}_max'].to_numpy() - x_mean]
        y_err = [y_mean - data[f'{y}_min'].to_numpy(), data[f'{y}_max'].to_numpy() - y_mean]
        ax.errorbar(x_mean, y_mean, xerr=x_err, yerr=y_err, fmt='none', capsize=5, alpha=0.7)
        sns.scatterplot(x=f'{x}_mean', y=f'{y}_mean', hue='expl_method', style='expl_method',
                        palette=kwargs['palette'], markers=kwargs['markers'], data=data, ax=ax, s=100)

    plot_combinations = [
        ('f1', 'n_unique_literals'),
        ('f1', 'pgi'),
        ('pgi', 'n_unique_literals')
    ]

    for x, y in plot_combinations:
        g = sns.FacetGrid(agg_df, col='task', hue='expl_method', col_wrap=4, height=5, aspect=1, palette=palette)
        g.map_dataframe(scatter_with_error_bars, x=x, y=y, palette=palette,
                        markers=markers[:len(agg_df['expl_method'].unique())])
        g.add_legend()
        g.set_axis_labels(f'{x}_mean', f'{y}_mean')
        g.set_titles(col_template="{col_name}")
        plt.show()


# Define target metrics
target_metrics = ['f1', 'accuracy_no_rejected', 'coverage', 'n_rules',]# 'pgi']


# Function to aggregate data
def aggregate_data(df):
    # Separate 'DT' from others
    df_dt = df[df['expl_method'] == 'DT']
    df_others = df[df['expl_method'] != 'DT']

    # Aggregation for others
    task_aggregated_others = df_others.groupby(['task', 'model_seed', 'expl_method', 'significance_threshold'])[
        target_metrics].agg(['mean', 'std']).reset_index()
    task_aggregated_others.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in
                                      task_aggregated_others.columns]

    # Aggregation for 'DT'
    task_aggregated_dt = df_dt.groupby(['task', 'model_seed', 'expl_method'])[target_metrics].agg(
        ['mean', 'std']).reset_index()
    task_aggregated_dt.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in
                                  task_aggregated_dt.columns]

    # Aggregation across model seeds for each task
    task_model_aggregated_others = task_aggregated_others.groupby(
        ['task', 'expl_method', 'significance_threshold']).agg({
        f'{metric}_mean': ['mean', 'std'] for metric in target_metrics
    }).reset_index()
    task_model_aggregated_others.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in
                                            task_model_aggregated_others.columns]

    task_model_aggregated_dt = task_aggregated_dt.groupby(['task', 'expl_method']).agg({
        f'{metric}_mean': ['mean', 'std'] for metric in target_metrics
    }).reset_index()
    task_model_aggregated_dt.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in
                                        task_model_aggregated_dt.columns]

    # Combine both dataframes
    task_model_aggregated = pd.concat([task_model_aggregated_others, task_model_aggregated_dt], ignore_index=True)

    # Aggregation across all tasks
    global_aggregated = task_model_aggregated.groupby(['expl_method', 'significance_threshold'], dropna=False).agg({
        f'{metric}_mean_mean': ['mean', 'std'] for metric in target_metrics
    }).reset_index()
    global_aggregated.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in
                                 global_aggregated.columns]

    return task_aggregated_others, task_aggregated_dt, task_model_aggregated, global_aggregated


# Function to create box plots with FacetGrid
def create_facet_box_plots(df, target_metrics, prefix):
    for metric in target_metrics:
        metric_col = f'{metric}_{prefix}_mean'
        if metric_col in df.columns:
            g = sns.FacetGrid(df, col="task", col_wrap=4, sharey=False, height=4)
            g.map_dataframe(sns.boxplot, x='expl_method', y=metric_col, hue='significance_threshold', dodge=True)
            g.add_legend()
            g.set_axis_labels('expl_method', metric)
            g.set_titles(col_template="{col_name}")
            plt.show()
        else:
            print(f"Column {metric_col} not found in dataframe.")


# Function to create heatmaps
def create_heatmaps(df, target_metrics):
    tasks = df['task'].unique()
    for task in tasks:
        task_df = df[df['task'] == task]
        for metric in target_metrics:
            metric_col = f'{metric}_mean_mean'
            if metric_col in task_df.columns:
                # Fill missing significance_threshold for 'DT' with a placeholder
                task_df_filled = task_df.copy()
                task_df_filled['significance_threshold'] = task_df_filled['significance_threshold'].fillna('N/A')

                heatmap_data = task_df_filled.pivot_table(index='expl_method', columns='significance_threshold',
                                                          values=metric_col, dropna=False)
                plt.figure(figsize=(10, 8))
                sns.heatmap(heatmap_data, annot=True, cmap='coolwarm')
                plt.title(f'Heatmap of {metric} by expl_method and significance_threshold for task {task}')
                plt.show()
            else:
                print(f"Column {metric_col} not found in dataframe.")


# Function to create bar plots
def create_bar_plots(df, target_metrics, prefix):
    for metric in target_metrics:
        metric_col = f'{metric}_{prefix}_mean'
        if metric_col in df.columns:
            plt.figure(figsize=(12, 6))
            sns.barplot(x='expl_method', y=metric_col, hue='significance_threshold', data=df, dodge=True)
            plt.title(f'Bar plot of {metric} by expl_method and significance_threshold')
            plt.xticks(rotation=45)
            plt.show()
        else:
            print(f"Column {metric_col} not found in dataframe.")


# Function to create summary tables
def create_summary_tables(task_model_aggregated, global_aggregated, description):
    print(f"{description} Task-wise Summary:")
    print(tabulate(task_model_aggregated, headers='keys', tablefmt='psql'))

    print(f"\n{description} Global Summary:")
    print(tabulate(global_aggregated, headers='keys', tablefmt='psql'))


# Function to perform ANOVA
def perform_anova(df, target_metrics):
    for metric in target_metrics:
        metric_col = f'{metric}_mean'
        if metric_col in df.columns:
            formula = f'{metric_col} ~ C(expl_method) + C(significance_threshold) + C(expl_method):C(significance_threshold)'
            model = ols(formula, data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            print(f'ANOVA results for {metric}:')
            print(anova_table)
        else:
            print(f"Column {metric_col} not found in dataframe.")

def pprint_dnf(dnf_df, feature_names, class_names, mean, std):
    # assert len(dnf_df) == 1
    rules = dnf_df['dnf'].rules
    def _format_literal(l, prec=2):
        i = l[0]
        le, ri = l[1][0], l[1][1]
        le = (le*std[i])+mean[i]
        ri = (ri*std[i])+mean[i]
        # return f'({le:.{prec}f} < \\text{{{feature_names[i]}}} \\leq {ri:.{prec}f})'
        return f'\\text{{{feature_names[i]}}} \in ( {le:.2f}, {ri:.2f} ]'
    l_rules_str = []
    for i, c in enumerate(rules):
        l_term_strs = []
        for term in c:
            if term == [-1, (np.inf, np.inf)]:
                t_str = "[ --- ]"
            else:
                t_str = '(' + '\\land '.join([_format_literal(literal) for literal in term]) + ')'
            l_term_strs.append(t_str)
        class_str = f'$\\text{{{class_names[i]}}}: ' +  '\\lor '.join(l_term_strs) + '$\\\\'
        l_rules_str.append(class_str)
    final_str = '\n'.join(l_rules_str)
    final_str.replace('_', '\_')
    final_str = f"F1 = {dnf_df['f1']:.2f}, Precision = {dnf_df['precision']:.2f}\\\\" + final_str
    print(final_str)
    return final_str



# Function to perform the complete analysis
def analyze_experiment_results(df):
    task_aggregated_others, task_aggregated_dt, task_model_aggregated, global_aggregated = aggregate_data(df)

    # Individual Task Analysis Plots
    create_facet_box_plots(task_model_aggregated, target_metrics, 'mean')
    create_heatmaps(task_model_aggregated, target_metrics)

    # Global Analysis Plots
    create_bar_plots(global_aggregated, target_metrics, 'mean_mean')

    # Summary Tables
    create_summary_tables(task_model_aggregated, global_aggregated, "Aggregated")

    # ANOVA Analysis
    perform_anova(task_model_aggregated, target_metrics)



def load_explanation_times(task, modelclasses, expls=['ks', 'li', 'ig']):

    _all_exps = []
    for m in modelclasses:
        for e in expls:
            if e == 'ig' and m != 'nn':
                continue

            fname = f'{e}_times.pkl'
            _pth = Path('plots/cfire/', m, task)
            if os.path.isfile(Path(_pth, fname)):
                pass
            else:
                model_seeds = lxg.util.get_top_k_models(_variables.get_data_dir(task, modelclass=m), k=50)
                model_seeds = [m[0] for m in model_seeds]
                expls = [lxg.util.load_explanations(_variables.get_data_dir(task, m), s, e, m)[0][0] for s in model_seeds]
                _expls = []
                for s, ex in zip(model_seeds, expls):
                    # print(task, len(ex[e]))
                    _expls.append(dict(expl_method=e, task=task, model_seed=s, modelclass=m, time=ex['time']))
                _all_exps.extend(_expls)
    all_expls_df = pd.DataFrame(_all_exps)
    return all_expls_df





task_pprint_data = {
    'btsc': dict(
        feature_names=['Recency', 'Frequency', 'Amount', 'Time'],
        class_names=['0', '1'],
        mean=np.array([9.52406417, 5.59625668, 1399.06417112, 34.09090909]),
        std=np.array([7.52807285, 5.74363977, 1435.9099413, 23.906844]),
    ),
    'iris': dict(
        feature_names=['sepal length', 'sepal width', 'petal length', 'petal width'],
        class_names=['Setosa', 'Versicolour', 'Virginica'],
        mean=np.array([5.80133333, 3.08133333, 3.704, 1.188]),
        std=np.array([0.83274539, 0.46294516, 1.7558618, 0.7601684]),
    )
}


def get_example_rules(df, seeds=None, task='btsc', tolerance=0):

    t = task
    ex = [d for d in
                  df[(df['task'] == t) & (df['gr_rnd']) & (df['class_missing'] == False)].groupby(
                      'model_seed') if
                  len(d[1]) > 0]
    if len(ex) == 0:
        raise ValueError
        return []

    ex = [(e[0], ee) for e in ex for ee in e[1]['dnf']]
    ex = [(e[0], e[1].n_rules, e[1].rules) for e in ex]


    # return all seeds with explanations of observed minimal length
    _min_len = min(np.array([e[1] for e in ex]))
    chosen_seeds = [e for i, e in enumerate(ex) if e[1] <= (_min_len + tolerance)]
    print(chosen_seeds)
    return chosen_seeds


def time_plots(df):
    plt.clf()
    plt.close()
    df_pivot = df.pivot_table(index=['task', 'model_seed', 'modelclass'], columns='alg', values='time_final')

    # Calculate relative times for 'cega' and 'anchorsinf' compared to 'us-cover'
    df_pivot['cega_rel'] = df_pivot['cega'] / df_pivot['us-cover']
    df_pivot['anchorsinf_rel'] = df_pivot['\\anchorsinf'] / df_pivot['us-cover']

    # Box plot (log scale) for relative times
    plt.figure(figsize=(10, 6))

    # Create boxplot for 'cega_rel' and 'anchorsinf_rel'
    df_relative = df_pivot[['cega_rel', 'anchorsinf_rel']].melt(var_name='algorithm', value_name='relative_time')

    # Log scale box plot
    plt.yscale('log')
    box = df_relative.boxplot(by='algorithm', column='relative_time', grid=False)
    # plt.title("Relative Time (CEGA and Anchorsinf vs. US-Cover)")
    plt.suptitle('')  # Remove the default subtitle
    plt.ylabel("Relative Time (Log Scale)")
    plt.xlabel("Algorithm")

    # Colorblind-friendly color palette
    box.set_facecolor("lightblue")
    plt.savefig('./plots/cfire/time_plot_box', format='png')
    plt.show()
    plt.clf()
    plt.close()

    # Line plot with min/max and shaded area
    plt.figure(figsize=(10, 6))

    # Group data by 'task', 'modelclass', and 'alg' to calculate min/max
    df_minmax = df.groupby(['task', 'modelclass', 'alg'])['time_final'].agg(['min', 'max']).reset_index()

    # Plotting lines for each algorithm with shaded areas between min and max
    for alg in df['alg'].unique():
        alg_data = df_minmax[df_minmax['alg'] == alg]
        plt.plot(alg_data['task'].to_numpy(), alg_data['min'].to_numpy(), label=f'{alg} min', linestyle='--')
        plt.plot(alg_data['task'].to_numpy(), alg_data['max'].to_numpy(), label=f'{alg} max', linestyle='-')
        plt.fill_between(alg_data['task'].to_numpy(), alg_data['min'].to_numpy(), alg_data['max'].to_numpy(), alpha=0.2)

    # Add labels and colorblind-friendly legend
    plt.title("Min/Max Time with Shaded Area for Algorithms")
    plt.ylabel("Time Final")
    plt.xlabel("Task")
    plt.legend()
    plt.savefig('./plots/cfire/time_plot_lines', format='png')
    plt.show()
    plt.clf()
    plt.close()


def time_bar_plot(df, n_samples_datasets, font_size=14,
                  seconds_font_size=12, figsize=(18, 6),
                  xtickrotation=23):
    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",  # Use pdflatex for compilation
        "font.family": "serif",       # Use LaTeX default serif font
        "text.usetex": True,          # Enable LaTeX rendering in text
        "pgf.rcfonts": False,         # Disable Matplotlib-specific font setup, defer to LaTeX
    })

    # Set font size for different parts of the plot
    plt.rcParams.update({
        'axes.labelsize': font_size,
        'xtick.labelsize': font_size * 0.9,
        'ytick.labelsize': font_size * 0.9,
        'legend.fontsize': font_size * 0.9,
        'figure.titlesize': font_size * 1.2
    })

    # Pivot the dataframe for time_final values
    df_pivot = df.pivot_table(index=['task', 'model_seed', 'modelclass'], columns='alg', values='time_final')

    # Group data by 'task', 'alg' to calculate mean, min, and max
    df_minmax = df.groupby(['task', 'alg'])['time_final'].agg(['mean', 'min', 'max']).reset_index()

    # Set up colors for the algorithms
    colors = {
        'us-cover': '#0072B2',
        'cega': '#D55E00',
        '\\anchorsinf': '#009E73'
    }

    algorithms = ['us-cover', 'cega', '\\anchorsinf']

    # Bar positions for each task
    task_list = _variables.tasks_plotting_order
    num_tasks = len(task_list)
    bar_width = 0.27  # Width of each bar; 0.25 is text height

    # X positions for each task and algorithm group
    x_positions = np.arange(num_tasks)  # Center positions for tasks
    fig, ax = plt.subplots(figsize=figsize)#(figsize=(12, 6))  # Width can be adjusted depending on the number of tasks

    # Loop through each task and plot bars with whiskers for each algorithm
    for task_idx, task in enumerate(task_list):
        task_data = df_minmax[df_minmax['task'] == task]
        available_algorithms = task_data['alg'].tolist()

        # Get the mean time for 'us-cover' to normalize the other algorithms
        if 'us-cover' in available_algorithms:
            us_cover_data = task_data[task_data['alg'] == 'us-cover'].iloc[0]
            us_cover_mean = us_cover_data['mean']

        # Adjust bar positions dynamically based on the available algorithms
        if 'cega' not in available_algorithms:
            # If CEGA is missing, move 'us-cover' and '\anchorsinf' slightly closer together
            us_cover_position = x_positions[task_idx] - bar_width / 2  # Moved slightly left
            anchorsinf_position = x_positions[task_idx] + bar_width / 2  # Moved slightly right
        else:
            # If all algorithms are present, position them normally
            us_cover_position = x_positions[task_idx] - bar_width
            cega_position = x_positions[task_idx]
            anchorsinf_position = x_positions[task_idx] + bar_width

        # Helper function to adjust text placement dynamically
        def adjust_text_position(y_value):
            """ Adjust the y-offset dynamically based on the value of y (log scale). """
            if y_value < 10:
                return y_value + 0.1  # Small nudge for small values
            elif y_value < 100:
                return y_value + 1.0  # Slightly larger nudge for mid-range values
            else:
                return y_value + 10.0  # Larger nudge for large values

        # Plot 'us-cover' as baseline (value of 1, but show seconds above the bar)
        if 'us-cover' in available_algorithms:
            ax.bar(
                us_cover_position, 1, width=bar_width, color=colors['us-cover'], label=r'algname' if task_idx == 0 else ""
            )
            ax.text(
                us_cover_position, adjust_text_position(1), f'{us_cover_mean:.1f} sec', ha='center', va='bottom', rotation=90, fontsize=seconds_font_size
            )

        # Plot 'cega' relative to 'us-cover', display seconds above
        if 'cega' in available_algorithms:
            cega_data = task_data[task_data['alg'] == 'cega'].iloc[0]
            cega_mean = cega_data['mean'] / us_cover_mean  # Relative to 'us-cover'
            cega_min = cega_data['min'] / us_cover_mean
            cega_max = cega_data['max'] / us_cover_mean
            ax.bar(
                cega_position, cega_mean, width=bar_width, color=colors['cega'],
                yerr=[[cega_mean - cega_min], [cega_max - cega_mean]], capsize=5, label='cega' if task_idx == 0 else ""
            )
            # Display seconds above the whiskers, with dynamic placement
            ax.text(
                cega_position, adjust_text_position(cega_max), f'{cega_data["mean"]:.1f} sec', ha='center', va='bottom', rotation=90, fontsize=seconds_font_size
            )

        # Plot '\anchorsinf' relative to 'us-cover', display seconds above
        if '\\anchorsinf' in available_algorithms:
            anchorsinf_data = task_data[task_data['alg'] == '\\anchorsinf'].iloc[0]
            anchorsinf_mean = anchorsinf_data['mean'] / us_cover_mean  # Relative to 'us-cover'
            anchorsinf_min = anchorsinf_data['min'] / us_cover_mean
            anchorsinf_max = anchorsinf_data['max'] / us_cover_mean
            ax.bar(
                anchorsinf_position, anchorsinf_mean, width=bar_width, color=colors['\\anchorsinf'],
                yerr=[[anchorsinf_mean - anchorsinf_min], [anchorsinf_max - anchorsinf_mean]], capsize=5, label='anchorsinf' if task_idx == 0 else ""
            )
            if task_idx in [8, 9, 12]:
                rotation = 0
                ha = 'center'
            else:
                rotation = 90
                ha = 'center'
            # Display seconds above the whiskers, with dynamic placement
            ax.text(
                anchorsinf_position, adjust_text_position(anchorsinf_max), f'{anchorsinf_data["mean"]:.1f} sec',
                ha=ha, va='bottom', rotation=rotation, fontsize=seconds_font_size
            )

    # Set x-ticks to the task names + sample sizes
    task_names_with_samples = [f"{task} ({n_samples_datasets[task]})" for task in task_list]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(task_names_with_samples, rotation=xtickrotation, ha="right", rotation_mode='anchor')

    # Add labels and title
    ax.set_xlabel("Task (Number of Samples)")
    ax.set_ylabel(r"Relative Time Final (Log Scale)")
    ax.set_yscale('log')  # Log scale for y-axis
    # ax.set_title("Relative Runtime per Algorithm (vs. US-Cover)\n with Min/Max Whiskers")

    # Remove top and right spines for a cleaner look
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Add legend
    ax.legend(loc="upper left")

    # Adjust layout to fit everything
    plt.tight_layout()

    # Save the plot as PGF
    plt.savefig('./plots/cfire/time_plot_bars.pgf', format='pgf')
    plt.show()
    print("done")

def full_eval():

    n_samples_datasets = {
    'btsc': 150,
    'spf': 389,
    'breastw': 1000,
    'spambase': 921,
    'heloc': 1975,
    "breastcancer": 114,
    "ionosphere": 71,
    'beans': 1050,
    'wine': 36,
    'diggle': 124,
    'abalone': 627,
    'iris': 38,
    'vehicle': 170,
    'autouniv': 140
    }
    try:
        times_df = pd.read_csv('times_processed.csv')
    except FileNotFoundError:
        pass
    # n_samples_df = pd.DataFrame(n_samples_datasets)
    # time_bar_plot(times_df, n_samples_datasets, figsize=(12,5), xtickrotation=17, font_size=16)
    __tasks = _variables.tasks_plotting_order
    _expls = ['ks', 'li', 'ig', 'ds']#, 'ksub', 'li', 'ksub'] # , 'ks']
    _modelclasses = ['nn']

    expl_times = [load_explanation_times(task, _modelclasses) for task in __tasks]
    expl_times_df = pd.concat(expl_times, axis=0, ignore_index=True)

    anchors_results = [load_anchors_results(task, filter_acc=False, modelclasses=_modelclasses) for task in __tasks]
    anchors_df = pd.concat(anchors_results)
    anchors_df['class_missing'] = anchors_df['class_missing'] > 0

    anchors_df.rename(columns={'time': 'time_final'}, inplace=True)
    anchors_cover_df = anchors_df[anchors_df['mode'] == 'cover'].copy()
    anchors_cover_df['alg'] = '\\anchorsk'
    anchors_prec_df = anchors_df[anchors_df['mode'] == 'precision'].copy()
    anchors_prec_df['alg'] = '\\anchorsinf'
    not_all_classes = lambda df: sum([any([len(r) == 0 for r in dnf.rules]) for dnf in df['dnf']])
    rules_pkls = [load_eval_results(task, filter_expl=_expls, filter_st=[0.01],
                                    filter_composition_strat=None, filter_acc=False,
                                    modelclasses=_modelclasses
                                    ) for task in __tasks]
    print(f"anchors cover not all classes: {not_all_classes(anchors_cover_df)}")
    print(f"anchors prec  not all classes: {not_all_classes(anchors_prec_df)}")

    rules_df = pd.concat(rules_pkls, ignore_index=True)
    rules_df_grouped_time = rules_df.groupby(['task', 'modelclass', 'model_seed', 'time'])
    grouped_time = expl_times_df.groupby(['task', 'modelclass', 'model_seed'])['time'].sum()
    expl_times_df['accumulated_time'] = expl_times_df.set_index(['task', 'modelclass', 'model_seed']).index.map(grouped_time)
    # now we alco need to accumulate time for
    m = pd.merge(rules_df, expl_times_df, on=['model_seed', 'modelclass', 'task', 'expl_method'], how='left')#.iloc[0]
    rules_df['time_short'] = rules_df['time'] + m['time_y']
    rules_df.rename(columns={'time': 'time_short'}, inplace=True)
    rules_df['time_long'] = rules_df['time'] + m['accumulated_time']  # accumulate time of all expls per model, add to rules_df['time']
    flattened = lambda l: [item for sublist in l for item in
                           (flattened(sublist) if isinstance(sublist, list) else [sublist])]
    dnfs = [DNFClassifier(d.rules) for d in rules_df['dnf']]
    rules_df = rules_df[rules_df['expl_method'] != 'cm']

    # rules_df['expl_method'].unique(), rules_df.groupby(['expl_method', 'pgi_sampling'])['expl_method']
    rules_df = rules_df[rules_df['alg']=='ours']
    rules_df['n_rules'] = [d.n_rules for d in rules_df['dnf']]
    rules_df['class_missing'] = rules_df['class_missing'] > 0

    _cover = rules_df[rules_df['composition_strategy'] == 'cover'].copy().reset_index(drop=True)
    _cover_grouped_times = _cover.groupby(['task', 'modelclass', 'model_seed'])['time'].sum()
    _cover['time_fcis_accumulated'] = _cover.set_index(['task', 'modelclass', 'model_seed']).index.map(_cover_grouped_times)
    m = pd.merge(_cover, expl_times_df, on=['model_seed', 'modelclass', 'task', 'expl_method'], how='left')
    _cover['time_final'] = _cover['time_fcis_accumulated'] + m['accumulated_time']
    _cover['time_rel'] = _cover['time_final'] / _cover['time_final']  # yes this is 1.

    print("GET EXMAMLPES FOR DISAGREEMENT LISTING - IG VS LI ON BTSC AND IRIS")
    ig_df = _cover[_cover['gr_rnd'] & (_cover['expl_method'].isin(['ig']))].copy()
    # seeds_ig = get_example_rules(ig_df)
    li_df = _cover[_cover['gr_rnd'] & (_cover['expl_method'].isin(['li']))].copy()
    # seeds_li = get_example_rules(li_df)
    ks_df = _cover[_cover['gr_rnd'] & (_cover['expl_method'].isin(['ks']))].copy()
    # seeds_ks = get_example_rules(ks_df)
    _intersect_seeds = {}
    for t in ['btsc', 'iris', 'brestcancer']:
        got_seeds = False
        tolerance = 0
        while not got_seeds:
            print(f"tol=", tolerance)
            seeds_ig = get_example_rules(ig_df, task=t, tolerance=tolerance)
            seeds_li = get_example_rules(li_df, task=t, tolerance=tolerance)
            seeds_ks = get_example_rules(ks_df, task=t, tolerance=tolerance)
            _intersect_seeds[t] = set([s[0] for s in seeds_ig]).intersection(
                set([s[0] for s in seeds_li]).intersection(set([s[0] for s in seeds_ks]))
            )
            if len(_intersect_seeds[t]) > 0: got_seeds = True
            else: tolerance += 1
    print(_intersect_seeds)

    # accs_ = ['accuarcy', 'tr_accuracy', 'coverage', 'f1']
    # _chosen_models = get_example_rules(_cover)
    # print("ANCHORS")
    # get_example_rules(anchors_df, _chosen_models)

    _cover['alg'] = 'us-cover'

    def stats_for_expl_methods(df, grpby=['task', 'model_seed', 'modelclass'], ):
        targets = ['f1', 'accuracy_no_rejected', 'coverage', 'n_rules']#, 'pgi']
        # targets = ['tr_f1', 'tr_accuracy_no_rejected', 'tr_coverage', 'tr_n_unique_literals', 'tr_pgi']
        results = []
        for t in targets:
            if t in ['n_unique_literals']:  # lower is better
                selected = df.loc[df.groupby(grpby)[t].idxmin()].copy()
            else:
                selected = df.loc[df.groupby(grpby)[t].idxmax()].copy()
            selected['target'] = t
            results.append(selected)
        # Combine all selected results
        results_df = pd.concat(results, axis=0, ignore_index=True)
        results_df = results_df[grpby + ['target'] + ['expl_method']]

        # Step 1: Check if the same expl_method is used across all targets for each combination of [task, model_seed, modelclass]
        consistent_method_df = results_df.groupby(grpby)['expl_method'].nunique().reset_index()

        # Step 2: Find the consistent combinations (where only one unique expl_method was selected for all targets)
        consistent_groups = consistent_method_df[consistent_method_df['expl_method'] == 1][grpby]

        # Step 3: Filter results_df to keep only the consistent cases
        consistent_results = results_df.merge(consistent_groups, on=grpby)

        # Step 4: Count the distribution of expl_method among consistent cases
        expl_method_distribution = consistent_results.groupby(grpby).first()['expl_method'].value_counts()

        consistent_count = len(consistent_groups)  # This should still be 111
        # Step 5: Count how often an expl_method is not selected for any target for each group
        missing_methods_df = results_df.groupby(grpby)['expl_method'].apply(
            lambda x: set(['ig', 'ks', 'li']) - set(x)).reset_index()
        missing_methods_count = missing_methods_df['expl_method'].apply(len).value_counts()

        method_presence = results_df.groupby('task')['expl_method'].apply(
            lambda x: set(['ig', 'ks', 'li']) - set(x)).reset_index()
        method_presence = method_presence[
            method_presence['expl_method'].apply(len) > 0]  # Only keep tasks where some method is missing

        return results_df, consistent_count, expl_method_distribution, missing_methods_count, method_presence

    def plot_disagreement(df):
        fontbase = 25
        labelsize = fontbase - 2
        fontsize = fontbase
        plt.rcParams.update({
            "pgf.texsystem": "pdflatex",  # Change this if you use xelatex or lualatex
            "text.usetex": True,  # Use LaTeX for text
            "font.family": "serif",  # Use a serif font (LaTeX's default)
            "font.serif": [],  # Use the default LaTeX serif font
            "axes.labelsize": labelsize,  # Set the label font size (increased)
            "font.size": labelsize,  # General font size (increased)
            "legend.fontsize": fontsize,  # Font size of the legend (increased)
            "xtick.labelsize": fontsize,  # Font size of the x-axis tick labels (increased)
            "ytick.labelsize": fontsize,  # Font size of the y-axis tick labels (increased)
            "pgf.preamble": [
                r"\usepackage{amsmath}",  # Use the AMS Math package for LaTeX
            ]
        })

        print(df.columns.values)  # gives: ['task', 'model_seed', 'modelclass', 'target', 'expl_method']
        x_labels = {'f1': 'F1', 'accuracy_no_rejected': 'Precision', 'coverage': 'Coverage',
                    'n_rules': '\#Rules'}#, 'pgi': 'PGI-K'}
        exps = {'ig': 'IG', 'li': 'LIME', 'ks': 'KSHAP', 'ds':'DSURF'}

        # Replace values in 'target' and 'expl_method'
        df['target'] = df['target'].replace(x_labels)
        df['expl_method'] = df['expl_method'].replace(exps)
        custom_order = [v for _, v in x_labels.items()]
        df['target'] = pd.Categorical(df['target'], categories=custom_order, ordered=True)
        # Step 1: Group by 'target' and 'expl_method', and calculate the count
        grouped_counts = df.groupby(['target', 'expl_method']).size().unstack(fill_value=0)

        # Step 2: Normalize the counts to get percentages
        grouped_counts_normalized = grouped_counts.div(grouped_counts.sum(axis=1), axis=0)

        # Step 3: Create a colorblind-friendly colormap (using only 3 colors)
        cmap = sns.color_palette('colorblind')
        # 0 blue, 1 orange, 2 green, 3 red, 4 purple, 5 brown, 6 pink, 7 grey, 8 yellow, 9 light blue
        cmap = [cmap[0], cmap[7], cmap[2]]  # Use only the first 3 colors

        # Step 4: Create a stacked bar plot with adjusted bar width (make bars closer together)
        ax = grouped_counts_normalized.plot(kind='bar', stacked=True, figsize=(10, 6), color=cmap,
                                            width=0.75)  # Increased bar width

        # Step 5: Customize the plot (title, labels, and remove bounding box)
        plt.title(r'Normalized Stacked Bar Plot of Attribution Methods',
                  fontsize=fontbase+3)  # Larger title font
        plt.xlabel(r'', fontsize=0)
        plt.ylabel(r'Proportion of Attribution Methods', fontsize=fontbase)
        plt.xticks(rotation=0)  # Ensure x-axis labels are horizontal

        # Adjust x-axis tick labels to reduce space between bars
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center')  # Center labels

        # Adjust legend to be below the plot, in a single row
        plt.legend(title=r'Attribution\\ Method', ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.1),
                   fontsize=fontbase)  # Legend in single row below x-axis

        # Slim bars and adjust layout
        plt.tight_layout()

        # Remove top, right, and left spines (bounding box)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # Keep only the x-axis line
        ax.spines['bottom'].set_linewidth(1)  # Ensure the x-axis line is visible

        # Step 6: Save the plot as .pgf file with LaTeX fonts
        plt.savefig('./plots/cfire/disagreement_bars.pgf', format='pgf')

        print("done")
        # plt.show()

    def plot_disagreement_transposed(df, fontbase=None):
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns

        fontbase = 20 if fontbase is None else fontbase
        labelsize = fontbase - 2
        fontsize = fontbase
        plt.rcParams.update({
            "pgf.texsystem": "pdflatex",  # Change if you use xelatex or lualatex
            "text.usetex": True,  # Use LaTeX for text
            "font.family": "serif",  # Use a serif font (LaTeX's default)
            "font.serif": [],  # Use the default LaTeX serif font
            "axes.labelsize": labelsize,
            "font.size": labelsize,
            "legend.fontsize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "pgf.preamble": [
                r"\usepackage{amsmath}",
            ]
        })

        # Update categorical labels
        x_labels = {'f1': 'F1', 'accuracy_no_rejected': 'Precision',
                    'coverage': 'Coverage', 'n_rules': r'\#Rules'}
        exps = {'ig': 'IG', 'li': 'LIME', 'ks': 'KSHAP', 'ds': 'DSURF'}

        df['target'] = df['target'].replace(x_labels)
        df['expl_method'] = df['expl_method'].replace(exps)
        custom_order = [v for _, v in x_labels.items()]
        df['target'] = pd.Categorical(df['target'], categories=custom_order, ordered=True)

        # Group and normalize counts
        grouped_counts = df.groupby(['target', 'expl_method']).size().unstack(fill_value=0)
        grouped_counts_normalized = grouped_counts.div(grouped_counts.sum(axis=1), axis=0)

        # Use only 3 colors from a colorblind-friendly palette
        cmap = sns.color_palette('colorblind')
        cmap = [cmap[0], cmap[7], cmap[2]]

        # Create a horizontal stacked bar plot without the 'height' parameter.
        ax = grouped_counts_normalized.plot(kind='barh', stacked=True, figsize=(18, 4),
                                            color=cmap)

        plt.title(r'Normalized Stacked Bar Plot of Attribution Methods', fontsize=fontbase+3)
        plt.xlabel(r'Proportion of Attribution Methods', fontsize=fontbase)
        plt.ylabel(r'')
        plt.xlim(0, 1)

        # Adjust y-axis tick labels to be centered
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=fontbase+3, rotation=0, va='center')

        # Adjust the legend to appear below the plot in a single row
        plt.legend(title=r'Attribution\\ Method', loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=fontbase)

        plt.tight_layout()

        # Remove unnecessary spines; keep only the x-axis line
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1)

        # Adjust the thickness of each horizontal bar by setting the patch height
        for patch in ax.patches:
            patch.set_height(0.75)

        plt.show()

        plt.savefig('./plots/cfire/disagreement_bars.pgf', format='pgf')
        print("done")

    disagreement_effect, agree_count, expl_method_distribution, missing_methods_count, method_presence \
        = stats_for_expl_methods(_cover[_cover['gr_rnd'] & (_cover['class_missing'] == False)])

    print(f"n models remain: ", len(_cover[_cover['gr_rnd'] & (_cover['class_missing'] == False)][['task', 'model_seed']].drop_duplicates()))
    print(f"for {agree_count} models a single explanation method led to best results in all target measures")
    print(f"{expl_method_distribution/agree_count}")
    print(f"missing methods:{missing_methods_count}")
    print(f"method present:{method_presence}")
    plot_disagreement_transposed(disagreement_effect, fontbase=30)


    # plot_disagreement_errorbars(disagreement_effect)
    # _acc = rules_df[(rules_df['composition_strategy'] == 'score_cover') & (rules_df['acc_weight'] == 1.)].copy()
    # _acc['alg'] = 'us-acc'
    # _rc = rules_df[(rules_df['composition_strategy'] == 'score_cover') & (rules_df['cs_weight'] == 1.)].copy()
    # _rc['alg'] = 'us-rc'


    cols_to_keep = ['f1', 'accuracy_no_rejected', 'coverage', 'n_unique_literals', #'pgi',
                    'task',
                    'alg', 'model_seed', 'modelclass', 'suff', 'time_short', 'time_long', 'le_rnd', 'class_missing',
                    'n_rules', 'empirical_avg_n_rules', 'n_avg_literals', 'fail'
                    # 'significance_threshold',  'expl_method',
                    ]
    disag_df = _cover.copy()
    disag_df.loc[disag_df['expl_method'] == 'ks', 'alg'] = '\\algname-KS'
    disag_df.loc[disag_df['expl_method'] == 'li', 'alg'] = '\\algname-LI'
    disag_df.loc[disag_df['expl_method'] == 'ig', 'alg'] = '\\algname-IG'
    # disag_df.loc[disag_df['expl_method'] == 'ds', 'alg'] = '\\algname-DS'

    filter_expl = lambda df: df.loc[df.groupby(['task', 'model_seed', 'modelclass'])['tr_accuracy_no_rejected'].idxmax()]
    # _rc = filter_expl(_rc)
    # _acc = filter_expl(_acc)
    _cover = filter_expl(_cover)
    _cover_disag = _cover.copy()
    with open('./plots/cfire/nn/chosen_models.csv', 'w') as _file_chosen_models:
        _str_chosen_models = _cover[['model_seed', 'task', 'expl_method']].to_csv()
        _file_chosen_models.write(_str_chosen_models)

    # np.savetxt('./chosen_models.csv', )
    _cover_disag['alg'] = '\\algname'
    disag_df = pd.concat([_cover, disag_df], ignore_index=True)
    disag_df['fail'] = ~disag_df['gr_rnd'] | disag_df['class_missing']
    disag_df['le_rnd'] = ~disag_df['gr_rnd']
    print("DISAGREEMENT TABLE COMPACT")
    table_str = print_table_dnf_cega_anchors(disag_df[cols_to_keep])
    print(table_str)
    print("DISAGREEMENT TABLE COMPLETE")
    table_long = print_table_dnf_cega_anchors(disag_df[cols_to_keep], compact=False)
    print(table_long)
    print("")

    _n_rules_cols_to_save = ['task', 'model_seed', 'expl_method', 'modelclass', 'n_rules']
    _cover[_n_rules_cols_to_save].to_csv('n_rules_dnfclassifiers.csv', index=False)
    print(f"us cover     not all classes: {not_all_classes(_cover)}")


    cega_pkls = [load_new_cega_results(task, filter_acc=False, filter_expl=_expls, modelclasses=_modelclasses) for task in __tasks]
    cega_pkls = [c for c in cega_pkls if c is not None]
    cega_df = pd.concat(cega_pkls, ignore_index=True)
    mcega = pd.merge(cega_df, expl_times_df, on=['model_seed', 'modelclass', 'task', 'expl_method'], how='left')
    cega_df['time_long'] = mcega['time_y'] + cega_df['time']
    cega_df['time_short'] = cega_df['time']
    cega_df['alg'] = 'cega'


    _cega_grouped_times = cega_df.groupby(['task', 'modelclass', 'model_seed'])['time'].sum()
    cega_df['time_fcis_accumulated'] = cega_df.set_index(['task', 'modelclass', 'model_seed']).index.map(_cega_grouped_times)
    m = pd.merge(cega_df, expl_times_df, on=['model_seed', 'modelclass', 'task', 'expl_method'], how='left')
    cega_df['time_final'] = cega_df['time_fcis_accumulated'] + m['accumulated_time']
    cega_df['class_missing'] = cega_df['class_missing'] > 0

    _cega = cega_df.loc[cega_df.groupby(['task', 'model_seed', 'modelclass'])['tr_accuracy_no_rejected'].idxmax()]


    def convert_rulemodel(r):

        def frozen_set_to_rule(f):
            _r = []
            for s in f:
                s = s.split('<')
                if len(s) == 2: # a<X or X<a
                    left, right = s
                    if left.startswith('V'): # right is '=XXX'
                        _r.append((int(left[1:])-1, (-np.inf, float(right[1:]))))
                    else:
                        _r.append((int(right[1:])-1, (float(left), np.inf)))
                else:
                    assert len(s) == 3
                    v = s[1][1:]
                    left, right = s[0], s[2][1:]
                    _r.append( (int(v)-1, (float(left), float(right))) )
            return _r
        converted_rules = [[], []]
        for _, _row in r.rules[['label', 'itemset']].iterrows():
            l, i = _row.label, _row.itemset
            converted = frozen_set_to_rule(i)
            converted_rules[int(l)].append(converted)
        if len(converted_rules[0]) == 0:
            converted_rules[0] = [(-1, (np.inf, np.inf))]
        if len(converted_rules[1]) == 0:
            converted_rules[1] = [(-1, (np.inf, np.inf))]
        return DNFClassifier(converted_rules)

    def cega_frame_to_example_input(df):
        # copy over necessary columns and transform cega rules to our rules
        df = df.reset_index(drop=True)
        new_df = df[['task', 'model_seed']]
        rules = [convert_rulemodel(rm) for rm in df['rulemodel']]
        new_df['dnf'] = rules
        return new_df

    def print_rules(cfire, anch, ceg, task):
        for seed in cfire['model_seed']:
            cf_dnf = cfire[cfire['model_seed'] == seed].iloc[0]
            an_dnf = anch[anch['model_seed'] == seed].iloc[0]
            print(f"\n\n### SEED {seed} ###")
            print("\nCFIRE")
            pprint_dnf(cf_dnf, **task_pprint_data[task])
            print("\nANCHORS")
            pprint_dnf(an_dnf, **task_pprint_data[task])
            if task != 'iris':
                print("\nCEGA")
                ceg_rm = ceg[ceg['model_seed'] == seed].iloc[0]
                pprint_dnf(ceg_rm, **task_pprint_data[task])
            print(f"\n\n")

    for t, iseeds in _intersect_seeds.items():
        # pick all models from Anchors and CEGA that are chosen for that seed
        _c = _cega[_cega['model_seed'].isin(iseeds) & _cega['task'].isin([t])].copy().sort_values(by='model_seed').reset_index(drop=True)
        _c['dnf'] = _c['tr_rulemodel'].apply(convert_rulemodel)
        _c = _c[['model_seed', 'f1', 'precision', 'dnf']]
        _a = anchors_prec_df[anchors_prec_df['model_seed'].isin(iseeds) & anchors_prec_df['task'].isin([t])].copy().sort_values(by='model_seed').reset_index(drop=True)
        _a = _a[['model_seed','f1', 'precision', 'dnf']]
        _r = _cover[_cover['model_seed'].isin(iseeds) & _cover['task'].isin([t])].copy().sort_values(by='model_seed').reset_index(drop=True)
        _r = _r[['model_seed','f1', 'precision', 'dnf']]
        print(f"TASK - {t}\n"); print_rules(_r, _a, _c, t)

    cols_to_keep = ['f1', 'accuracy_no_rejected', 'coverage', 'n_unique_literals',
                    'task',
                    'alg', 'model_seed', 'modelclass', 'suff', 'time_short', 'time_long', 'le_rnd', 'class_missing',
                    'n_rules', 'empirical_avg_n_rules', 'n_avg_literals', 'fail'
                    ]

    merged = pd.concat([_cover,
                        anchors_cover_df, anchors_prec_df,
                        _cega], axis=0, ignore_index=True)
    merged['fail'] = (~merged['gr_rnd']) | merged['class_missing']
    merged['le_rnd'] = ~merged['gr_rnd']  #
    _filtered_merged = merged[cols_to_keep].copy()

    def save_table_to_file(latex_table, filename="output_table.txt"):
        with open(filename, "w") as file:
            file.write(latex_table)

    print("TABLE ALGO COMPARISON COMPACT")
    table_str = print_table_dnf_cega_anchors(_filtered_merged)
    print(table_str)
    print("TABLE ALGO COMPARISON COMPLETE")
    table_str = print_table_dnf_cega_anchors(_filtered_merged, compact=False)
    print(table_str)

    _filtered_merged.to_csv('filtered_dataframe.csv', index=False)
    analyze_experiment_results(merged)

    marker='o'
    marker_size = 18
    width = 16
    height = 12
    aspect = width / height
    plt.figure(dpi=500)
    acc_dim, cx_dim, cm_dim = 'f1', 'n_unique_literals', 'pgi'
    hue = "expl_method" # "alg"
    g = sns.FacetGrid(merged, col="task", hue=hue, col_wrap=4, height=3, sharex=False, sharey=False, aspect=aspect)
    g.map(sns.scatterplot, cx_dim, acc_dim, alpha=0.4, marker=marker, s=marker_size)
    g.set_axis_labels("Complexity", "Accuracy")
    g.set_titles(col_template="{col_name}")

    for ax in g.axes.flatten():
        # _y = _random_baseline_acc[ax.get_title()]
        _task = ax.get_title()
        # ax.set_ylim(_y-.05, 1)
        # ax.set_ylim(merged[merged['task']==_task]['accuracy'].min()-0.01, 1)
        ax.set_ylim(0., 1)
        # if ax.get_title() in ['beans']:
        # else:
        #     ax.set_ylim(0.5, 1)


        # if ax.get_title() in ['heloc', 'spf', 'spambase', 'beans', 'breastcancer']:
        #     ax.set_xscale('log')

        ax.set_xlabel('N unique literals')
        ax.set_ylabel('Accuracy')
        print(ax.title)
        # ax.axhline(y=_y, color='black', linestyle='--', label="random baseline")
    g.fig.subplots_adjust(hspace=0.6)
    g.add_legend()
    # plt.tight_layout()
    g.fig.suptitle(f'Accuracy vs Complexity | expls: {_expls}')
    # plt.savefig(str(Path('./plots/cfire/summaries/', f'scatter_acc_nrules.pdf')), format='pdf');plt.close()
    plt.show()
    plt.close()


_random_baseline_acc ={
    'heloc': 0.54,
    "ionosphere": 0.74,
    'breastw': 0.65,
    "breastcancer": 0.65,
    'spf': 0.74,
    'btsc': 0.9,
    'spambase': 0.66,
    'beans': 0.28,
}



if __name__ == '__main__':

    full_eval()

    pass
