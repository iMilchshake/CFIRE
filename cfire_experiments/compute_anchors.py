default_path = '../'
import os
os.chdir(default_path)

import argparse
from itertools import product
from pathlib import Path
import pickle as pkl
from joblib import parallel_backend, Parallel, delayed

import torch
import numpy as np

from anchor.anchor import utils
from anchor.anchor import anchor_tabular

import lxg
from lxg.util import (get_all_model_seeds_sorted, load_meta_data, load_idxs_from_multiple_models, load_sklearn_models,
                      anchors_to_rule_model, anchors_dataset, timed_task)
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def _comp_anchors(task, modelclass, model_seed):
    print(f"starting {task} {modelclass}-{model_seed}")
    data_dir = './data/cfire/' + modelclass + '/' + task
    anchors_dir = Path(data_dir, 'anchors')
    logging.debug(f'base dir: {str(data_dir)}')

    model_dir = Path(data_dir, 'models')

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    anchors_dir = data_dir + '/anchors'
    Path(anchors_dir).mkdir(parents=True, exist_ok=True)
    print(f"starting baseline computation for {modelclass} {model_seed} {task}")

    metadata = load_meta_data(data_dir, task)

    data = metadata['X']

    if type(data) is np.ndarray:
        data = torch.from_numpy(data)

    if modelclass == 'nn':
        _model = load_idxs_from_multiple_models(data_dir, task, [model_seed], idxs=model_idxs, return_fns=True)[0]
        model, inference_fn, _ = _model
    else:
        _model = load_sklearn_models(data_dir, [model_seed], wrap=True)[0]
        model, inference_fn = _model, _model.forward
        lxg.util.fix_parallelization_sklearn(model, modelclass, n_jobs=1)


    model = model.to('cpu')
    targets = lxg.util._get_targets(inference_fn, data, model, 'cpu')
    anchors_dataset = lxg.util.anchors_dataset_simple(data, targets, n_classes=lxg.datasets.__info_dim_classes[task][1])

    anchors_inference_fn = lambda x: (
        lxg.util._get_targets(inference_fn, torch.tensor(x), model, 'cpu').detach().numpy())

    explainer = anchor_tabular.AnchorTabularExplainer(
        anchors_dataset.class_names,
        anchors_dataset.feature_names,
        anchors_dataset.test,
        anchors_dataset.categorical_names)

    n_jobs_anchors = 6
    _timed_task = lambda _inputs: timed_task(explainer.explain_instance, _inputs)
    with parallel_backend(backend='loky', n_jobs=n_jobs_anchors):  # 14
        # Parallel()(delayed(rename)(task=a[0], expl_method=a[1], model_set=a[2], gely_threshold=a[3],
        #                                         significance_threshold=a[4], k_means_max_bins=a[5]) for a in arg_sets)
        results = Parallel(verbose=10, batch_size=2)(
            delayed(_timed_task)(dict(data_row=i, classifier_fn=anchors_inference_fn, threshold=0.9)) for i in anchors_dataset.test)
    exps, times = [r[0] for r in results], [r[1] for r in results]
    _anchors = [e.exp_map for e in exps]
    for a, t in zip(_anchors, times):
        a.update(dict(time=t))
    # SAVE ANCHORS
    fname = f"anchors_{model_seed}_{metadata['data_seed']}.pkl"
    lxg.util.dump_pkl(_anchors, Path(anchors_dir, fname))



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

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--modelclass', type=str, default='xgb')
    parser.add_argument('--baselines', type=str, default=None)
    parser.add_argument('--update-existing', type=bool, default=True)

    return parser


if __name__ == '__main__':

    import logging

    logging.logLevel = logging.DEBUG
    try:
        torch.backends.cudnn.tie_break = True
        torch.backends.cudnn.benchmark = False
    except NameError or ModuleNotFoundError:
        pass

    # can cause RuntimeError to be thrown if one of the operations is used where no deterministic impl is available, see:
    # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms
    # torch.use_deterministic_algorithms(True)
    # the CNNs use some functions where no deterministic alternative is available
    from cfire import _variables_cfire as _variables

    _base_dir = _variables.base_dir
    model_idxs = [-1]

    args = make_parser().parse_args()

    # tasks = _variables.make_classification_configs
    # tasks = ['heloc', 'beans', 'ionosphere', 'breastcancer']
    # tasks = ['spf', 'spambase', 'btsc', 'breastw']
    if args.task is None:
        tasks = _variables.tasks
    else:
        tasks = [args.task]

    modelclass = args.modelclass.lower()

    print(tasks)

    model_id_acc = {task: lxg.util.get_top_k_models(_variables.get_data_dir(task, modelclass=modelclass), k=50)
                    for task in tasks}
    for task, acc in model_id_acc.items():
        print(f"task {task}\t\t -> min, mean acc {np.min([a[1] for a in acc]):.2f}, {np.mean([a[1] for a in acc]):.2f}")
        # print(f"{np.mean([a[1] for a in acc]):.2f} ~ {np.std([a[1] for a in acc]):.2f}")
    # import sys;sys.exit()
    model_sets = {task: [x[0] for x in id_acc]
                  for task, id_acc in model_id_acc.items()}

    arg_sets = []
    for t in tasks:
        ids = model_sets[t]
        arg_sets.extend(list(product([t], ids)))

    n_jobs = 2

    with parallel_backend(backend='loky', n_jobs=n_jobs):  # 14
        Parallel(verbose=10, batch_size=2)(delayed(_comp_anchors)(
            task=a[0], model_seed=a[1], modelclass=modelclass
        ) for a in arg_sets)