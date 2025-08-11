import argparse
import logging
import os.path

import torch
import pickle as pkl

from numpy.typing.tests.test_isfile import FILES
from torch.utils.hipify.hipify_python import meta_data
from tqdm import tqdm

from pathlib import Path

import lxg.util
from lxg.attribution import (
    integrated_gradients, smooth_grad, vanilla_grad,
    kernelshap, lime,
    # lime_limetab, shap_kernelshap,
    greedy_pgi_attribution, nup_ds_attribution)
from lxg import datasets

# from baselines.util import load_baselines

from lxg.evaluation import PGI_all_k, PGI_le_k


def save_expls(expl_dir,
               _dict_explanations, prefix, add_existing=False, new_key=None):
    """
    saves explanations and metrics in expl_dir and outputs_dir
    :param base_dir:
    :param expl_dir:
    :param model_dir:
    :param _dict_metrics: dictionary of misc metrics to be saved
    :param _dict_explanations: dictionary of explanations to be saved
    :param prefix: applied to filenames of both explanations and metrics
    """
    # task_ModelSeed_DataSeed_epoch_batch
    if not add_existing:
        for _exp_name, _expls in _dict_explanations.items():
            with open(f"{expl_dir}{prefix}_{_exp_name}.pkl", 'wb') as f:
                pkl.dump(_expls, f)
    else:
        for _exp_name, _expls in _dict_explanations.items():
            fname = f"{expl_dir}{prefix}_{_exp_name}.pkl"
            _existing_expls = dict()
            if os.path.exists(fname) and os.path.isfile(fname):
                f = open(fname, 'rb')
                _existing_expls = pkl.load(f)
            if new_key is not None:
                _existing_expls[new_key] = _expls
            else:
                _existing_expls.update(_expls)

            with open(f"{expl_dir}{prefix}_{_exp_name}.pkl", 'wb') as f:
                pkl.dump(_existing_expls, f)



def calc_explanations(X, Y=None, expl_funs: list[tuple[str, callable]] = [], eval_expl_fns=None, eval_fns=None):
    """
    call functions in expl_funs with args=(X,Y)
    :param X: input data
    :param Y: target labels
    :param expl_funs: list of tuple(explanation name: string, explanation(data,target): callable), eg ('ig', integrated_gradients)
    :return: dict[explanation name] containing explanations
    """

    _expl_dict = {}

    if eval_fns is not None:
        # functions that do sth with model on data and targets
        for _eval_name, _eval_fn in eval_fns:
            _expl_dict[_eval_name] = _eval_fn(X, Y)
    for _name_str, _expl_fun in expl_funs:
        print(_name_str)
        _expl_dict[_name_str] = {}
        results = lxg.util.timed_task(_expl_fun, dict(x=X, y=Y))
        explanations, time = results

        if Y is None:
            Y_pred = explanations[-1]
            if len(explanations) == 2:
                explanations = explanations[0]
            else:  # ig
                explanations = explanations[:-1]
        else:
            Y_pred = Y

        if eval_expl_fns is not None:
            # functions that quantify the computed explanations
            _eval_expl_results = dict(time=time)
            for _eval_expl_name, _eval_expl_fn in eval_expl_fns:
                print(f"eval {_eval_expl_name}")
                if 'ig' in _name_str:
                    explanations = explanations[0]
                _eval_expl_results[_eval_expl_name] = _eval_expl_fn(X, Y_pred, explanations)
            _eval_expl_results[_name_str] = explanations
            _eval_expl_results['Y'] = Y_pred
            _expl_dict[_name_str] = _eval_expl_results



    return _expl_dict


def train_step(model, optim, loss_fn, X, Y, device='cpu'):
    """
    Performs standard trainstep in pytorch
    :param model: model to be trained
    :param optim: optimizer holding models parameters
    :param loss_fn: loss function with input (model prediction, target labels)
    :param X: one batch of training data
    :param Y: target labels
    :param device: devidce where to run on
    :return: loss value as python type
    """
    X = X.to(device)
    Y = Y.to(device)
    out = model(X)
    loss = loss_fn(out, Y)
    optim.zero_grad()
    loss.backward()
    optim.step()
    optim.zero_grad()

    return loss.item()


def set_logger_level(l):
    l = l.upper()
    if l in logging._nameToLevel.keys():
        logging.getLogger().setLevel(logging._nameToLevel[l])
    else:
        print(f'loglevel {l} not recognized')


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
    parser.add_argument('--gpu-id', default=-1, type=int)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--directory', default='./data/')
    parser.add_argument('--model-ids', default=None, type=int,
                        help=f"if none provided, expls for all models will be calculated")
    parser.add_argument('--loglevel', default='ERROR', type=str)
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--target', type=str, default='model')
    parser.add_argument('--expltarget', type=str, default=None)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--add-synthetic', type=bool, default=True)

    return parser


from lxg.util import get_all_model_seeds_sorted, load_meta_data, load_idxs_from_multiple_models, dump_pkl

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

    # from xai4description import _variables_xai4description as _variables
    # from bvf import _variables_cobot as _variables


    # from cfire import _variables_cfire as _variables
    # _base_dir = _variables.base_dir
    model_idxs = [-1]

    args = make_parser().parse_args()


    if args.task == 'classification-*':
        # classif_root = Path(_variables.get_data_dir(''))
        classif_root = Path('./data/xai4interpretation/')
        tasks = []
        for d in classif_root.iterdir():
            if d.is_dir() and 'classification' in str(d):
                tasks.append(str(d).split('/')[-1])
            print(f"found following classification tasks: {tasks}")
    elif args.task == 'hypercube':
        # classif_root = Path(_variables.get_data_dir(''))
        classif_root = Path('./data/xai4interpretation/')
        tasks = []
        for d in classif_root.iterdir():
            if d.is_dir() and 'hypercube' in str(d):
                tasks.append(str(d).split('/')[-1])
            print(f"found following classification tasks: {tasks}")
    else:
        tasks = [args.task]
    # tasks = [
    #     "hypercube-10000-3-0-0-0-0.01-0.01",
    #     "hypercube-10000-3-3-0-0-0.01-0.01",
    #     "hypercube-10000-4-0-0-0-0.01-0.01",
    #     "hypercube-10000-4-4-0-0-0.01-0.01",
    #     "hypercube-10000-5-0-0-0-0.01-0.01",
    #     "hypercube-10000-5-5-0-0-0.01-0.01",
    # ]
    print(f"tasks: {tasks}")
    # tasks = _variables.make_classification_configs
    # tasks = ['heloc', 'beans', 'ionosphere', 'breastcancer']
    # tasks = ['spf', 'spambase', 'btsc', 'breastw']
    # tasks = [
    #     'breastw',
    #     "ionosphere",
    #     "breastcancer",
    #     'btsc',
    #     'spf',
    #     'spambase',
    #     'heloc',
    #     'beans',]
    # tasks = [args.task]
    assert len(tasks) == 1
    for task in tqdm(tasks, desc=f"tasks", position=0, leave=True):
        # data_dir = _variables.get_data_dir(task)
        print("\n\n\n")
        print(task)


        data_dir = './data/xai4interpretation/nn/' + task + '/'


        logging.debug(f'base dir: {str(data_dir)}')
        
        model_dir = Path(data_dir, 'models')
        results_dir = _variables.get_result_dir(task)
        results_dir.mkdir(parents=True, exist_ok=True)

        model_dir = data_dir + 'models/';
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        expl_dir = data_dir + 'explanations/';
        # print(f"looking for {expl_dir}...")
        # if Path(expl_dir).exists():
        #     print("... found, skipping")
        #     continue
        Path(expl_dir).mkdir(exist_ok=True)
        outputs_dir = data_dir + 'outputs/';
        Path(outputs_dir).mkdir(exist_ok=True)
        losses_dir = data_dir + 'losses/';
        Path(losses_dir).mkdir(exist_ok=True)
        acc_dir = data_dir + 'accuracies/';
        Path(acc_dir).mkdir(exist_ok=True)

        # model_seeds = get_all_model_seeds_sorted(data_dir)
        # print(model_seeds)
        # print(f"{task} -> {len(model_seeds)}")
        # continue
        from lxg.util import get_top_k_models
        model_seeds = [s[0] for s in get_top_k_models(data_dir, k=50)]
        print(f"computing expls for {len(model_seeds)} models")
        # ----------------------------------------------------------

        metadata = load_meta_data(data_dir, task)

        _models = load_idxs_from_multiple_models(data_dir, task, model_seeds, idxs=model_idxs, return_fns=True)
        (X_train, _), _, _, _, _ = datasets._get_dataset_callable(task)(random_state=metadata['data_seed'],
                                                                        batch_sizes=[1,1],
                                                                               as_torch=False)

        # continue
        compute_val_expls = True if args.expltarget == 'validationset' else False
        print(f"args.expltarget: {args.expltarget}")
        print(f"-> compute_eval_expls: {compute_val_expls}")
        if compute_val_expls:
            data, targets = metadata['X_val'], metadata['Y_val']
        else:
            print("adding synthetic data")
            data, targets = metadata['X'], metadata['Y']
            if args.add_synthetic:
                import numpy as np
                _min, _max = np.min(data.numpy(), 0), np.max(data.numpy(), 0)
                _span = np.abs(_max - _min)
                n_samples = len(data)
                # _noise_level = 0.05
                # synth_pos = data + np.expand_dims(_span*_noise_level, 0)
                # synth_neg = data - np.expand_dims(_span*_noise_level, 0)
                n_synth_samples = len(data)*19
                # n_synth_samples = max([250-n_samples, n_samples]) # min n_samples+n_synth = 250, max double data effectively
                synth_data = np.random.uniform(_min, _max, size=(n_synth_samples, len(_min)))
                targets = None
                data = torch.vstack([data, torch.from_numpy(synth_data)]).float()
                # data = torch.from_numpy(synth_data).float()
                # synth_data = torch.vstack((synth_pos, synth_neg))
                # data = torch.vstack([data, synth_data])
                # save synthetic data
                _fname = data_dir+'/synthetic_data.pkl'
                dump_pkl(synth_data, _fname)
                print(f"saved new synthetic data to {_fname}, num synthetic samples: {n_synth_samples}")




        # compute_expl_for_model_label = True
        if args.target == 'model':
            compute_expl_for_data_label = False
        elif args.target == 'data':
            compute_expl_for_data_label = True
        else:
            raise ValueError

        assert args.target == 'model'


        if not compute_expl_for_data_label:
            targets = None

        if task not in datasets.nlp_tasks:
            data_val_range = torch.abs(torch.max(data) - torch.min(data))
        else:
            data_val_range = 2.  # ~variance of BiLSTM.embedding

        kernelshap_mask = torch.arange(0, data.shape[1])

        # convergence_deltas = {}


        results_by_model = []
        for i, (model, inference_fn, preprocess_fn) in enumerate(_models):#tqdm(_models, desc="models", position=1, leave=True):
            # if torch.cuda.is_available():
            #     model.to('cuda')
            print(f'{task} - {i} / {len(_models)} models - nsamples = {len(data)}')
            model_prefix = f"{task}_{model_seeds[i]}_{metadata['data_seed']}_EPOCH-BATCH" # somehow we need the model name here
            if model_prefix in Path(expl_dir).iterdir():
                print(f"found {model_prefix} in {expl_dir}")
                continue
            model.to('cpu')
            _perturb_args = {
                'model': model,
                'inference_fn': inference_fn,
            }
            _grad_expls_args = {
                'model': model,
                'inference_fn': inference_fn,
                'pre_process_fn': preprocess_fn
            }

            # from lxg.attribution import cmaes_baseline
            # n_dims, n_classes = datasets.__info_dim_classes[task]
            # torch_cmaes_baseline = _prediction = cmaes_solution = None
            # if compute_val_expls:  # load baseline
            #     for _exp_name in ['liub', 'igub', 'ksub']:
            #         fname = f"{expl_dir}{model_prefix}_{_exp_name}.pkl"
            #         if os.path.exists(fname) and os.path.isfile(fname):
            #             f = open(fname, 'rb')
            #             _existing_expls = pkl.load(f)
            #             torch_cmaes_baseline = _existing_expls['mask']
            #             _prediction = _existing_expls['mask_prediction']
            #             cmaes_solution = torch_cmaes_baseline.numpy()
            #             break
            #     print(f"using baseline: {cmaes_solution}")
            #     print(f"which model predicts as: {_prediction}")
            # # import sys; sys.exit()
            # if torch_cmaes_baseline is None or cmaes_solution is None:
            #     cmaes_solution, _prediction = cmaes_baseline(inference_fn, n_dims=n_dims, n_classes=n_classes)
            #     torch_cmaes_baseline = torch.tensor(cmaes_solution, dtype=torch.float)
            # baseline_dir = Path(data_dir, 'baselines/')
            targets = lxg.util._get_targets(inference_fn, data, model, 'cpu')

            # baselines = load_baselines(baseline_dir, model_seeds[i], data_seed=metadata['data_seed'])

            _expl_fns = [ # signature lambda DATA, TARGETS
                # ('vg', lambda x, y: vanilla_grad(**_grad_expls_args, data=x, targets=y)),
                # ('sg', lambda x, y: smooth_grad(**_grad_expls_args, std=data_val_range, data=x, targets=y,
                #                                       n_samples=100),
                #  ),
                # ('grdpgi', lambda x, y: greedy_pgi_attribution(**_perturb_args, data=x, targets=y)
                #  ),
                # ('ig', lambda x, y: integrated_gradients(**_grad_expls_args, data=x, targets=y, n_samples=200,
                #                                                return_convergence_delta=True),
                #  ),
                ('ks', lambda x, y: kernelshap(**_perturb_args, data=x, targets=y, masks=kernelshap_mask,
                                                     n_samples=300),
                 ),
                ('li', lambda x, y: lime(**_perturb_args, data=x, targets=y, n_samples=300),
                 ),
                # ('lit', lambda x, y: lime_limetab(**_perturb_args, data=x, targets=y, baselines=np.array(X_train))),
                # ('sks', lambda x, y: shap_kernelshap(**_perturb_args, data=x, targets=y, baselines=np.array(X_train))),
                ('ig', lambda x, y: integrated_gradients(**_grad_expls_args, data=x, targets=y, n_samples=200,
                                                               return_convergence_delta=True, subtract_baseline=True),
                 ),

            # ('ksub', lambda x, y: kernelshap(**_perturb_args, data=x, targets=y, baseline=torch_cmaes_baseline,
                #                                masks=kernelshap_mask,
                #                                n_samples=300),
                #  ),
                # ('liub', lambda x, y: lime(**_perturb_args, baselines=torch_cmaes_baseline, data=x, targets=y, n_samples=300),
                #  ),
                # ('igub', lambda x, y: integrated_gradients(**_grad_expls_args, data=x, targets=y, n_samples=200,
                #                                            baselines=cmaes_solution, fit_baseline_data=True,
                #                                            return_convergence_delta=True),
                #  ),
                # ('ds', lambda x, y: nup_ds_attribution(**_perturb_args, data=x, targets=y, mass=1., device='cpu',
                #                                        n_classes=None)
                #  ),
            ]

            _pgi_args = {
                'model': model,
                'inference_fn': inference_fn
            }

            # _eval_fns = [ # signature lambda DATA, TARGETS
            #     ('pgiall3', lambda X, Y: PGI_all_k(**_pgi_args, data=X, targets=Y, k=3, mask=0)),
            # ]
            _eval_fns = None

            _eval_expl_fns = [  # signature
                ('pgile0.33', lambda X, Y, E: PGI_le_k(**_pgi_args, data=X, targets=Y, attributions=E, k=0.33,
                                                       mask=torch.zeros_like(X))),
            ]
            _expl_expl_fns = [
                ('stud', lambda X,Y,E: [])
            ]
            # assert targets is None
            _explanations = calc_explanations(data, targets, _expl_fns, _eval_expl_fns, _eval_fns)
            # for _ub_name in ['liub', 'igub', 'ksub']:
            #     if _ub_name in _explanations.keys():
            #         _explanations[_ub_name]['mask'] = torch_cmaes_baseline
            #         _explanations[_ub_name]['mask_prediction'] = _prediction

            # import sys; sys.exit()
            print(f"saving {expl_dir}, {model_prefix}")
            import sys;sys.exit()
            if args.expltarget == 'validationset':
                save_expls(expl_dir, _explanations, model_prefix, add_existing=True, new_key='val')
            elif args.add_synthetic and args.expltarget == 'testset':
                save_expls(expl_dir, _explanations, model_prefix, add_existing=True, new_key='synth')
            else:
                try:
                    save_expls(expl_dir, _explanations, model_prefix, add_existing=True, new_key=None)
                except FileNotFoundError:
                    save_expls(expl_dir, _explanations, model_prefix, add_existing=False)

            # SAVE EXPLANATIONS
            model.to('cpu')  # remove from gpu to make space for next model

