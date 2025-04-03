import os

import argparse
import logging
import torch
import pickle as pkl
import numpy as np

from tqdm import tqdm

from torch import nn
from time import time
from pathlib import Path
from sklearn.metrics import confusion_matrix

import lxg.util
from lxg.util import create_checkpoint, _get_n_digits, _check_improved, _training_finished, _sample_new_model, TorchRandomSeed
from lxg.datasets import _get_dataset_callable, nlp_tasks


def save_eval(base_dir, outputs_dir,
              _dict_metrics, prefix):
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
    if len(_dict_metrics) > 0:
        with open(f"{outputs_dir}{prefix}_out.pkl", 'wb') as f:
            pkl.dump(_dict_metrics, f)


def eval_model(model, X, Y, inference_fn, expl_funs: list[tuple[str, callable]] = [], preprocess_fn=None):
    '''
    calculate
    - model-outputs
    - explanations from callables
    :param model: callable
    :param X: test-data
    :param Y: labels for X
    :param expl_funs: list of tuple(explanation name: string, explanation(data,target): callable), eg ('ig', integrated_gradients)
    :return:
    '''

    logging.debug("EVAL MODEL")
    start = time()

    model.eval()
    device = next(model.parameters()).device

    # if preprocess_fn is not None:
    #     _X = preprocess_fn(X.to(device))
    # else:
    #     _X = X
    X = X.to(device);
    _metrics_dict = eval_performance(model, X, Y)


    model.train()  # tracking gradients not possible lstm otherwise; we have to zero grads then before exit?


    end = time()

    logging.debug(f"EVAL TOOK {end-start} seconds")

    return _metrics_dict


def eval_performance(model, X, Y):
    """
    compute accuracy, output distribution and confusion matrix
    :param model: the model, used only to determine on what device to run evaluation on
    :param X: input data
    :param Y: target labels
    :param inference_fn: callable that returns model prediction for batch input
    :param preprocess_fn: optional, default: None; callable that is applied to X before passing it to the model
    :return: dict with keyes 'accuracy', 'output_distribution', 'confusion_matrix' containing results
    """

    with torch.no_grad():
        _y_distr = model(X)
        _y = _y_distr.argmax(-1).to('cpu')
        accuracy = (_y == Y).to(torch.float32).mean().item()


    _confusion_matrix = confusion_matrix(Y, _y)
    _metrics_dict = {
        'accuracy': accuracy,
        'output_distribution': _y_distr.to('cpu'),
        'confusion_matrix': _confusion_matrix,
    }

    return _metrics_dict


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

    def dict_list(input: str) -> dict:
        input = input.replace('[', '').replace(']', '').replace(' ', '')
        input = input.split(',')
        n_args = len(input)
        _d_out = {}
        for i in np.arange(n_args)[0::2]:
            _str_arg = input[i]
            _str_arg_val = input[i+1]
            if _str_arg in ['flip_y', 'std']:
                _arg_val = float(_str_arg_val)
            else:
                _arg_val = int(_str_arg_val)
            _d_out[_str_arg] = _arg_val
        return _d_out



    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-id', default=-1, type=int)
    parser.add_argument('--dataset', type=str, default='hypercube')
    parser.add_argument('--batch-sizes', type=int_list, default=[64, 128])
    parser.add_argument('--modelparams', type=int_list, default=[], help="parameters beside input/output size")
    parser.add_argument('--directory', default='./data/')
    parser.add_argument('--data-seed', default=42, type=int)
    parser.add_argument('--num-runs', default=50, type=int)
    parser.add_argument('--max-epochs', default=20, type=int)
    parser.add_argument('--eval-freq', default=1, type=int, help="frequency in batches how often model is evaluated")
    parser.add_argument('--model-seed', default=42, type=int)
    parser.add_argument('--loglevel', default='ERROR', type=str)
    parser.add_argument('--training-length', default=0, type=int, help="if > 0, is the exact number of batches one "
                                                                       "model will be trained for")
    parser.add_argument('--kwargs-data', default=None, type=dict_list)

    return parser


def _fit_decision_tree_dnf(X, Y, _x_test, _y_test, n_features, max_depth=None, seed=42):
    n_classes = len(np.unique(Y))
    if max_depth is None:
        max_depth = n_features * 2
    max_leaf_nodes = min(n_classes ** 2, 3 * n_classes)
    from sklearn.tree import DecisionTreeClassifier as DT
    dt = DT(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, random_state=seed)
    dt.fit(X, Y)
    train_acc = dt.score(X, Y)
    test_acc = dt.score(_x_test, _y_test)

    from lxg.models import DNFClassifier
    dnf = DNFClassifier.from_DT(dt)

    return {
        'dt': dt,
        'acc_tr': train_acc,
        'acc_te': test_acc,
        'dnf': dnf
    }


if __name__ == '__main__':

    try:
        torch.backends.cudnn.tie_break = True
        torch.backends.cudnn.benchmark = False
    except NameError or ModuleNotFoundError:
        pass

    # can cause RuntimeError to be thrown if one of the operations is used where no deterministic impl is available, see:
    # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms
    # torch.use_deterministic_algorithms(True)
    # the CNNs use some functions where no deterministic alternative is available

    # parse args
    args = make_parser().parse_args()
    set_logger_level(args.loglevel)
    # get args
    n_models = args.num_runs
    task = args.dataset.lower();
    if 'hypercube' not in task: assert callable(_get_dataset_callable(task))
    tasks = [task]

    from cfire import _variables_cfire as _variables
    #
    # tasks = _variables.make_classification_configs

    for task in tasks:
        if task == 'classification':
            d = args.kwargs_data
            task = (f"classification-{d['n_samples']}-{d['n_features']}-{d['n_informative']}-{d['n_redundant']}-"
                    f"{d['n_repeated']}-{d['n_classes']}-{d['n_clusters_per_class']}-{d['flip_y']}")
        elif task == 'hypercube':
            d = args.kwargs_data
            task = (f"hypercube-{d['n_samples']}-{d['n_informative']}-{d['n_constant']}-{d['n_redundant']}-"
                    f"{d['n_repeated']}-{d['flip_y']}-{d['std']}")

        device = 'cpu' if torch.cuda.device_count()-1 > args.gpu_id < 0  else f"cuda:{args.gpu_id}"
        logging.debug(f'device: {device}')
        modelparams = args.modelparams
        batch_sizes = args.batch_sizes
        training_length = args.training_length

        # make/ check folders
        base_dir = args.directory+f'/{task}/'; logging.debug(f'base dir: {base_dir}')
        model_dir = base_dir+'models/'; Path(model_dir).mkdir(parents=True, exist_ok=True)
        outputs_dir = base_dir+'outputs/'; Path(outputs_dir).mkdir(exist_ok=True)
        losses_dir = base_dir+'losses/'; Path(losses_dir).mkdir(exist_ok=True)
        acc_dir = base_dir+'accuracies/'; Path(acc_dir).mkdir(exist_ok=True)

        # generate seeds for models
        with TorchRandomSeed(args.model_seed):
            seeds = torch.randint(high=420000, size=(n_models,), dtype=torch.int)
            seeds = [s.item() for s in seeds]

        # get dataloaders and input/ output dims for task
        if task == 'bool':
            train_loader, test_loader, input_size, n_classes = \
                _get_dataset_callable(task)(random_state=args.data_seed, batch_sizes=batch_sizes)
        elif task in _variables.make_classification_configs:
            __task_data = lxg.util.load_pkl(Path(base_dir, 'data.pkl'))
            (np_X_tr, np_Y_tr) = __task_data['train']
            (np_X_te, np_Y_te) = __task_data['test']
            batch_sizes = batch_sizes[0], len(np_X_te)
            from lxg.datasets import _wrap_numpy_to_loader
            input_size =np_X_tr.shape[1]
            n_classes = max(np.unique(np_Y_tr))+1
            train_loader, test_loader = (_wrap_numpy_to_loader(X=np_X_tr, Y=np_Y_tr, batch_size=batch_sizes[0], shuffle=True),
                           _wrap_numpy_to_loader(X=np_X_te, Y=np_Y_te, batch_size=batch_sizes[1]))
            __cfg = lxg.util.get_classification_config_from_fname(task)
            if __cfg['n_features'] < 4:
                modelparams = [8, 8]
            elif __cfg['n_features'] == 4:
                modelparams = [16, 16]
            elif __cfg['n_features'] > 4:
                modelparams = [32, 32]
            args.modelparams = modelparams

        elif task in _variables.hypercube_configs:
            __task_data = lxg.util.load_pkl(Path(base_dir, 'data.pkl'))
            (np_X_tr, np_Y_tr) = __task_data['X_tr'], __task_data['Y_tr']
            (np_X_te, np_Y_te) = __task_data['X_te'], __task_data['Y_te']
            batch_sizes = batch_sizes[0], len(np_X_te)
            from lxg.datasets import _wrap_numpy_to_loader
            input_size =np_X_tr.shape[1]
            n_classes = len(np.unique(np_Y_tr))
            train_loader, test_loader = (_wrap_numpy_to_loader(X=np_X_tr, Y=np_Y_tr, batch_size=batch_sizes[0], shuffle=True),
                           _wrap_numpy_to_loader(X=np_X_te, Y=np_Y_te, batch_size=len(np_Y_te)))
            __cfg = lxg.util.get_hypercube_config_from_fname(task)
            modelparams = [100, 100]


        elif str(task).startswith('classification'):
            batch_sizes = batch_sizes[0], -1#int(args.kwargs_data['n_samples']*0.2)
            train_loader, test_loader, val_loader, input_size, n_classes = \
                _get_dataset_callable('classification')(random_state=args.data_seed, batch_sizes=batch_sizes,
                                            kwargs=args.kwargs_data)
            (X_tr, Y_tr), (X_te, Y_te), (X_val, Y_val) = _get_dataset_callable('classification')(random_state=args.data_seed, batch_sizes=batch_sizes,
                                                    kwargs=args.kwargs_data, as_torch=False)

            print(args.kwargs_data)
        else:
            # we flip validation and test here to keep the 'official' test split separate and instead use 'validation' to compute explanations
            train_loader, test_loader, val_loader, input_size, n_classes = (
                _get_dataset_callable(task)(random_state=args.data_seed, batch_sizes=batch_sizes))
            validation = next(iter(val_loader))
        # print(len(validation));import sys; sys.exit()
        # Save test data and some basic statistics
        Y_train = []
        n_batches = 0

        for _, y in train_loader:
            Y_train.append(y)
            n_batches += 1
            if training_length > 0 and n_batches >= training_length:
                break

        # used in epoch-batch prefix in filenames to avoid ordering 1,10,11,..., 2, 20 ...
        n_digits_batch = _get_n_digits(n_batches)
        n_digits_epoch = _get_n_digits(args.max_epochs)

        Y_train = torch.cat(Y_train)
        Y_train_distr = torch.unique(Y_train, return_counts=True)[1].to(torch.float)
        Y_train_distr /= torch.sum(Y_train_distr).item()
        print(Y_train_distr)


        X_test, Y_test = next(iter(test_loader))
        Y_test_distr = torch.unique(Y_test, return_counts=True)[1].to(torch.float)
        Y_test_distr /= torch.sum(Y_test_distr); logging.debug(Y_test_distr)
        print(Y_test_distr)
        assert len(torch.unique(Y_test)) == n_classes

        pid = os.getpid()
        metadata_fname = f'/meta_data_pid{pid}_{task}_{args.data_seed}.pkl'

        with open(base_dir+metadata_fname, 'wb') as f:
            meta_data = {
                'dataset': task,
                'X': X_test,
                'Y': Y_test,
                'Y_train_distr': Y_train_distr,
                'Y_test_distr': Y_test_distr,
                'data_seed': args.data_seed,
                'batch_sizes': batch_sizes,
                'note': 'X, Y are test data for historic reasons, X_tr/X_val are train and validation splits'
                }
            if 'X_tr' in locals():
                meta_data['X_tr'] = X_tr
                meta_data['Y_tr'] = Y_tr
            if 'validation' in locals():
                meta_data['X_val'] = validation[0]
                meta_data['Y_val'] = validation[1]
            meta_data.update(vars(args))
            pkl.dump(meta_data, f)
            logging.debug(f'saved meta_data: {[k for k in meta_data.keys()]}')
        # prepare model setup
        modelparams = [input_size] + modelparams + [n_classes]
        logging.debug(f'model params: {modelparams}')

        # prepare arguments for explanations (SG)
        if task not in nlp_tasks:
            X_train = torch.cat([x for x, _ in train_loader])
            X_train_val_range = torch.abs(torch.max(X_train) - torch.min(X_train))
        else:
            X_train_val_range = 2.  # magic number: ~ diameter of space occupied by embedding vectors


        # n_batches = 1234, max_epochs=100, e=2, b=64 -> "0100-0064"
        def epoch_batch_prefix(e, b):
            epochs = ''.join((n_digits_epoch-_get_n_digits(e))*['0']+[str(e)]) if e != 0 else ''.join(n_digits_epoch*['0'])
            batches = ''.join((n_digits_batch-_get_n_digits(b))*['0']+[str(b)]) if b != 0 else ''.join(n_digits_batch*['0'])
            return f'{epochs}-{batches}'

        kernelshap_mask = torch.arange(0, X_test.shape[1]).unsqueeze(0)

        for s, seed in enumerate(seeds):  # run everything; data remains the same
            print(f'STARTING SEED {s}/{n_models}')
            logging.debug(f'running with seed {seed}')
            # only in case of nlp will inference_fn and preprocessing be given, the latter to embed sequence for IG/SG
            model, inference_fn, _ = _sample_new_model(task, modelparams, seed)
            model.to(device)
            model_id = f'{seed}_{args.data_seed}'
            model_id = f'{task}_{model_id}'
            if inference_fn is None:
                inference_fn = model



            # setup check to stop training
            performance_improved = False
            training_finished = False
            test_acc, losses = [], []

            optim = torch.optim.Adam(model.parameters())
            loss_function = nn.CrossEntropyLoss()

            # eval + save before training
            _dict_metrics = eval_model(model=model, X=X_test, Y=Y_test, inference_fn=inference_fn)
            test_acc.append(_dict_metrics['accuracy'])
            prefix = model_id + '_' + epoch_batch_prefix(0, 0)  # task_ModelSeed_DataSeed_epoch-batch

            logging.debug(f'saving with prefix {prefix}')
            save_eval(
                base_dir, outputs_dir,
                _dict_metrics,
                prefix=prefix
            )
            fname = model_dir + model_id + '_' + epoch_batch_prefix(0, 0) + '.ckpt'
            create_checkpoint(fname,
                              model, optimizer=None)
            logging.debug(f'checkpoint created {fname}')
            # keep count on how many batches we trained; esp. relevant if args.training_length > 0
            n_batches_trained_on = 0
            if tasks == _variables.make_classification_configs:
                _tree_dnf_baseline_dict = _fit_decision_tree_dnf(np_X_tr, np_Y_tr, np_X_te, np_Y_te, input_size,
                                                                 seed=seed)
                tree_pth = Path(base_dir,'trees'); _variables.__create_dir(tree_pth)
                lxg.util.dump_pkl(_tree_dnf_baseline_dict,
                                  fname=Path(tree_pth, f"tree_{seed}.pkl"))


            with TorchRandomSeed(args.data_seed):
                for epoch in range(args.max_epochs):
                    print(f'    STARTING EPOCH {epoch}/{args.max_epochs}')
                    for i, (x, y) in tqdm(enumerate(train_loader)):

                        # do one training step, log loss
                        loss = train_step(model, optim, loss_function, x, y, device=device)
                        losses.append(loss)
                        n_batches_trained_on += 1

                        if args.eval_freq > 0 and i % args.eval_freq == 0:
                            # print(f"eval @ {i}")
                            # _X_test = X_test if not callable(_preprocess_for_explanations) else _preprocess_for_explanations(X_test)
                            _dict_metrics = \
                                eval_model(model=model, X=X_test, Y=Y_test, inference_fn=inference_fn)
                            test_acc.append(_dict_metrics['accuracy'])
                            prefix = model_id + '_' + epoch_batch_prefix(epoch, i+1)  # task_ModelSeed_DataSeed_epoch-batch
                            logging.debug(f'saving with prefix {prefix}')
                            save_eval(
                                base_dir, outputs_dir,
                                _dict_metrics,
                                prefix=prefix
                            )

                        if i % 100 == 0:
                            print(f'    FINISHED: {epoch}.{i} - test acc {test_acc[-1]}')

                        if 0 < training_length: # if parameter training_length is set
                            # print(training_length, n_batches_trained_on)
                            if training_length <= n_batches_trained_on:  # and we have trained for enough batches, break
                                break
                        else:
                            if i+epoch*i < 10*args.eval_freq:  # don't do anything before we have not evaluated model at least 10x
                                continue
                            if not performance_improved: # if performance has not improved until previous batch, continue training anyway
                                performance_improved = _check_improved(test_acc, losses)
                                continue
                            # performance had improved since last batch, check if we're done
                            training_finished = _training_finished(test_acc, losses)
                            if training_finished:  # break out of batch loop
                                break
                    if args.eval_freq < 0:
                        _dict_metrics = eval_model(model=model, X=X_test, Y=Y_test, inference_fn=inference_fn)
                        test_acc.append(_dict_metrics['accuracy'])
                        prefix = model_id + '_' + epoch_batch_prefix(epoch, i + 1)  # task_ModelSeed_DataSeed_epoch-batch
                        logging.debug(f'saving with prefix {prefix}')
                        save_eval(
                            base_dir, outputs_dir,
                            _dict_metrics,
                            prefix=prefix
                        )

                    # break out of epoch loop
                    if training_finished or (0 < training_length and training_length <= n_batches_trained_on) or epoch >= (args.max_epochs-1):
                        print(f'        FINISHED MODEL #{s}@{epoch}{i} after {n_batches_trained_on} batches total')
                        model.eval()
                        train_accs = []
                        for (_x, _y_true) in train_loader:
                            # _y_distr = model(_x.to('cuda:1'))
                            _y_distr = model(_x.to('cpu'))
                            _y = _y_distr.argmax(-1).to('cpu')
                            accuracy = (_y == _y_true).to(torch.float32).mean().item()
                            train_accs.append(accuracy)
                        model.train()
                        train_acc = np.mean(train_accs)
                        print(f'final train accuracy: {train_acc}')
                        print(f'final test accuracy:  {test_acc[-1]}')
                        break
                    print(f'        last test accuracies: \n\t\t\t{test_acc[-10:]}')

            create_checkpoint(model_dir+model_id+'_'+epoch_batch_prefix(epoch, i)+'.ckpt',
                                  model, optimizer=None)

            _dict_metrics = eval_model(model=model, X=X_test, Y=Y_test, inference_fn=inference_fn)

            prefix = model_id + '_' + epoch_batch_prefix(args.max_epochs, len(train_loader)+2)  # task_ModelSeed_DataSeed_epoch-batch
            logging.debug(f'saving with prefix {prefix}')
            save_eval(
                base_dir, outputs_dir,
                _dict_metrics,
                prefix=prefix
            )
            # task_ModelSeed_DataSeed_loss/testacc
            fname_loss = losses_dir+model_id+'_loss.pkl'
            fname_acc = acc_dir+model_id+'_testacc.pkl'
            logging.debug(f'saving loss, acc in \n\t{fname_loss}\n\t{fname_acc}')
            with open(fname_loss, 'wb') as f:
                pkl.dump(losses, f)
            with open(fname_acc, 'wb') as f:
                pkl.dump(test_acc, f)


            del losses, test_acc, model, optim


    '''
    for ease of access:
    plot all accuracies, save list of seeds sorted by accuracies
    '''
