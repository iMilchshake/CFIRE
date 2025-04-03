import os.path
import pathlib

import torch
import logging
import numpy as np
import pickle as pkl
import json

from pathlib import Path

from .models import BiLSTMClassif, make_fmnist_small, make_ff, SimpleTorchWrapper, DNFClassifier

from .datasets import TorchRandomSeed, nlp_tasks, cv_tasks, _get_dim_classes

from torch.utils.data import DataLoader, TensorDataset
# ---

import psutil
import multiprocessing
import time

from anchor.anchor.utils import Bunch, greedy_pick_anchor
def terminate_process_tree(pid):
    """Terminate a process tree rooted at the given pid."""
    try:
        logging.debug(f"killing process tree with pid {pid}")
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            child.terminate()
        _, still_alive = psutil.wait_procs(children, timeout=2)
        # for child in still_alive:
        #     child.kill()
        parent.terminate()
        parent.wait(10)
        # parent.kill()
    except psutil.NoSuchProcess:
        logging.debug(f"terminating process couldn't find process")
        pass

def monitor_memory(proc, memory_limit, queue):
    """Monitor the memory usage of a process and terminate it if it exceeds the limit."""
    logging.debug(f"Started monitoring process with PID: {proc.pid}")
    while proc.is_alive() and queue.empty():
        try:
            mem_info = psutil.Process(proc.pid).memory_info()
            if mem_info.rss >= memory_limit:
                logging.warning(f"Memory limit exceeded: {mem_info.rss} > {memory_limit}. Terminating process tree.")
                terminate_process_tree(proc.pid)
                return False
            time.sleep(1)
        except psutil.NoSuchProcess:
            logging.debug(f"Process {proc.pid} no longer exists.")
            break
    logging.debug(f"Finished monitoring process with PID: {proc.pid}")
    return True


def safe_call(func, kwargs, memory_limit):
    """Call a function in a separate process and monitor its memory usage."""

    queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=func, kwargs={'result_queue': queue, **kwargs})
    proc.start()

    monitor_result = monitor_memory(proc, memory_limit, queue)

    results = None


    if not monitor_result:
        logging.debug(f"Function call with args {kwargs} was terminated due to excessive memory usage.")
    else:
        logging.debug(f"func put result in queue")
        if queue.empty():
            logging.debug(f"YET THE QUEUE WAS EMPTY")
            raise RuntimeError
        else:
            logging.debug(f"Retrieving results from non-empty queue.")
            logging.debug(queue)
            results = queue.get()
            logging.debug("results:")
            logging.debug(results)
            logging.debug(f"Result retrieved from queue.")

    proc.join(timeout=5)  # Add a timeout to prevent indefinite hanging
    if proc.is_alive():
        logging.debug(f"Process did not finish in time and is being terminated. PID: {proc.pid}")
        terminate_process_tree(proc.pid)
        proc.join()
    logging.debug(f"Memory monitor returned {monitor_result}")
    logging.debug(f"joining proc {proc.pid}")
    queue.close()
    queue.join_thread()
    # print(results)
    # import sys;sys.exit()
    return results


def timed_task(callable, kwargs):
    start = time.time()
    result = callable(**kwargs)
    end = time.time()
    elapsed = end - start
    return result, elapsed

# ---

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        return json.JSONEncoder.default(self, obj)


# ---

def _set_random_state(random_state=None):
    if random_state is None:
        npr = np.random.RandomState(1234)
    elif isinstance(random_state, int):
        npr = np.random.RandomState(random_state)
    elif isinstance(random_state, np.random.RandomState):
        npr = random_state
    else:
        raise Exception('Unknown input type of random state:', type(random_state))
    np.random.set_state(npr)


def _sample_new_model(task, modelparams, seed) -> (callable, callable, callable):
    with TorchRandomSeed(seed):
        if task in nlp_tasks:
            vocab_size, embed_dim, hid_size, nr_classes = modelparams
            model = BiLSTMClassif(nr_classes=nr_classes, embed_dim=embed_dim,
                                  hid_size=hid_size, vocab_size=vocab_size)
            return model, model._forward_embedded_softmax, model.embed_sequences

        elif task in cv_tasks:
            _, n_classes = modelparams
            model = make_fmnist_small(n_classes)
            return model, model.predict_batch_softmax, None

        else:
            model = make_ff(modelparams)
            return model, model.predict_batch_softmax, None
            # return model, None, None


# ---


def _get_n_digits(num: int):
    """
    Count how many decimals a number as
    :param num: natural number
    :return: number of decimals
    """
    n = 0
    while num != 0:
        num = num // 10
        n += 1
    return n


def _check_improved(test_accs, losses, acc_window=5, loss_window=10):
    """Determines whether the model has improved wrt accuracy on test set and significant decrease in loss"""

    # if most recent accuracy is not 1.75 as good as baseline -> False
    baseline_acc = sum(test_accs[:acc_window])/acc_window
    baseline_acc = min(1.75*baseline_acc, 1.)  #
    recent = sum(test_accs[-acc_window:])/acc_window
    logging.debug(f'_check_improved: baseline acc {baseline_acc} vs recent acc {recent}')
    if baseline_acc > recent:
        return False
    # if most recent loss still more than half of initial loss -> False
    baseline_loss = 0.5 * sum(losses[:loss_window])/loss_window
    recent_loss = sum(losses[-loss_window:])/loss_window
    logging.debug(f'_check_improved: baseline loss {baseline_loss} vs recent loss {recent_loss}')
    if baseline_loss <= recent_loss:
        return False
    # so if accuracy on test-set has improved and loss has decreased we say the model has improved!
    return True


def _training_finished(test_accs, losses, thresh_acc=0.01, thresh_loss=0.01, last_n=3, win_size=9):
    """
    _check_improved says the model has improved,
    _training_finished checks whether we still want to go on,
        eg loss keeps decreasing, test_acc increases again ...

    we do not consider the case where last_n or win_size is larger than length of test_accs or losses
    but since _training_finished only gets called afer _check_improved is True, this is probably fine

    """

    loss_recent = sum(losses[-last_n:])/last_n
    loss_win = sum(losses[-win_size:])/win_size
    loss_thresh = loss_win * thresh_loss
    loss_keeps_changing = abs(loss_win - loss_recent) > loss_thresh

    acc_recent = sum(test_accs[-last_n:])/last_n
    acc_win = sum(test_accs[-win_size:])/win_size
    acc_thresh = acc_win*thresh_acc
    test_acc_changing = abs(acc_win - acc_recent) > acc_thresh
    logging.debug(f'acc mark:{acc_win}, recent:{acc_recent}, thresh:{acc_thresh}')
    logging.debug(f'loss mark:{loss_win}, recent:{loss_recent}, thresh:{loss_thresh}')
    return not (loss_keeps_changing or test_acc_changing)


# ---
def create_checkpoint(path, model, optimizer=None):
    if optimizer is None:
        torch.save({
            'model_state_dict': model.state_dict(),
        }, path)
    else:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)


def restore_checkpoint(path, model, optimizer=None, train=True):
    if not torch.cuda.is_available():
        ckpt = torch.load(path, map_location=torch.device('cpu'))
    else:
        ckpt = torch.load(path, map_location=torch.device('cuda'))
    for k, v in ckpt['model_state_dict'].items():
        if not v.is_contiguous():
            ckpt['model_state_dict'][k] = v.contiguous()
    model.load_state_dict(ckpt['model_state_dict'])
    if train:
        model.train()
    try:
        if optimizer is not None:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    except KeyError as ke:
        print(str(ke))
        print("RESTORE_CHECKPOINT: Optimizer given but no Optimizer found in state dict.")

# ---


def _get_filenames(pth: Path):
    return [
        str(f) for f in pth.iterdir() if f.is_file()
    ]


def _get_batchnum(fname):
    batch = fname.split('_')[3].split('-')[1]
    batch = batch.split('.')[0].split('_')[0]  # first split suffices for models
    if batch == "BATCH":
        batch = -1
    return batch


def _get_epoch(fname):
    epoch = fname.split('_')[3].split('-')[0]
    if epoch == "EPOCH":
        epoch = -1
    return epoch


def _filter_by_data_seed(fnames, seed, sklearn=False):
    idx = 2
    if sklearn:
        idx = 2
    return sorted([fname for fname in fnames if seed == fname.split('_')[idx]])


def _filter_by_model_seed(fnames, seed, sklearn=False):
    idx = 1
    if sklearn:
        idx = 1
    return sorted([fname for fname in fnames if seed == fname.split('_')[idx]])


def _filter_by_explanation_method(fnames, xai_method):
    return sorted([fname for fname in fnames if xai_method == fname.split('_')[-1].split('.')[0]])


def get_all_model_seeds_sorted(data_dir, sklearn=False):
    idx = 1
    if sklearn:
        idx = 2
    fnames = _get_filenames(Path(data_dir, 'models'))
    seeds = []
    # task_modelseed_dataseed_epoch-batch
    for fname in fnames:
        seed = fname.split('_')[idx]
        seeds.append(seed)
    seeds = np.unique(seeds)
    seeds = sorted(seeds)
    return seeds

def _sort_fnames_by_epoch_batch(fnames):
    return sorted(fnames, key=lambda x: (int(_get_epoch(x)), int(_get_batchnum(x))))


# def _loss_idx_to_eval_idx(idxs, eval_freq=1):
#     # needed if eval was not done after each batch
#     if eval_freq == 1:
#         return idxs
#     else:
#         return [np.floor(idx/eval_freq, dtype=np.int) for idx in idxs]
#
#     pass


def load_losses(data_dir):  # have option to only load a certain modelID?
    fnames = _get_filenames(Path(data_dir, 'losses'))
    losses = {}
    # model seed becomes key
    for fname in fnames:
        modelSeed = fname.split('_')[1]
        with open(fname, 'rb') as file:
            _losses_model = pkl.load(file)
            losses[modelSeed] = _losses_model
    return losses


def load_accuracies(data_dir):  # have option to only load a certain modelID?
    fnames = _get_filenames(Path(data_dir, 'accuracies'))
    accuracies = {}
    # model seed becomes key
    for fname in fnames:
        if '/nn/' in str(data_dir):
            modelSeed = fname.split('_')[1]
        else:
            modelSeed = fname.split('_')[2]

        with open(fname, 'rb') as file:
            accs = pkl.load(file)
            accuracies[modelSeed] = accs
    return accuracies

def load_confmats(data_dir):
    fnames = _get_filenames(Path(data_dir, 'outputs'))
    cfm = {}
    # model seed becomes key
    for fname in fnames:
        if '/nn/' in str(data_dir):
            modelSeed = fname.split('_')[1]
        else:
            modelSeed = fname.split('_')[2]

        with open(fname, 'rb') as file:
            _eval_dict = pkl.load(file)
            cfm[modelSeed] = _eval_dict['confusion_matrix']
    return cfm


def load_outputs_sorted(data_dir, model_seed):
    outputs = []
    fnames = _get_filenames(Path(data_dir, 'outputs'))
    fnames = _filter_by_model_seed(fnames, model_seed)

    for fname in fnames:
        with open(fname, 'rb') as f:
            out = pkl.load(f)
            # TASK_MODELID_epoch-batchnum_{ig, sg, vg, ...}.pkl
            out['batch'] = _get_batchnum(fname)
            out['epoch'] = _get_epoch(fname)
            outputs.append(out)
    outputs = sorted(outputs, key= lambda o: (int(o['epoch']), int(o['batch'])))
    return outputs

def load_sklearn_models(data_dir, model_seeds, wrap=False):
    fnames_models = [str(f) for f in Path(data_dir, 'models').iterdir()]
    fnames = [fname for fname in fnames_models if fname.split('_')[2] in model_seeds]
    models = []
    for fname in fnames:
        model = load_pkl(fname)
        if wrap:
            model = SimpleTorchWrapper(model)
        models.append(model)
    return models

def fix_parallelization_sklearn(model, modelclass, n_jobs):
    if modelclass == 'xgb':
        model.model.set_params(n_jobs=n_jobs)
    if modelclass == 'rf':
        model.model.n_jobs = n_jobs  # model is of class SimpleTorchWrapper
    if modelclass == 'svm':
        pass  # not parallelism here

def load_sklearn_outputs(data_dir, model_seeds):
    fnames_models = [str(f) for f in Path(data_dir, 'outputs').iterdir()]
    fnames = [fname for fname in fnames_models if fname.split('_')[2] in model_seeds]
    outputs = []
    for fname in fnames:
        out = load_pkl(fname)
        outputs.append(out)
    return outputs


def load_idxs_from_model(data_dir, task, model_seed, idxs: list[int], return_fns=False):
    # get all modelparams =
    meta_data = load_meta_data(data_dir, just_one=True)
    modelparams = meta_data['modelparams']
    del meta_data
    dim, n_classes = _get_dim_classes(task)
    modelparams = [dim] + modelparams + [n_classes]
    fnames_models = [str(f) for f in Path(data_dir, 'models').iterdir()]
    fnames_models = _filter_by_model_seed(fnames_models, model_seed)
    fnames_models = _sort_fnames_by_epoch_batch(fnames_models)

    models = []
    for idx in idxs:
        fname = fnames_models[idx]
        model, inference_fn, preprocess_fn = _sample_new_model(task, modelparams, seed=0)
        restore_checkpoint(fname, model)
        models.append(
            (model, inference_fn, preprocess_fn) if return_fns else model
        )

    return models

def load_idxs_from_multiple_models(data_dir, task, model_seeds: list, idxs: list[int], return_fns=False):
    '''
    :return: list(model(idxs)) if len(idxs)>1 else list(models)
    '''
    meta_data = load_meta_data(data_dir, just_one=True)
    modelparams = meta_data['modelparams']
    if 'classification' in task:
        _t = task.split('-')
        dim, n_classes = int(_t[2]), int(_t[6])
    elif 'hypercube' in task:
        _t = task.split('-')
        dim, n_classes = meta_data['X'].shape[1], len(np.unique(meta_data['Y'].numpy()))
    else:
        dim, n_classes = _get_dim_classes(task)
    del meta_data
    modelparams = [dim] + modelparams + [n_classes]

    fnames_models = [str(f) for f in Path(data_dir, 'models').iterdir()]
    models = []
    for seed in model_seeds:
        fnames_models_filtered = _filter_by_model_seed(fnames_models, seed)
        fnames_models_filtered = _sort_fnames_by_epoch_batch(fnames_models_filtered)
        if len(idxs) > 1:
            models.append([])
        for idx in idxs:
            fname = fnames_models_filtered[idx]
            # this won't work for nlp models
            model, inference_fn, preprocess_fn = _sample_new_model(task, modelparams, seed=0)
            restore_checkpoint(fname, model)
            model.to('cpu')
            if len(idxs) > 1:
                models[-1].append(
            (model, inference_fn, preprocess_fn) if return_fns else model
        )
            else:
                models.append(
            (model, inference_fn, preprocess_fn) if return_fns else model
        )

    return models

def load_outputs_by_idxs(data_dir, model_seed, idxs: list[int]):
    # load the idx's
    fnames = _get_filenames(Path(data_dir, 'outputs'))
    fnames = _filter_by_model_seed(fnames, model_seed)
    fnames = _sort_fnames_by_epoch_batch(fnames)
    outputs = []
    for idx in idxs:
        fname = fnames[idx]
        with open(fname, 'rb') as f:
            out = pkl.load(f)
            # TASK_MODELID_epoch-batchnum_{ig, sg, vg}.pkl
            out['batch'] = _get_batchnum(fname)
            out['epoch'] = _get_epoch(fname)
            outputs.append(out)
    return outputs

def load_idxs_from_multiple_outputs(data_dir, model_seeds, idxs: list[int]):
    '''
    :return: list(model(idxs)) if len(idxs)>1 else list(models)
    '''
    fnames = _get_filenames(Path(data_dir, 'outputs'))
    outputs = []

    for seed in model_seeds:
        if '/nn/' in str(data_dir):
            fnames_outputs_filtered = _filter_by_model_seed(fnames, seed)
            fnames_outputs_filtered = _sort_fnames_by_epoch_batch(fnames_outputs_filtered)
        else:
            # sklearn
            fnames_models = [str(f) for f in Path(data_dir, 'models').iterdir()]
            fnames_outputs_filtered = [fname for fname in fnames_models if fname.split('_')[2] in model_seeds]

        if len(idxs) > 1:
            outputs.append([])
        for idx in idxs:
            fname = fnames_outputs_filtered[idx]
            with open(fname, 'rb') as f:
                out = pkl.load(f)
                out_dist = out['output_distribution']
                if type(out_dist) == torch.tensor:
                    if 'cpu' not in out['output_distribution'].device.type:
                        out['output_distribution'] = out['output_distribution'].to('cpu')
                # TASK_MODELID_epoch-batchnum_{ig, sg, vg}.pkl
                out['batch'] = _get_batchnum(fname)
                out['epoch'] = _get_epoch(fname)
                if len(idxs) > 1:
                    outputs[-1].append(out)
                else:
                    outputs.append(out)
    return outputs


def load_explanations_by_idxs(data_dir, model_seed, idxs, explanation_method='sg'):
    fnames = _get_filenames(Path(data_dir, 'explanations'))
    fnames = _filter_by_model_seed(fnames, model_seed)
    fnames = _filter_by_explanation_method(fnames, explanation_method)
    fnames = _sort_fnames_by_epoch_batch(fnames)
    explanations = []
    epoch_batch = []
    for idx in idxs:
        fname = fnames[idx]
        with open(fname, 'rb') as f:
            explanations.append(pkl.load(f))
            # TASK_MODELID_epoch-batchnum_{ig, sg, vg}.pkl
            epoch_batch.append((_get_epoch(fname), _get_batchnum(fname)))

    return explanations, epoch_batch

def load_idxs_from_multiple_explanations(data_dir, model_seeds, idxs: list[int], explanation_method='ig', train=False):

    if train:
        fnames = _get_filenames(Path(data_dir, 'explanationsTrain'))
    else:
        fnames = _get_filenames(Path(data_dir, 'explanations'))

    b_sklearn = '/nn/' not in str(data_dir)
    explanations = []
    epoch_batch = []

    for seed in model_seeds:
        fnames_seed = _filter_by_model_seed(fnames, seed, sklearn=b_sklearn)
        fnames_expl = _filter_by_explanation_method(fnames_seed, explanation_method)
        if not b_sklearn:
            fnames_expl = _sort_fnames_by_epoch_batch(fnames_expl)

        if len(idxs) > 1:
            explanations.append([])
            epoch_batch.append([])
        for idx in idxs:
            fname = fnames_expl[idx]
            with open(fname, 'rb') as f:
                if not b_sklearn:
                    if len(idxs) > 1:
                        explanations[-1].append(pkl.load(f))
                        epoch_batch[-1].append((_get_epoch(fname), _get_batchnum(fname)))
                    else:
                        explanations.append(pkl.load(f))
                        epoch_batch.append(
                            (_get_epoch(fname), _get_batchnum(fname))
                        )
                else:
                    if len(idxs) > 1:
                        explanations[-1].append(pkl.load(f))
                    else:
                        explanations.append(pkl.load(f))


    return explanations, epoch_batch


def load_explanations(data_dir, model_seed, explanation_method='sg', modelclass='nn', has_epoch_batch=True):
    explanations = []
    epoch_batch = []
    fnames = _get_filenames(Path(data_dir, 'explanations'))
    if len(fnames) == 0:
        print('expls not found')
    fnames = _filter_by_model_seed(fnames, model_seed)
    if len(fnames) == 0:
        print(f'expls for seed {model_seed} not found')
    fnames = _filter_by_explanation_method(fnames, explanation_method)
    if len(fnames) == 0:
        print(f'expls for seed {model_seed} and method {explanation_method} not found')
    if modelclass == 'nn' and has_epoch_batch: #
        fnames = _sort_fnames_by_epoch_batch(fnames)
    if len(fnames) == 0:
        print("breakpoint")

    for fname in fnames:
        with open(fname, 'rb') as f:
            explanations.append(pkl.load(f))
            # TASK_MODELID_epoch-batchnum_{ig, sg, vg}.pkl
            if modelclass == 'nn' and has_epoch_batch:
                epoch_batch.append((_get_epoch(fname), _get_batchnum(fname)))

    return explanations, epoch_batch


def load_meta_data(data_dir, just_one=False):
    fnames = _get_filenames(Path(data_dir))
    meta_data = {}
    for fname in fnames:
        if 'meta_data' in fname:
            #print(f'loading {fname}')
            dataseed = fname.split('_')[-1].split('.')[0]
            with open(fname, 'rb') as md:
                data = pkl.load(md)
                if just_one:
                    return data
                else:
                    meta_data[dataseed] = data

    return meta_data


def _load_meta_data_by_pid(data_dir):
    fnames = _get_filenames(Path(data_dir))
    meta_data = {}
    for fname in fnames:
        if 'meta_data' in fname:
            #print(f'loading {fname}')
            pid = fname.split('_')[2]
            with open(fname, 'rb') as md:
                data = pkl.load(md)
                meta_data[pid] = data
    return meta_data

def get_top_k_models(data_dir, k=10):
    accs = load_accuracies(data_dir=data_dir)
    res = []
    for id, acc in accs.items():
        res.append((id, acc[-1]))
    res = sorted(res, key=lambda x: (x[1], x[0]), reverse=True)
    return res[:k]

def dump_pkl(data, fname, mode='pkl'):
    if type(fname) == Path:
        fname = str(fname)
    if 'pkl' in mode:
        with open(fname, 'wb') as f:
            pkl.dump(data, f)
    elif 'json' in mode:
        with open(fname, "w") as f:
            json.dump(data, f, cls=NumpyEncoder)
    else:
        raise ValueError(f"dump_pkl: invalid value for mode. val: {mode}")


def load_pkl(fname, mode='pkl'):
    if type(fname) == Path or type(fname) == pathlib.PosixPath:
        fname = str(fname)
    if fname.endswith('.pkl'):
        with open(fname, 'rb') as f:
            data = pkl.load(f)
    elif fname.endswith('.json'):
        with open(fname, "r") as f:
            data = json.load(f)
    else:
        raise ValueError(f"load_pkl: invalid file type '{fname.split('.')[-1]}', must be pkl or json")
    return data

# def _load_pkls(fnames):
#     pkls = []
#     for fname in fnames:
#         with open(fname, 'rb') as f:
#             pkls.append(pkl.load(f))
#     return pkls

def print_make_classification_config(cfg):
    n_feat, n_info, n_redu, n_rep, n_classes = cfg['n_features'], cfg['n_informative'], cfg['n_redundant'], cfg[
        'n_repeated'], cfg['n_classes']
    n_uninf = n_feat - n_rep - n_redu - n_info
    print(f"{n_feat} ({n_info}, {n_redu}, {n_rep}, {n_uninf}):")


def make_classification_config_to_string(cfg) -> str:
    n_samples, n_feat, n_info, n_redu, n_rep, n_classes, n_cpc, flip_y = (cfg['n_samples'], cfg['n_features'],
                                                                          cfg['n_informative'], cfg['n_redundant'],
                                                                          cfg['n_repeated'], cfg['n_classes'],
                                                                          cfg['n_clusters_per_class'], cfg['flip_y'])
    return f"{n_samples}-{n_feat}-{n_info}-{n_redu}-{n_rep}-{n_classes}-{n_cpc}-{flip_y}"

def get_classification_config_from_fname(fname):
    '''classification - n_samples - n_features - n_informative - n_redundant - n_repeated - n_classes - n_clusters_per_class - flip_y_modelseed_dataseed_epoch - batch.ckpt
    ->     eg: classification - 10000 - 20 - 10 - 0 - 10 - 4 - 1 - 0.01_391302_11880_14 - 249.
    ckpt'''
    fname = fname.split('-')
    cfg = {
        'n_samples': int(fname[1]),
        'n_features': int(fname[2]),
        'n_informative': int(fname[3]),
        'n_redundant': int(fname[4]),
        'n_repeated': int(fname[5]),
        'n_classes': int(fname[6]),
        'n_clusters_per_class': int(fname[7]),
        'flip_y': float(fname[8].split('_')[0]),
    }
    # fname = fname[-2:]  # "flipY_modelseed_dataseed_batch-epoch.ckpt
    # _, _str_modelseed, _str_dataseed, _str_batch = fname[0].split('_')[1:]
    # _str_epoch, _ = fname[1].split('.')
    # epoch = -1 if _str_epoch == 'EPOCH' else int(_str_epoch)
    # batch = -1 if _str_batch == 'BATCH' else int(_str_batch)
    #
    # cfg['epoch'] = epoch
    # cfg['batch'] = batch
    # cfg['modelseed'] = int(_str_modelseed)
    # cfg['dataseed'] = int(_str_dataseed)

    return cfg

def get_hypercube_config_from_fname(fname):
    '''    # hypercube - n_samples - n_informative - n_constant - n_redundant - n_repeated - flip_y_modelseed - std
    ckpt'''
    fname = fname.split('-')
    cfg = {
        'n_samples': int(fname[1]),
        'n_informative': int(fname[2]),
        'n_constant': int(fname[3]),
        'n_redundant': int(fname[4]),
        'n_repeated': int(fname[5]),
        'flip_y': float(fname[6]),
        'std': float(fname[7].split('_')[0]),
    }
    return cfg

def _get_outputs(inference_fn, data, model, device, batch_size=512):

    _data = DataLoader(TensorDataset(data), shuffle=False, batch_size=batch_size)

    try:
        _y_out = []
        for x in _data:
            _y = inference_fn(x[0].to(device))
            _y_out.append(_y.cpu())
        return torch.vstack(_y_out)
    except RuntimeError as re:
        if "CUDA out of memory" in str(re):
            model.to('cpu')
            outputs = _get_outputs(inference_fn, data, model, 'cpu')
            model.to('cuda')
            return outputs
        else:
            raise re


def _get_targets(inference_fn, data, model, device):
    return torch.argmax(_get_outputs(inference_fn, data, model, device), dim=1)


## ANCHORS

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

def anchors_to_rule_model(anchors, n_classes):
    rules = [anchor_to_rule(a) for a in anchors]
    rules_classes = []
    for c in range(n_classes):
        rules_classes.append([])
        rc = [r[1] for r in rules if r[0] == c]
        rules_classes[-1].extend(rc)
    dnf = DNFClassifier(rules_classes)
    dnf._meta_information = "ANCHORS"
    return dnf


def anchors_dataset_simple(x, y, n_classes):
    data = np.array(x)
    n_dims = data.shape[1]
    categorical_features = []
    ordinal_features = list(range(n_dims))
    feature_names = [f'V{i}' for i in range(n_dims)]
    labels = np.array(y)
    d = dict()
    d['categorical_features'] = categorical_features
    d['categorical_names'] = {}
    d['class_names'] = [str(i) for i in range(n_classes)]  # might need to be bytes / b''
    d['feature_names'] = feature_names
    d['ordinal_features'] = ordinal_features
    d['data'] = data
    d['train'] = None
    d['labels_train'] = None
    d['test'] = data
    d['labels_test'] = labels
    d['validation'] = None
    d['labels_validation'] = None

    anchors_dataset = Bunch(d)

    return anchors_dataset

def anchors_dataset(x, y, tr_idx, te_idx, validation_idx, n_classes, n_dims, discretize=False):

    data = np.array(x)
    categorical_features = []
    ordinal_features = list(range(n_dims))
    feature_names = [f'V{i}' for i in range(n_dims)]
    if discretize:
        raise NotImplementedError
        # discretizer = QuartileDiscretizer(data, categorical_features, feature_names)
        # data_discretized = discretizer.discretize(data)
        # ordinal_features = [x for x in range(data_discretized.shape[1])
        #                     if x not in categorical_features]
        # categorical_features = list(range(data_discretized.shape[1]))
        # data = QuartileDiscretizer(data, categorical_features, feature_names)
    labels = np.array(y)

    d = dict()
    d['categorical_features'] = categorical_features
    d['categorical_names'] = {}
    d['class_names'] = [str(i) for i in range(n_classes)]  # might need to be bytes / b''
    d['feature_names'] = feature_names
    d['ordinal_features'] = ordinal_features
    d['data'] = data
    d['train'] = data[tr_idx]
    d['labels_train'] = labels[tr_idx]
    d['test'] = data[te_idx]
    d['labels_test'] = labels[te_idx]
    d['validation'] = data[validation_idx]
    d['labels_validation'] = labels[validation_idx]

    anchors_dataset = Bunch(d)

    return anchors_dataset


