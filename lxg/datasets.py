import warnings
import itertools
import os
import random

import torch
import torchvision
import numpy as np
from sphinx.util import requests

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, Sampler
from torch.nn.utils.rnn import pad_sequence
from torchdata.datapipes.iter import Shuffler, Sampler
from torchtext.data.utils import get_tokenizer
from torchtext.data.functional import to_map_style_dataset
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import DBpedia, AG_NEWS, YelpReviewFull

from sklearn.datasets import \
    make_classification, fetch_covtype, load_iris, load_breast_cancer, load_wine
from sklearn.preprocessing import StandardScaler

from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

# ----------------------------------------------------------------------------------------------------------------------

DATA_ROOT = './datasets'

# ----------------------------------------------------------------------------------------------------------------------

class NumpyRandomSeed(object):
    """
    Class to be used when opening a with clause. On enter sets the random seed for numpy based sampling, restores previous state on exit
    """
    def __init__(self, seed):
        self.seed = seed
        self.prev_random_state = None

    def __enter__(self):
        self.prev_random_state = np.random.get_state()
        np.random.seed(self.seed)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        np.random.set_state(self.prev_random_state)


class TorchRandomSeed(object):
    """
    Class to be used when opening a with clause. On enter sets the random seed for torch based sampling, restores previous state on exit
    """
    def __init__(self, seed):
        self.seed = seed
        self.prev_random_state = None

    def __enter__(self):
        self.prev_random_state = torch.get_rng_state()
        torch.set_rng_state(torch.manual_seed(self.seed).get_state())

    def __exit__(self, exc_type, exc_value, exc_traceback):
        torch.set_rng_state(self.prev_random_state)

# ----------------------------------------------------------------------------------------------------------------------

def _train_test_val_split(X, Y, splits=(50, 20, 30), random_state=42):
    splits = np.array(splits)
    splits = splits/np.sum(splits)
    with NumpyRandomSeed(random_state):
        nr_samples = len(X)
        split_tr = int(np.ceil(splits[0] * nr_samples))
        split_te = split_tr + int(np.ceil(splits[1] * nr_samples))
        split_val = split_tr + split_te + int(np.ceil(splits[2] * nr_samples))
        idxs = np.arange(nr_samples)
        np.random.shuffle(idxs)
        # x train, x test, x val, y train, y test, y val
    return (X[idxs[:split_tr]], X[idxs[split_tr:split_te]], X[idxs[split_te:]],
            Y[idxs[:split_tr]], Y[idxs[split_tr:split_te]], Y[idxs[split_te:]])

def _train_test_split(X, y, train_size=0.5, random_state=42):
    """
    Randomly splits data into train and test sets, keeping 100*train_size percent as training data
    :param X: array with training data, first dim is sample
    :param y: array of labels
    :param train_size: float, (0, 1], how much of the data is used for the training set
    :param random_state: Seed/ Random State used for shuffling
    :return: X_train, X_test, y_train, y_test
    """
    with NumpyRandomSeed(random_state):
        nr_samples = len(X)
        split = int(train_size * nr_samples)
        assert split <= nr_samples
        idxs = np.arange(nr_samples)
        np.random.shuffle(idxs)
        # x train, x test, x val, y train, y test, y val
    return X[idxs[:split]], X[idxs[split:]], y[idxs[:split]], y[idxs[split:]]


def _return_dataset(data: np.ndarray, target: np.ndarray, batch_size, train_size: float,
                    as_torch, random_state, split_validation=True, splits=(50, 20, 30)):
    """
    train-test-split sklearn datasets and return as DataLoader or as numpy arrays
    :param data: data in a numpy array
    :param target: labels in numpy array
    :param batch_size: size of the batches the DataLoader returns
    :param train_size: float in (0, 1], share of data used for training data
    :param as_torch: bool, if True return DataLoader, else return
    :param random_state:
    :return: Dataloder/ tuple(array, array) training set, Dataloder/ tuple(array, array) test set, int dimensionality of data, int number of classes
    """
    if split_validation:
        X_tr, X_te, X_val, Y_tr, Y_te, Y_val = _train_test_val_split(data, target, random_state=random_state, splits=splits)
    else:
        X_tr, X_te, Y_tr, Y_te = _train_test_split(data, target, train_size=train_size, random_state=random_state)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    if split_validation:
        X_val = scaler.transform(X_val)
    n_dim = X_tr.shape[1]
    n_classes = len(np.unique(Y_tr))

    assert np.all(np.unique(Y_tr) == np.unique(Y_te))

    if not as_torch:
        if split_validation:
            _cutoff_val_idx = min([len(X_val), batch_size[-1]])
            X_val, Y_val = X_val[:_cutoff_val_idx], Y_val[:_cutoff_val_idx]
            return (X_tr, Y_tr), (X_te, Y_te), (X_val, Y_val), n_dim, n_classes
        else:
            return (X_tr, Y_tr), (X_te, Y_te), n_dim, n_classes

    else:
        # gen_train = torch.Generator('cpu'); gen_train.manual_seed(random_state)
        # gen_test = torch.Generator('cpu'); gen_test.manual_seed(random_state)
        gen_train, gen_test = None, None
        train_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(Y_tr).long()),
                                  shuffle=True, batch_size=batch_size[0], generator=gen_train)
        test_loader = DataLoader(TensorDataset(torch.from_numpy(X_te).float(), torch.from_numpy(Y_te).long()),
                                 shuffle=False, batch_size=min([len(X_te), batch_size[-1]]), generator=gen_test)
        if split_validation:
            val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(Y_val).long()),
                                 shuffle=False, batch_size=min([len(X_val), batch_size[-1]]), generator=gen_test)
            return train_loader, test_loader, val_loader, n_dim, n_classes
        else:
            return train_loader, test_loader, n_dim, n_classes

def _loader_to_numpy(dataloader):
    X, Y = [], []
    for x, y in dataloader:
        x = x.detach().numpy()
        y = y.detach().numpy()
        X.append(x)
        Y.append(y)
    X = np.vstack(X)
    Y = np.hstack(Y)

    return (X, Y)

def _wrap_numpy_to_loader(X, Y, batch_size=32, to_one_hot=False, shuffle=False):

    Y = torch.from_numpy(Y).long()
    X = torch.from_numpy(X).float()
    if to_one_hot:
        Y = torch.nn.functional.one_hot(Y)
    loader = DataLoader(TensorDataset(X, Y), shuffle=shuffle, batch_size=batch_size, generator=None)
    return loader
def _make_waveform(n=300, random_state=None, batch_sizes=None):
    """ create waveform data """
    if random_state is None:
        npr = np.random.RandomState(1234)
    elif isinstance(random_state, int):
        npr = np.random.RandomState(random_state)
    elif isinstance(random_state, np.random.RandomState):
        npr = random_state
    else:
        raise Exception('Unknown input type of random state:', type(random_state))

    y = npr.randint(3, size=n)
    u = npr.uniform(size=n)
    x = npr.normal(size=(n, 21))
    m = np.arange(1, 22)
    h1 = np.maximum(0, 6 - np.abs(m - 7))
    h2 = np.maximum(0, 6 - np.abs(m - 15))
    h3 = np.maximum(0, 6 - np.abs(m - 11))
    x[y == 0] += np.outer(u[y == 0], h1) + np.outer(1 - u[y == 0], h2)
    x[y == 1] += np.outer(u[y == 1], h1) + np.outer(1 - u[y == 1], h3)
    x[y == 2] += np.outer(u[y == 2], h2) + np.outer(1 - u[y == 2], h3)

    return x, y


def _get_vocab(name='AG_NEWS', train_iter=None):
    vocab_path = f"{DATA_ROOT}/{name}/vocab.torch"
    print(f"looking for vocab in {vocab_path}")
    try:
        vocab = torch.load(vocab_path)
    except FileNotFoundError:
        print("Vocab not found, building vocab ...")
        tokenizer = get_tokenizer('basic_english')
        def yield_tokens(data_iter):
            for _, text in data_iter:
                yield tokenizer(text)
        vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=['<unk>', '<pad>'])
        vocab.set_default_index(vocab['<unk>'])
        torch.save(vocab, vocab_path)
        print(f"... done, saved vocab of {len(vocab)} words in {vocab_path}")
    return vocab


def _build_collate_fn(vocab, label_pipeline):
    """
    given a text dataset, returns preprocessing function as required for DataLoader
    :param train_iter: iterator over text dataset
    :return:
    """
    tokenizer = get_tokenizer('basic_english')
    padding_val = vocab['<pad>']

    # label_pipeline = lambda x: int(x) - 1
    # text_pipeline = lambda x: vocab(tokenizer(x))
    def text_pipeline(input):
        return vocab(tokenizer(input))

    def collate_batch(batch):
        label_list, text_list, offsets = [], [], [0]
        for (_label, _text) in batch:
            label_list.append(label_pipeline(_label))
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            # offsets.append(processed_text.size(0))
        labels = torch.tensor(label_list, dtype=torch.int64)
        # offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        # text_list = torch.cat(text_list)
        text = pad_sequence(text_list, batch_first=True, padding_value=padding_val)
        return text, labels #, offsets

    return collate_batch, len(vocab)

# ----------------------------------------------------------------------------------------------------------------------


"""
    Functions that return a torch DataLoader that wraps the respective dataset.
    Datasets loaded via sklearn can also be returned as numpy arrays.
"""

# --------------- SKLEARN


def get_covtype(random_state, batch_sizes=(64, 1050), train_size=0.8, as_torch=True):
    # dim = 54, classes = 7
    covertype_bunch = fetch_covtype(data_home=DATA_ROOT)
    data, target = covertype_bunch.data, covertype_bunch.target
    target -= 1  # covtype targets start with 1 not 0
    return _return_dataset(data, target, batch_sizes, train_size, as_torch=as_torch, random_state=random_state)


def get_iris(random_state=42, batch_sizes=(32, 450), train_size=0.8, as_torch=True):
    # dim = 4, classes = 3
    data, target = load_iris(return_X_y=True)
    return _return_dataset(data, target, batch_sizes, train_size, as_torch=as_torch, random_state=random_state,
                           splits=(50,25,25))


def get_wine(random_state=42, batch_sizes=(32, 450), train_size=0.8, as_torch=True):
    # dim = 13, classes = 3
    data, target = load_wine(return_X_y=True)
    return _return_dataset(data, target, batch_sizes, train_size, as_torch=as_torch, random_state=random_state,
                           splits=(50,20,30))


def  get_breast_cancer(random_state, batch_sizes=(64, 300), train_size=0.8, as_torch=True):
    # dim = 30, classes =2
    data, target = load_breast_cancer(return_X_y=True)
    return _return_dataset(data, target, batch_sizes, train_size, as_torch=as_torch, random_state=random_state)


def get_classification(nr_samples=10000, random_state_data=6810267, random_state_split=185619,
                       batch_sizes=(64, 450), train_size=0.6, as_torch=True, random_state=None,
                       kwargs=None):
    if kwargs is None:
        kwargs = dict(n_samples=10_000, n_features=20, n_informative=2,
                               n_redundant=2, n_classes=2, n_repeated=0,
                      n_clusters_per_class=1, flip_y=0.01, class_sep=3.)
    # shuffling shuffles rows _and columns_
    # no shuffling the dimensinos are stacked [n_info, n_redund, n_rep] followed by noisy dims]
    X, y = make_classification(class_sep=.75, shuffle=False, random_state=random_state_data, **kwargs)
    return _return_dataset(X, y, batch_sizes, train_size, as_torch=as_torch, random_state=random_state_split)

def _make_classification_get_repeated_mapping(X, n_informative):
    mapping = []
    X_inf = X[:, :n_informative]
    X_else = X[:, n_informative:]
    for i in range(n_informative):
        x_inf = X_inf[:, i]
        for j, x_else in enumerate(X_else.T):
            if np.all(x_inf == x_else):
                mapping.append( (i, n_informative+j) )
    if len(mapping) == 0:
        mapping = None
    return mapping

def _make_classification_get_coefficients_redundant_dims(X, n_informative, n_redundant):
    if len(X) >= 5000:
        idxs = np.arange(len(X))
        np.random.shuffle(idxs)
        _X = X[idxs[:5000]]
    else:
        _X = X
    X_inf = _X[:, :n_informative]
    X_red = _X[:, n_informative:n_informative+n_redundant]
    _XTX_inf_inv = np.linalg.inv(X_inf.T @ X_inf)
    Y = _XTX_inf_inv @ X_inf.T @ X_red

    mae = np.mean(abs((X[:, :n_informative] @ Y) -
                      X[:, n_informative:n_informative+n_redundant]))
    if not np.isclose(mae, 0):
        warnings.warn(f"Approximation of coefficient MAE = {mae}", RuntimeWarning)

    return Y

def _make_classification_get_centroids(X, Y, n_informative, n_classes=None, n_clusters_per_class=1, class_sep=0.75):
    if n_classes is None:
        n_classes = max(Y)+1

    # if n_informative == 2**(n_classes*n_clusters_per_class):
    #     centroids = np.zeros((2**n_informative, n_informative))
    #     width = n_informative
    #     for i in range(2**n_informative):
    #         b = np.binary_repr(i)
    #         for j, c in enumerate(b):
    #             centroids[i, j] = 0. if c=='0' else 1.
    #
    # else:
    _k = n_classes * n_clusters_per_class

    _X_inf = X[:, :n_informative] # assume informative features are the first dimensions

    from sklearn.cluster import KMeans
    KM = KMeans(n_clusters=_k,)

    _cluster_assignment = KM.fit_predict(_X_inf)
    _cluster_labels = []
    for _cluster_id in range(KM.n_clusters):
        _cluster_data_idxs = np.argwhere(_cluster_assignment == _cluster_id).squeeze()
        # majority vote of class labels of data points within cluster number _cluster_id
        _cluster_label = np.argmax(np.bincount(Y[_cluster_data_idxs]))
        _cluster_labels.append(_cluster_label)


    _centers = KM.cluster_centers_
    # binarize to obtain 'hypercube vertices'
    _centers = 1. * (_centers < class_sep*0.85)

    # analogous to data generation
    # https://github.com/scikit-learn/scikit-learn/blob/093e0cf14aff026cca6097e8c42f83b735d26358/sklearn/datasets/_samples_generator.py#L249
    _centers *= 2*class_sep
    _centers -= class_sep

    return [(label, centroid) for (label, centroid) in zip(_cluster_labels, _centers)]


# --------------- TORCH

# ---------------------- VISION

def get_fmnist(random_state, batch_sizes=(64,512), root=DATA_ROOT):
    """load Fashion MNIST from torchvision"""
    # dim = 28x28, classes = 10
    with TorchRandomSeed(random_state):

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,))
        ])
        data = torchvision.datasets.FashionMNIST

        train = data(root, train=True, download=True, transform=transform)
        test = data(root, train=False, download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_sizes[0], shuffle=True)
        test_loader = torch.utils.data.DataLoader(test, batch_size=batch_sizes[-1], shuffle=False)

    return train_loader, test_loader, (28, 28), 10

def get_emnist(random_state, batch_sizes=(64, 512), root=DATA_ROOT):
    """load extended-MNIST from torchvision"""
    with TorchRandomSeed(random_state):

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,))
        ])
        data = torchvision.datasets.EMNIST

        train = data(root, split='balanced', train=True, download=True, transform=transform)
        test = data(root, split='balanced', train=False, download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_sizes[0], shuffle=True)
        test_loader = torch.utils.data.DataLoader(test, batch_size=batch_sizes[-1], shuffle=False)

    return train_loader, test_loader, (28, 28), 47


# ---------------------- NLP

def get_agnews(random_state, batch_sizes=(64, 200), root=DATA_ROOT):
    """load AGNews dataset from torchtext"""
    # -, classes = 4
    def label_pipeline(input):
        return int(input) - 1

    with TorchRandomSeed(random_state):
        # base vocab in collate function on training data
        gen_train = torch.Generator(); gen_train.manual_seed(random_state)
        gen_test = torch.Generator(); gen_test.manual_seed(random_state)

        train_iter = AG_NEWS(root, split='train')
        train_iter = to_map_style_dataset(train_iter)

        test_iter = AG_NEWS(root, split='test')
        test_iter = to_map_style_dataset(test_iter)

        vocab = _get_vocab('AG_NEWS', train_iter)
        collate_batch, size_vocab = _build_collate_fn(vocab, label_pipeline)

        train_loader = DataLoader(train_iter, batch_size=batch_sizes[0],
                                  shuffle=True, collate_fn=collate_batch, generator=gen_train)
        test_loader = DataLoader(test_iter, batch_size=batch_sizes[-1], shuffle=True, collate_fn=collate_batch,
                                 generator=gen_test)

    return train_loader, test_loader, size_vocab, 4


def get_dbpedia(random_state, batch_sizes=(64, 128), root=DATA_ROOT):
    """load DBPedia dataset from torchtext"""
    # -, classes = 14
    if not os.path.exists(DATA_ROOT+'/DBpedia/dbpedia_csv.tar.gz'):
        print(f"DBpedia dataset not found. In case of error, download manually from\n"
              f"https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k\n"
              f"and place dbpedia_csv.tar.gz in root/DBpedia")

    def label_pipeline(input):
        return int(input)-1

    with TorchRandomSeed(random_state):
        # base vocab in collate function on training data
        train_iter = DBpedia(root, split='train')
        train_iter = to_map_style_dataset(train_iter)
        test_iter = DBpedia(root, split='test')
        test_iter = to_map_style_dataset(test_iter)

        vocab = _get_vocab('DBpedia', train_iter)
        collate_batch, size_vocab = _build_collate_fn(vocab, label_pipeline,)

        train_loader = DataLoader(train_iter, batch_size=560000, shuffle=True, collate_fn=collate_batch)
        test_loader = DataLoader(test_iter, batch_size=batch_sizes[-1], shuffle=True, collate_fn=collate_batch)

    return train_loader, test_loader, size_vocab, 14

def get_yelpreviewfull(random_state, batch_sizes=(64, 128), root=DATA_ROOT):
    """load YelpReviewFull datasedt from torchtext"""
    # -, classes = 5
    if not os.path.exists(DATA_ROOT+'/YelpReviewFull/yelp_review_full_csv.tar.gz'):
        print(f"YelpReviewFull dataset not found. In case of error, download manually from\n"
              f"https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbNUpYQ2N3SGlFaDg\n"
              f"and place yelp_review_polarity_csv.tar.gz in project_root/YelpReviewFull")

    def label_pipeline(input):  # think this doesnt work?
        return int(input)-1

    with TorchRandomSeed(random_state):
        # base vocab in collate function on training data
        train_iter = YelpReviewFull(root, split='train')
        train_iter = to_map_style_dataset(train_iter)

        test_iter = YelpReviewFull(root, split='test')
        test_iter = to_map_style_dataset(test_iter)

        vocab = _get_vocab('YelpReviewFull', train_iter)
        collate_batch, size_vocab = _build_collate_fn(vocab, label_pipeline)

        train_loader = DataLoader(train_iter, batch_size=batch_sizes[0], shuffle=True, collate_fn=collate_batch)
        test_loader = DataLoader(test_iter, batch_size=batch_sizes[-1], shuffle=True, collate_fn=collate_batch)

    return train_loader, test_loader, size_vocab, 5


# --------------- MISC


def get_waveform(nr_samples=10000, random_state_data=10419187, random_state_split=185619,
                       batch_sizes=(64, 450), train_size=0.8, as_torch=True, random_state=None):
    X, y = _make_waveform(nr_samples, random_state_data)
    return _return_dataset(X, y, batch_sizes, train_size, as_torch=as_torch, random_state=random_state_split)


def get_ionosphere(random_state, batch_sizes=(64, 300), train_size=0.8, as_torch=True, root=DATA_ROOT):
    """load ionosphere dataset from DATA_ROOT subfolder"""
    pth = root+'/IONOSPHERE/ionosphere.data'

    X = np.genfromtxt(pth, delimiter=',', usecols=np.arange(34))
    _y = np.genfromtxt(pth, delimiter=',', dtype=str, usecols=[34])
    y = np.array([0 if l == 'g' else 1 for l in _y])
    return _return_dataset(X, y, batch_sizes, train_size, as_torch=as_torch, random_state=random_state)

def get_beans(random_state, batch_sizes=(32, 1050), train_size=0.8, as_torch=True, root=DATA_ROOT):
    """load DryBeansDataset dataset from DATA_ROOT subfolder"""
    # 16, 7
    pth = root+'/DryBeansDataset/Dry_Beans_Dataset.csv'
    try:
        X = np.genfromtxt(pth, delimiter=';', dtype=float, usecols=np.arange(16), skip_header=1)
        _y = np.genfromtxt(pth, delimiter=';', dtype=str, usecols=[16], skip_header=1)
    except OSError as o:
        print(o)
        print(f"Get Dry Beans Dataset at https://archive-beta.ics.uci.edu/ml/datasets/dry+bean+dataset"
              f"Convert Excel to ;-separated CSV, replace comma in numbers by dot"
              f"Place in {pth}")
        exit()
    classmap = {c: n for n, c in enumerate(np.unique(_y))}
    y = np.array([classmap[c] for c in _y])

    return _return_dataset(X, y, batch_sizes, train_size, random_state=random_state, as_torch=as_torch)



# --------------- DATAVERSE SETS WITH _PREDETERMINED_ TEST-SPLITS

def _return_dataverse(X_tr_orig, Y_tr_orig, X_te, Y_te, batch_sizes, as_torch=True, stratified_sampling=False):

    n_samples_tr = len(X_tr_orig)
    n_samples_te = len(X_te)
    tr_te_ratio = n_samples_tr/n_samples_te
    split = min(0.25, tr_te_ratio)
    split_idx = int(n_samples_tr * (1-split))
    with NumpyRandomSeed(42):
        # create validation set deterministically
        _idxs = np.arange(n_samples_tr)
        np.random.shuffle(_idxs)
        _idxs_tr, _idxs_val = _idxs[:split_idx], _idxs[split_idx:]
        n_samples_tr = len(_idxs_tr)
        X_tr = X_tr_orig[_idxs_tr]
        X_val = X_tr_orig[_idxs_val]
        Y_tr = Y_tr_orig[_idxs_tr]
        Y_val = Y_tr_orig[_idxs_val]

    # normalizing
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)
    X_te = scaler.transform(X_te)


    n_dims = X_tr_orig.shape[1]
    counts = np.bincount(np.array(Y_tr, dtype=np.int64), minlength=int(max(Y_tr)+1))
    n_classes = len(counts)
    assert n_classes == len(np.unique(Y_te)) == len(np.unique(Y_val))

    if as_torch:
        # use stratified sampling for training loader because we have highly imbalanced datasets
        sampler = None
        if stratified_sampling:
            target_ratio = 100./n_classes
            ratios = counts / n_samples_tr
            _tr_stratified_weighting = np.zeros(len(Y_tr))
            for c in range(n_classes):
                selector_c = Y_tr==c
                _tr_stratified_weighting[selector_c] = target_ratio/np.sum(selector_c)
                # assert np.isclose(np.sum(_tr_stratified_weighting[selector_c]), target_ratio)
            sampler = torch.utils.data.WeightedRandomSampler(weights=_tr_stratified_weighting, num_samples=len(X_tr),
                                                             replacement=True)
        train_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(Y_tr).long()),
                                shuffle=True if sampler is None else False,
                                  batch_size=batch_sizes[0], sampler=sampler)
        test_loader = DataLoader(TensorDataset(torch.from_numpy(X_te).float(), torch.from_numpy(Y_te).long()),
                                shuffle=False, batch_size=len(X_te))
        val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(Y_val).long()),
                                shuffle=False, batch_size=len(X_val))
        return train_loader, test_loader, val_loader, n_dims, n_classes
    else:
        return (X_tr, Y_tr), (X_te, Y_te), (X_val, Y_val), n_dims, n_classes


def get_heloc(random_state=None, batch_sizes=(32, 1050), as_torch=True, root=DATA_ROOT, stratified_sampling=True):
    # "Home Equity Line Of Credit"
    # numerical, 24d
    # train https://dataverse.harvard.edu/file.xhtml?fileId=8550942&datasetVersionId=374726
    # test https://dataverse.harvard.edu/file.xhtml?fileId=8550943&datasetVersionId=374726
    # first row is column description
    # n_dims = 22 + label
    if random_state is not None:
        print(f"WARNING: randomizing dataverse set has no effect, test and validation splits are fixed ")
    label_dim = -1
    dims_keep = np.arange(24)  # all

    pth_tr = root+'/HELOC/heloc-train.csv'
    pth_te = root+'/HELOC/heloc-test.csv'

    data_tr = np.genfromtxt(pth_tr, delimiter=',', dtype=float, usecols=dims_keep, skip_header=1)
    X_tr, Y_tr = data_tr[:, :-1], data_tr[:, -1]
    data_te = np.genfromtxt(pth_te, delimiter=',', dtype=float, usecols=dims_keep, skip_header=1)
    X_te, Y_te = data_te[:, :-1], data_te[:, -1]
    return _return_dataverse(X_tr, Y_tr, X_te, Y_te, batch_sizes, as_torch, stratified_sampling)


def get_pima(random_state, batch_sizes=(32, 1050), as_torch=True, root=DATA_ROOT, stratified_sampling=True):
    # diabetes data
    # numerical ?
    # train https://dataverse.harvard.edu/file.xhtml?fileId=8550937&version=3.0
    # test https://dataverse.harvard.edu/file.xhtml?fileId=8550938&version=3.0
    # n_dims = 8 + 1
    if random_state is not None:
        print(f"WARNING: randomizing dataverse set has no effect, test and validation splits are fixed ")
    label_dim = -1
    dims_keep = np.arange(9)
    pth_tr = root+'/PIMA/pima-train.csv'
    pth_te = root+'/PIMA/pima-test.csv'
    data_tr = np.genfromtxt(pth_tr, delimiter=',', dtype=float, usecols=dims_keep, skip_header=1)
    X_tr, Y_tr = data_tr[:, :-1], data_tr[:, -1]
    data_te = np.genfromtxt(pth_te, delimiter=',', dtype=float, usecols=dims_keep, skip_header=1)
    X_te, Y_te = data_te[:, :-1], data_te[:, -1]
    return _return_dataverse(X_tr, Y_tr, X_te, Y_te, batch_sizes, as_torch, stratified_sampling)


def get_heart(random_state, batch_sizes=(32, 1050), as_torch=True, root=DATA_ROOT, stratified_sampling=True):
    # heart desease
    # train https://dataverse.harvard.edu/file.xhtml?fileId=8550932&version=3.0
    # test https://dataverse.harvard.edu/file.xhtml?fileId=8550935&version=3.0
    # unbalanced
    # 8/16 feature numerical
    if random_state is not None:
        print(f"WARNING: randomizing dataverse set has no effect, test and validation splits are fixed ")
    label_dim = 15
    dims_keep = [1, 2, 4, 9, 10, 11, 12, 13, 14, 15]
    pth_tr = root+'/HEART/heart-train.csv'
    pth_te = root+'/HEART/heart-test.csv'
    data_tr = np.genfromtxt(pth_tr, delimiter=',', dtype=float, usecols=dims_keep, skip_header=1)
    X_tr, Y_tr = data_tr[:, :-1], data_tr[:, -1]
    data_te = np.genfromtxt(pth_te, delimiter=',', dtype=float, usecols=dims_keep, skip_header=1)
    X_te, Y_te = data_te[:, :-1], data_te[:, -1]
    return _return_dataverse(X_tr, Y_tr, X_te, Y_te, batch_sizes, as_torch, stratified_sampling)


def get_gmsc(random_state, batch_sizes=(32, 1050), as_torch=True, root=DATA_ROOT, stratified_sampling=True):
    # HIGHLY IMBALANCED
    # give me some credit
    # train https://dataverse.harvard.edu/file.xhtml?fileId=8550934&version=3.0
    # test https://dataverse.harvard.edu/file.xhtml?fileId=8550939&version=3.0
    if random_state is not None:
        print(f"WARNING: randomizing dataverse set has no effect, test and validation splits are fixed ")
    label_dim = 0
    # removed dims are technically natural numbers counting some quantity, but all actual values are <=4
    dims_keep = [0, 1, 2, 4, 5, 6]

    pth_tr = root+'/GMSC/gmsc-train.csv'
    pth_te = root+'/GMSC/gmsc-test.csv'
    data_tr = np.genfromtxt(pth_tr, delimiter=',', dtype=float, usecols=dims_keep, skip_header=1)
    X_tr, Y_tr = data_tr[:, 1:], data_tr[:, 0]
    data_te = np.genfromtxt(pth_te, delimiter=',', dtype=float, usecols=dims_keep, skip_header=1)
    X_te, Y_te = data_te[:, 1:], data_te[:, 0]
    return _return_dataverse(X_tr, Y_tr, X_te, Y_te, batch_sizes, as_torch, stratified_sampling)

# ----------------------------------------------------------------------------------------------------------------------

def _get_openml(id, as_array=True):
    data = fetch_openml(data_id=id, data_home=DATA_ROOT, as_frame=True, cache=True)
    X, y, attribute_names = data.data, data.target, data.feature_names
    if not as_array:
        return X, y, attribute_names
    else:
        X = X.to_numpy()

        if type(y.dtype) == pd.CategoricalDtype:
            categories = y.dtype.categories
            _map = {c: i for i, c in zip(np.arange(len(categories)), categories)}
            y = y.to_numpy()
            y = np.array([_map[yc] for yc in y])
        else:
            y = y.to_numpy(dtype=int)

        if min(y) > 0:  # eg spf has classes {1, 2} ..
            y -= min(y)

        return X, y, attribute_names


def get_btsc(random_state=42, batch_sizes=(32, 1050), as_torch=True, root=DATA_ROOT, stratified_sampling=False):
    # blood-transfusion-service-center
    id = 1464
    X, y, attribute_names = _get_openml(id)
    return _return_dataset(X, y, batch_size=batch_sizes, as_torch=as_torch, train_size=-1,
                           random_state=random_state, split_validation=True)


def get_breastw(random_state=42, batch_sizes=(32, 1050), as_torch=True, root=DATA_ROOT, stratified_sampling=False):
    id = 251
    X, y, attribute_names = _get_openml(id)
    return _return_dataset(X, y, batch_size=batch_sizes, as_torch=as_torch, train_size=-1,
                           random_state=random_state, split_validation=True)


def get_spambase(random_state=42, batch_sizes=(32, 1050), as_torch=True, root=DATA_ROOT, stratified_sampling=False):
    id = 44
    X, y, attribute_names = _get_openml(id)
    return _return_dataset(X, y, batch_size=batch_sizes, as_torch=as_torch, train_size=-1,
                           random_state=random_state, split_validation=True)


def get_spf(random_state=42, batch_sizes=(32, 1050), as_torch=True, root=DATA_ROOT, stratified_sampling=False):
    #steel plates fault
    id = 1504
    X, y, attribute_names = _get_openml(id)
    cols_to_drop = set([11, 12, 20, 27, 28, 29, 30, 31, 32])
    # X = X[:, list(cols_to_drop)]  # only keep nominal dimensions -> this info suffices for DT(depth=6) to solve the task perfectly.
    cols_to_keep = list(set(np.arange(33))-cols_to_drop)
    X = X[:, cols_to_keep]
    return _return_dataset(X, y, batch_size=batch_sizes, as_torch=as_torch, train_size=-1,
                           random_state=random_state, split_validation=True)


def get_winequality(random_state=42, batch_sizes=(32, 1050), as_torch=True, root=DATA_ROOT, stratified_sampling=False):
    # the dataset is about ... wine quality.
    # https://www.openml.org/search?type=data&status=active&id=40691
    id = 40691
    X, y, attribute_names = _get_openml(id)
    n_unique_vals_per_dim = [len(np.unique(X[:, i])) for i in range(X.shape[1])]
    n_classes = np.unique(y)
    # keep all dimensions
    return _return_dataset(X, y, batch_size=batch_sizes, as_torch=as_torch, train_size=-1,
                           random_state=random_state, split_validation=True)


def get_diggle(random_state=42, batch_sizes=(32, 1050), as_torch=True, root=DATA_ROOT, stratified_sampling=False,
               splits=(40, 40, 20)):
    # wool prices
    # https://www.openml.org/search?type=data&status=active&id=694
    id = 694
    X, y, attribute_names = _get_openml(id)
    n_unique_vals_per_dim = [len(np.unique(X[:, i])) for i in range(X.shape[1])]
    n_classes = np.unique(y)
    return _return_dataset(X, y, batch_size=batch_sizes, as_torch=as_torch, train_size=-1,
                           random_state=random_state, split_validation=True, splits=splits)

def get_abalone(random_state=42, batch_sizes=(32, 1050), as_torch=True, root=DATA_ROOT, stratified_sampling=False,
                splits=(70, 15, 15)):
    # 3 class variant of abalone dataset, usually contains 28 classes
    # https://www.openml.org/search?type=data&status=active&id=1557
    id = 1557
    X, y, attribute_names = _get_openml(id)
    n_unique_vals_per_dim = [len(np.unique(X[:, i])) for i in range(X.shape[1])]
    n_classes = np.unique(y)
    # filter out 'Sex' feature, first column
    X = X[:, 1:]
    return _return_dataset(X, y, batch_size=batch_sizes, as_torch=as_torch, train_size=-1,
                           random_state=random_state, split_validation=True, splits=splits)

def get_pageblocks(random_state=42, batch_sizes=(32, 1050), as_torch=True, root=DATA_ROOT, stratified_sampling=False):
    # https://www.openml.org/search?type=data&status=active&id=30

    id = 30
    X, y, attribute_names = _get_openml(id)
    n_unique_vals_per_dim = [len(np.unique(X[:, i])) for i in range(X.shape[1])]
    n_classes = np.unique(y)
    # keep all dimensions
    return _return_dataset(X, y, batch_size=batch_sizes, as_torch=as_torch, train_size=-1,
                           random_state=random_state, split_validation=True)


def get_vehicle(random_state=42, batch_sizes=(32, 1050), as_torch=True):
    # https://www.openml.org/search?type=data&status=any&id=54
    id = 54
    X, y, attribute_names = _get_openml(id)
    return _return_dataset(X, y, batch_size=batch_sizes, as_torch=as_torch, train_size=-1,
                           random_state=random_state, split_validation=True)


def get_autouniv(random_state=42, batch_sizes=(32, 1050), as_torch=True):
    # https://www.openml.org/search?type=data&status=active&id=1553
    id = 1553
    X, y, attribute_names = _get_openml(id)
    X = X[:, [0, 1, 2, 6, 10]]
    return _return_dataset(X, y, batch_size=batch_sizes, as_torch=as_torch, train_size=-1,
                           random_state=random_state, split_validation=True)



# ----------------------------------------------------------------------------------------------------------------------


def get_dnf(random_state=None, batch_sizes=(64, 200), root=DATA_ROOT, dnf_id='16-4-2-4', dnf_num=0, as_torch=True):
    num_variables, num_terms, min_len, max_len = [int(s) for s in dnf_id.split('-')]
    fname = 'dnfs_'+dnf_id.replace('-', '_')
    dnf = None
    with open(Path(DATA_ROOT, 'DNFs', fname), 'r') as f:
        for i in range(dnf_num + 1):
            if i < dnf_num:
                next(f)
                continue
            dnf = next(f).replace('"', '')
            dnf = eval(dnf)
            break
    dnf = DNF(dnf)
    X, Y = GenerateDNFData.create_data(dnf, num_variables)
    X = np.vstack(X)
    # mu = np.mean(X, 0)
    # std = np.std(X, 0)
    # X = (X - mu) / std
    Y = np.array(Y, dtype=int)

    if as_torch:
        train_loader = DataLoader(TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).long()),
                                shuffle=True, batch_size=batch_sizes[0])
        test_loader = DataLoader(TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).long()),
                                shuffle=False, batch_size=X.shape[0])

        return train_loader, test_loader, num_variables, 2
    else:
        return (X, Y), (X, Y), num_variables, 2



class DNF:
    def __init__(self, clauses: list[list[tuple[int, int]]]=None):
        if clauses is None:
            self.clauses = []
        else:
            self.clauses = clauses

    def add_clause(self, literals):
        self.clauses.append(literals)

    def evaluate(self, assignment):
        for clause in self.clauses:
            for literal in clause:
                variable, value = literal[0], literal[1]
                if (variable in assignment and assignment[variable] == value) or (variable not in assignment and value):
                    break
            else:
                return False
        return True

    def __str__(self):
        dnf_str = ''
        for clause in self.clauses:
            clause_str = ' '.join(['{}{}'.format(literal[1] and '' or '~', literal[0]) for literal in clause])
            dnf_str += '({}) ∧ '.format(clause_str)
        return dnf_str[:-3]

    def distance(self, other):
        return self.clauses == other.clauses

class GenerateDNFData:
    def __init__(self, num_dnfs, num_variables, num_clauses, min_positive_literals, max_positive_literals, seed=0):
        self.num_dnfs = num_dnfs
        self.num_variables = num_variables
        self.num_clauses = num_clauses
        self.min_positive_literals = min_positive_literals
        self.max_positive_literals = max_positive_literals
        self.seed = seed
        random.seed(seed)
        self.dnf_data = []
        self.generate_dnfs()

    def generate_dnfs(self):
        for _ in range(self.num_dnfs):
            dnf = self.get_next_dnf()
            self.dnf_data.append(dnf)

    def get_next_dnf(self) -> list[DNF]:
        dnf = self.generate_random_dnf(self.num_variables, self.num_clauses, self.min_positive_literals, self.max_positive_literals)
        output = []
        for clause in dnf:
            positive_literals = [index for index in clause if index > 0]
            output.append(positive_literals)
        output.sort()
        print(output)
        dnf = DNF(output)
        return dnf

    @staticmethod
    def create_data(dnf, num_variables):
        images = [GenerateDNFData.num_to_binary_vector(i, num_variables) for i in range(2 ** num_variables)]
        labels = [GenerateDNFData.label(image, dnf) for image in images]
        # labels = [np.count_nonzero(image)>=8 for image in images]
        sorted_lists = sorted(zip(labels, images), key=lambda x: x[0])
        labels, images = zip(*sorted_lists)
        labels = list(labels)
        images = list(images)
        return images, labels

    @staticmethod
    def num_to_binary_vector(num, n):
        # Convert num to binary string
        binary_str = bin(num)[2:]

        # Pad binary string with zeros if necessary
        padded_binary_str = binary_str.zfill(n)

        # Convert padded binary string to NumPy array
        binary_array = np.array(list(padded_binary_str), dtype=int)

        return binary_array

    def generate_binary_samples(self, n, m):
        # Generate list of all possible binary vectors with m^2 entries
        binary_vectors = list(itertools.product([0, 1], repeat=m ** 2))

        # Choose n random samples without duplicates
        samples = random.sample(binary_vectors, n)

        # Reshape each sample as a m x m numpy array
        reshaped_samples = [np.reshape(sample, (m, m)) for sample in samples]

        return reshaped_samples

    # Define the DNF
    @staticmethod
    def label(image: np.ndarray, dnf: DNF) -> bool:
        for clause in dnf.clauses:
            clause_assignment = False
            for literal in clause:
                clause_assignment = image[abs(literal)] == 1
                if not clause_assignment:
                    break
            if clause_assignment:
                return True
        return False

    def generate_random_dnf(self, num_variables, num_clauses, min_positive_literals, max_positive_literals):
        dnf = []
        for _ in range(num_clauses):
            clause = []
            num_positive_literals = random.randint(min_positive_literals, max_positive_literals)
            positive_literal_indices = random.sample(range(1, num_variables + 1), num_positive_literals)
            for i in range(1, num_variables + 1):
                literal = i if i in positive_literal_indices else -i
                clause.append(literal)
            dnf.append(clause)
        return dnf

# --------------- HELPER ---------------

__info_dim_classes = {
    # 'name': (n_dims, n_classes),
    'covtype': (54, 7),
    'iris': (4, 3),
    'wine': (13, 3),
    'breastcancer': (30, 2),
    'classification': (20, 3),
    'beans': (16, 7),
    'bool': (16, 2),

    'abalone': (7, 3),
    'vehicle': (18, 4),
    'diggle': (8, 9),
    'autouniv': (5, 3),

    'fmnist': ((1, 28, 28), 10),
    'emnist': ((1, 28, 28), 47),

    'agnews': (95812, 4),
    'dbpedia': (802999, 14),
    'yelpreviewfull': (519820, 5),

    'waveform': (21, 3),
    'ionosphere': (34, 2),

    'heloc': (23, 2),
    'heart': (9, 2),
    'pima': (5, 2),
    'gmsc': (5, 2),

    'spf': (24, 2),
    'btsc': (4, 2),
    'breastw': (9, 2),
    'spambase': (57, 2)

}

def _info_make_classification(filename=None):
    '''
    eg: classification-10000-20-10-0-10-4-1-0.01_391302_11880_14-249.ckpt
    -> classification-n_samples-n_features-n_informative-n_redundant-n_repeated-n_classes-n_clusters_per_class-flip_y_modelseed_dataseed_epoch-batch.ckpt
    :param filename:
    :return:
    '''
    filename = filename.split('_')[0]  # remove _ suffixes
    str_params = filename.split('-')[1:]  # remove 'classification'
    params = [eval(n) for n in str_params]  # misuse eval to automatically cast ints and floats correctly
    keys = ['n_samples', 'n_features', 'n_informative', 'n_redundant', 'n_repeated', 'n_classes', 'n_clusters_per_class', 'flip_y']
    assert len(params) == len(keys)
    return {k: v for k, v in zip(keys, params)}

def _info_hypercube(filename=None):
    filename = filename.split('_')[0]  # remove _ suffixes
    str_params = filename.split('-')[1:]  # remove 'classification'
    params = [eval(n) for n in str_params]  # misuse eval to automatically cast ints and floats correctly
    keys = ['n_samples', 'n_informative', 'n_constant', 'n_redundant', 'n_repeated', 'flip_y', 'std']
    assert len(params) == len(keys)
    info = {k: v for k, v in zip(keys, params)}
    info['n_classes'] = 2
    return info

def _get_dim_classes(dataset : str):
    if dataset.startswith("classification-"):
        dim, nc = int(dataset.split('-')[2]), int(dataset.split('-')[6])
        return dim, nc
    elif dataset.startswith('hypercube'):
        nc = 2
        dim = sum([int(i) for i in dataset.split('-')[2:6]])
        return dim, nc
    else:
        return __info_dim_classes[dataset]

dataset_callables = {
    # tabular, multi-class
    'covtype': get_covtype,
    'iris': get_iris,
    'wine': get_wine,
    'beans': get_beans,
    'abalone': get_abalone,  # openml
    'diggle': get_diggle,  # openml
    'vehicle': get_vehicle,  # openml


    # tabular, binary
    'breastcancer': get_breast_cancer,
    'heloc': get_heloc,
    'pima': get_pima,
    'heart': get_heart,
    'gmsc': get_gmsc,
    'ionosphere': get_ionosphere,
    'spf': get_spf,
    'btsc': get_btsc,
    'spambase': get_spambase,
    'breastw': get_breastw,

    # synthetic
    'classification': get_classification,
    'bool': get_dnf,
    'waveform': get_waveform,
    'autouniv': get_autouniv,  # openml

    # image, multi-class
    'fmnist': get_fmnist,
    'emnist': get_emnist,

    # text, multi-class
    'agnews': get_agnews,
    # text, binary
    'dbpedia': get_dbpedia,
    'yelpreviewfull': get_yelpreviewfull,

}

def _get_dataset_callable(dataset: str):
    return dataset_callables[dataset]


nlp_tasks = ['agnews', 'dbpedia', 'yelpreviewfull']
cv_tasks = ['fmnist', 'emnist']
dataverse_tasks = ['heart', 'heloc', 'pima', 'gmsc'] # they have a predetermined train/ test split and with our loading function a fixed validation set as well
openml_tasks = ['spf', 'spambase', 'breastw', 'btsc', 'autouniv', 'vehicle', 'abalone', 'diggle']
tabular_tasks = [k for k in dataset_callables.keys()
                 if k not in nlp_tasks and k not in cv_tasks
                 and k not in dataverse_tasks and k not in openml_tasks]  # :)


# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    from sklearn.tree import DecisionTreeClassifier as DT
    from models import make_ff, simple_training
    print(os.getcwd())
    os.chdir('../')

    # _spambase = get_spambase(random_state=11880, as_torch=False,
    #                          batch_sizes=(32, 921))  # no limit, depth->21, beats random, 90% test
    get_btsc(11880)
    _iris = get_iris(random_state=11880)
    _spf = get_spf(as_torch=False, batch_sizes=(32, 389))  # without depth limit tree is not better than random
    _btsc = get_btsc(as_torch=False, batch_sizes=(32, 150))  # without depth limit tree is not better than random
    _breastw = get_breastw(as_torch=False, batch_sizes=(32, 1000))  # no limit, depth->18, beats random, 98% test
    _heloc = get_heloc(as_torch=False, batch_sizes=(32, 1000))
    _iono = get_ionosphere(as_torch=False)
    pass
    get_spambase()
    # TODO:
    #   make sure non-numeric/ non-dense dims are removed
    #   assert data is properly normalized
    #   get train, test, val splits from data -> data distribution ok?
    #   test out NN architectures and trees

    import pandas as pd
    from copy import deepcopy

    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.width', None)  # Auto-detect display width
    pd.set_option('display.float_format', '{:.3f}'.format)  # Set float precision to 3 decimal places


    def to_df(x):
        cols = [f'V{i}' for i in range(x.shape[1])]
        x = pd.DataFrame(deepcopy(x), columns=cols)
        return x
    def print_info(xtr, ytr, xte, yte, xval, yval, nclasses):
        bincount = lambda y: np.bincount(y, minlength=nclasses)/len(y)
        tr = to_df(xtr)
        te = to_df(xte)
        val = to_df(xval)
        print(f"class distributions:")
        print("tr ", bincount(ytr))
        print("te ", bincount(yte))
        print("val", bincount(yval))
        print(f"DATA")
        print(f"\n\ttrain")
        print(tr.describe())
        print(f"\n\ttest")
        print(te.describe())
        print(f"\n\tvalidation")
        print(val.describe())


    def filter_openml():
        import openml
        dt = openml.datasets.list_datasets(output_format='dataframe')
        filtered = dt[
                (2 < dt['NumberOfClasses']) &
                (5 >= dt['NumberOfClasses']) &

                (dt['NumberOfInstances'] > 400) &
                (dt['NumberOfInstances'] < 5000) &
                (dt['MinorityClassSize'] > (0.2*dt['NumberOfInstances'])) &

                (dt['NumberOfFeatures'] >= 5) &
                (dt['NumberOfFeatures'] <= 20) &

                (dt['NumberOfNumericFeatures'] >= 5) &

                (dt['NumberOfInstancesWithMissingValues'] == 0)
            ]
        filtered.describe()
        pass

    def train_tree_nn(callable):
        from sklearn.model_selection import GridSearchCV, ParameterGrid
        from sklearn.tree import DecisionTreeClassifier as DT
        (xtr, ytr), (xte, yte), (_, _), ndims, nclasses = callable(as_torch=False)
        print(f"ndims={ndims}, nclasses={nclasses}")
        print(f"n samples")
        print(f"tr {len(xtr)}, te {len(xte)}")

        param_grid = {'max_depth': range(1, 10)}
        grid_search = GridSearchCV(DT(), param_grid, cv=5, scoring='accuracy')
        # Fit the GridSearchCV object to the data
        grid_search.fit(xtr, ytr)
        dt = grid_search.best_estimator_

        acc_dt = dt.score(xte, yte)

        batch_size = 32
        _torch_tr = _wrap_numpy_to_loader(xtr, ytr, batch_size=batch_size)
        _torch_te = _wrap_numpy_to_loader(xte, yte, batch_size=len(yte))
        _nn_layers = [
            [8, 8],
            [8, 8, 8, 8],
            [32, 32],
            [32, 32, 32, 32],
            [128, 128],
            [128, 128, 128],
            [256, 256]
        ]
        n_epochs = 30
        n_batches = int(min([4000, n_epochs*len(xtr)/batch_size]))
        nn_accs = []
        for layers in _nn_layers:
            _arch = [ndims] + layers + [nclasses]
            nn = make_ff(_arch,)
            accs = simple_training(nn, torch.optim.Adam(nn.parameters()), torch.nn.CrossEntropyLoss(), _torch_tr,
                            next(iter(_torch_te)), n_batches=n_batches, verbose=False)[0]
            nn_accs.append(accs)

        # argmax, max, last ten accuracies
        np.set_printoptions(precision=3, suppress=True, floatmode='fixed')
        print(f"DT results")
        print(f"DT {acc_dt:.3f} @ {dt.get_depth()}")
        print("NN results")
        print("layers -> best @ batchnum; last accuracies")
        for l, a in zip(_nn_layers, nn_accs):
            best_idx = np.argmax(a)
            best_acc = a[best_idx]
            last_acc = np.array(a[-3:][::-1])
            print(f"{l} -> {best_acc:.3f} @ {best_idx+1}/{n_batches}; last : {np.array2string(last_acc)}")





    # print('\n\n\n')
    print("winequality")
    # (wXtr, wYtr), (wXte, wYte), (wXval, wYval), wdims, wclasses = get_winequality(as_torch=False)
    # print_info(wXtr, wYtr, wXte, wYte, wXval, wYval, wclasses)
    # print('\n\n\n')
    # print("pageblocks")
    # (pXtr, pYtr), (pXte, pYte), (pXval, pYval), pdims, pclasses = get_pageblocks(as_torch=False)
    # print_info(pXtr, pYtr, pXte, pYte, pXval, pYval, pclasses)
    # print('\n\n\n')
    # print("vehicle")
    (vXtr, vYtr), (vXte, vYte), (vXval, vYval), vdims, vclasses = get_vehicle(as_torch=False)
    # print_info(vXtr, vYtr, vXte, vYte, vXval, vYval, vclasses)
    # print('\n\n\n')
    # print("autouniv")
    (auXtr, auYtr), (auXte, auYte), (auXval, auYval), audims, auclasses = get_autouniv(as_torch=False)
    # print_info(auXtr, auYtr, auXte, auYte, auXval, auYval, auclasses)
    # print('\n\n\n')
    # print("abalone")
    (aXtr, aYtr), (aXte, aYte), (aXval, aYval), adims, aclasses = get_abalone(as_torch=False)
    # print_info(aXtr, aYtr, aXte, aYte, aXval, aYval, aclasses)
    # print('\n\n\n')
    # print("diggle")
    (dXtr, dYtr), (dXte, dYte), (dXval, dYval), ddims, dclasses = get_diggle(as_torch=False)
    # print_info(dXtr, dYtr, dXte, dYte, dXval, dYval, dclasses)
    # print('\n\n\n')

    datasets = [
        'wine', 'iris'
    ]
    for dc in datasets:
        print(dc)
        train_tree_nn(dataset_callables[dc])
        print('\n\n')

    import sys; sys.exit()



    # get_ann = lambda: make_ff([ndims, *[8] * 4, nc])
    # for dset in [_heloc]:
    #     print()
    #     # unqs = [len(u) for u in [np.unique(s) for s in dset[0][0].T]]
    #     # bal = (np.mean(dset[0][1].T), np.mean(dset[1][1].T), np.mean(dset[2][1].T))
    #     # print(min(unqs), max(unqs), np.round(bal, 3))
    #
    #     tr, val, te, ndims, nclasses = dset
    #     # print(f"tr {len(tr[0])}, val {len(val[0])}, te {len(te[0])}, ndim {ndims}")
    #     # dt = DT(max_depth=12, random_state=42)
    #     # dt.fit(X=tr[0], y=tr[1])
    #     # print(f"tree depth: {dt.get_depth()}")
    #     # print(f"dt train score: {np.round(dt.score(*tr), 3)}")
    #     # print(f"dt val   score: {np.round(dt.score(*val), 3)}")
    #
    #     nd, nc = dset[-2:]
    #     ann = make_ff([nd, *[32]*4, nc])
    #     tr_torch = _wrap_numpy_to_loader(tr[0], tr[1], shuffle=True)
    #     val_torch = _wrap_numpy_to_loader(val[0], val[1], batch_size=len(val[0]), shuffle=False)
    #     accs, _ = simple_training(ann, torch.optim.Adam(ann.parameters()),
    #                                   torch.nn.CrossEntropyLoss(), tr_torch, next(iter(val_torch)),
    #                                   n_batches=500)
    #     print(np.round(max(accs), 3), '@', np.argmax(accs), '|',  np.round(accs[-1], 3))
    #
    # print()
    # pass


    # print("heloc")
    tr, val, te, ndims, nc = get_beans(None, as_torch=False)
    print(f"tr {len(tr[0])}, val {len(val[0])}, te {len(te[0])}, ndim {ndims}")
    for i in range(1, 15):
        dt = DT(max_depth=i, random_state=42)
        dt.fit(X=tr[0], y=tr[1])
        print(f"dt score: {dt.score(*val):.3f} @ depth = {dt.get_depth()}")
    print()
    # tr, val, te, ndims, nc = get_heloc(None, batch_sizes=(32, 0), root='../datasets', as_torch=True,
    #                                    stratified_sampling=True)
    # ann = make_ff([ndims, 100, 100, nc])
    # simple_training(ann, torch.optim.Adam(ann.parameters()), torch.nn.CrossEntropyLoss(), tr, next(iter(val)),
    #                 n_batches=3_000)
    #
    # print()
    # print("pima")
    # tr, val, te, ndims, nc = get_pima(None, root='../datasets', as_torch=False)
    # print(f"tr {len(tr[0])}, val {len(val[0])}, te {len(te[0])}, ndim {ndims}")
    # dt = DT(max_depth=5, random_state=42)
    # dt.fit(X=tr[0], y=tr[1])
    # print(f"dt pima score: {dt.score(*val)}")
    #
    # tr, val, te, ndims, nc = get_pima(None, batch_sizes=(32, 0), root='../datasets', as_torch=True,
    #                                   stratified_sampling=True)
    # ann = make_ff([ndims, 100, 100, nc])
    # simple_training(ann, torch.optim.Adam(ann.parameters()), torch.nn.CrossEntropyLoss(), tr, next(iter(val)),
    #                 n_batches=40_000)
    #
    # print()
    # print("gmsc")
    # tr, val, te, ndims, nc = get_gmsc(None, root='../datasets', as_torch=False)
    # print(f"tr {len(tr[0])}, val {len(val[0])}, te {len(te[0])}, ndim {ndims}")
    # dt = DT(max_depth=5, random_state=42) # md=6
    # dt.fit(X=tr[0], y=tr[1])
    # print(f"dt gmsc score: {dt.score(*val)}")
    #
    # tr, val, te, ndims, nc = get_gmsc(None, batch_sizes=(32, 0), root='../datasets', as_torch=True,
    #                                   stratified_sampling=True)
    # # ann = make_ff([ndims, 8, 8, nc]), batchsize=8
    # ann = make_ff([ndims, 100, 100, nc])
    # simple_training(ann, torch.optim.Adam(ann.parameters()), torch.nn.CrossEntropyLoss(), tr, next(iter(val)),
    #                 n_batches=10_000)

    from copy import deepcopy


    def manipulate_data(x, y, n_dims_added=5, flipy=0.10):
        # print(x.shape)
        # print()
        _new_x = []
        for i in range(n_dims_added):
            _idxs = np.arange(len(y))
            if type(flipy) == list:
                _flipy = flipy[i]
            else:
                _flipy = flipy
            cut = int(np.ceil(len(y) * _flipy))
            np.random.shuffle(_idxs)
            _noisy = _idxs[:cut]
            _x = deepcopy(y)
            for n in _noisy:
                _x[n] = int(not y[n])
            # _x = np.zeros(len(y))-1
            _new_x.append(_x)
        _new_x = np.vstack(_new_x).T
        _x = np.hstack((deepcopy(x), _new_x))
        # print(np.mean(_new_x[:, -1] == y))
        # print(_x.shape)
        return _x


    print()
    print("heart")
    tr, val, te, ndims, nc = get_heloc(None, root='../datasets', as_torch=False)
    print(f"tr {len(tr[0])}, val {len(val[0])}, te {len(te[0])}, ndim {ndims}")
    dt = DT(max_depth=6, random_state=42) #md=6
    dt.fit(X=tr[0], y=tr[1])
    print(f"dt heart tr={dt.score(*tr)}, te={dt.score(*val)} @ max depth {dt.get_depth()} with {dt.get_n_leaves()} leaves")

    tr_torch, val_torch, te_torch, ndims, nc = get_heloc(None, batch_sizes=(32, 0), root='../datasets', as_torch=True,
                                       stratified_sampling=True)
    print(f"ff data not manipulated")
    from models import BootstrapEnsembleNested as BSE, extend_Bootstrap, validate
    x, y = next(iter(tr_torch))
    ann = make_ff([ndims, *[100] * 4, nc])

    get_ann = lambda: make_ff([ndims, *[8] * 4, nc])
    base_bses = [BSE(get_ann(), None) for _ in range(3)]
    trained_bse = []
    n_batches = 1000
    for i, bse in enumerate(base_bses):
        print(f"fit base bse {i}")
        _, m = simple_training(bse, torch.optim.Adam(bse.model.parameters()), torch.nn.CrossEntropyLoss(), tr_torch,
                        next(iter(val_torch)),
                        n_batches=n_batches)
        trained_bse.append(m)
    assert np.all(np.array([validate(b.to('cuda'), next(iter(val_torch))) for b in trained_bse]) > 0.5)
    # print(f"fit bse lvl 2}")
    # bse_2 = extend_Bootstrap(trained_bse, ndims, nc, layers=[100]*4)
    # simple_training(bse_2, torch.optim.Adam(bse_2.model.parameters()), torch.nn.CrossEntropyLoss(), tr_torch, next(iter(val_torch)),
    #                 n_batches=1000)
    depth = 3
    ensemble_size = 3
    final_bse = None
    cur_bses = trained_bse
    d = 0
    while d < depth:
        d += 1
        _next_bses = []
        for _m in range(ensemble_size):
            print(f"{d} - {_m}")
            _bse_d_m = extend_Bootstrap(cur_bses, ndims, nc, layers=[8]*4)
            _, _bse_d_m = simple_training(_bse_d_m, torch.optim.Adam(_bse_d_m.model.parameters()), torch.nn.CrossEntropyLoss(), tr_torch, next(iter(val_torch)),
                    n_batches=n_batches)
            _next_bses.append(_bse_d_m)
        assert np.all(np.array([validate(b.to('cuda'), next(iter(val_torch))) for b in _next_bses]) > 0.5)
        cur_bses = _next_bses
    final_bse = extend_Bootstrap(cur_bses, ndims, nc)
    print(f"final before training: {validate(final_bse.to('cuda'), next(iter(val_torch)))}")
    _, final_bse = simple_training(final_bse, torch.optim.Adam(final_bse.model.parameters()), torch.nn.CrossEntropyLoss(), tr_torch, next(iter(val_torch)),
                    n_batches=n_batches)
    print(f"final after training: {validate(final_bse.to('cuda'), next(iter(val_torch)))}")
    print(f"ff manipulated data")
    import sys; sys.exit()

    extra_dims = [1, 2, 3, 5]
    noise = [0.05, [0.05, 0.4]]
    for ed, no in itertools.product(extra_dims, noise):
        print(f"dims+{ed}, noise@{no}")
        _manip_tr = manipulate_data(*deepcopy(tr), n_dims_added=ed, flipy=no)
        _manip_te = manipulate_data(*deepcopy(val), n_dims_added=ed, flipy=no)
        ndims = _manip_tr.shape[1]
        _tr = _wrap_numpy_to_loader(_manip_tr, tr[1], batch_size=32, shuffle=True)
        _te = _wrap_numpy_to_loader(_manip_te, val[1], batch_size=len(val[0]))
        (x_te, y_te) = next(iter(_te))
        # ann = make_ff([ndims, 16, 16, 16, nc]) # [8 ,8] @ 32batch
        # 8, 8 @ 8 batch
        ann = make_ff([ndims, *[8] * 4, nc])
        simple_training(ann, torch.optim.Adam(ann.parameters()), torch.nn.CrossEntropyLoss(),
                        _tr, (x_te, y_te), n_batches=n_batches)


    import sys; sys.exit()
    from tqdm import tqdm
    from copy import deepcopy
    #
    # def assess_randomization(loader1, loader2, D=3, seed=199):
    #     def get_some_batches(loader):
    #         X, Y = [], []
    #
    #         n_loops = 3
    #         n_batches = 3
    #         # sample n_loops times n_batches from both train and test
    #         for e in range(n_loops):
    #             X.append([]); Y.append([])
    #             for i, (x1, y1) in enumerate(loader):
    #                 if i > n_batches:
    #                     break
    #                 X[-1].append(x1); Y[-1].append(y1)
    #         return X, Y
    #
    #     x1, y1 = get_some_batches(loader1)
    #     x2, y2 = get_some_batches(loader2)
    #     with TorchRandomSeed(seed):
    #         x1r, y1r = get_some_batches(loader1)
    #     with TorchRandomSeed(seed + 1):
    #         x2r, y2r = get_some_batches(loader2)
    #
    #     # assert that l1 == l2 and l1_rand != l2_rand
    #
    #     # data = [[tr_data[d][0] for d in range(D)], [te_data[d][1] for d in range(D)]]
    #     print('done')
    #     pass
    #
    def check_data(train, test, dim, n_classes):
        print(f"{dim}, {n_classes}")
        print(f"sizes train: {len(train)} test {len(test)}")
        print(f"dim {dim}, n_classes {n_classes}")
        for (x_tr, y_tr), (x_te, y_te) in zip(train, test):
            print("shapes train data - labels: ", x_tr.shape, y_tr.shape)
            print("shapes test data - labels: ", x_te.shape, y_te.shape)
            print(x_tr)
            print(x_te)
            break

    # print("############################")
    # print("### TESTING ALL DATAESTS ###")
    # print("############################\n\n")
    #
    # t1, _,_,_ = get_agnews(random_state=32, batch_sizes=(1, 1))
    # l = []
    # for x, y in t1:
    #     l.append(x.shape[1])
    # print(sum(l)/len(l))
    #
    # random_state = 42
    # with TorchRandomSeed(random_state):
    #     tr1, te1, _, _ = get_agnews(123)
    # with TorchRandomSeed(random_state):
    #     tr2, te2, _, _ = get_agnews(123)
    # assess_randomization(tr1, tr2)
    # tr3, te3, _, _ = get_agnews(132)
    # assess_randomization(tr1, tr3)
    # assess_randomization(te3, te3)
    # exit()
    # import sys
    # sys.exit()


    # train, test, n_feat, n_classes = get_beans(random_state)
    # check_data(train, test, n_feat, n_classes)
    # train, test, size_vocab, n_classes = get_dbpedia(random_state)
    # check_data(train, test, size_vocab, n_classes)
    # train, test, size_vocab, n_classes = get_agnews(random_state)
    # check_data(train, test, size_vocab, n_classes)
    # train, test, n_feat, n_classes = get_emnist(random_state, batch_sizes=(64, 18800))
    # check_data(train, test, n_feat, n_classes)
    # train, test, n_feat, n_classes = get_covtype(random_state)
    # check_data(train, test, n_feat, n_classes)
    # print("### GENERATE MAKE_CLASSIFICATION")
    # train, test, n_feat, n_classes = get_classification()
    # check_data(train, test, n_feat, n_classes)
    # print("### GENERATE WAVEFORM")
    # train, test, n_feat, n_classes = get_waveform()
    # check_data(train, test, n_feat, n_classes)
    from sklearn.tree import DecisionTreeClassifier as DTC
    def fit_tree(train, test, maxdepth):
        dt = DTC(max_depth=maxdepth)
        dt.fit(train[0], train[1])
        print(f"fitted tree depth: {dt.get_depth()}")
        print(f"train acc: {dt.score(train[0], train[1])}")
        print(f"test acc: {dt.score(test[0], test[1])}")
        from sklearn.metrics import confusion_matrix
        print(f"test confusion:\n {confusion_matrix(test[1], dt.predict(test[0]))}")
        pass

    random_state = 42
    print("\n### LOADING COVERTYPE")
    train, test, n_feat, n_classes = get_covtype(random_state, as_torch=False)
    idxs = np.arange(len(train[0]))
    np.random.shuffle(idxs)
    n_samples = 10000
    train = (train[0][idxs[:n_samples]], train[1][idxs[:n_samples]])
    test = (test[0][:n_samples], test[1][:n_samples])
    fit_tree(train, test, 5)
    # check_data(train, test, n_feat, n_classes)
    print("\n### IONOSPHERE")
    train, test, n_feat, n_classes = get_ionosphere(random_state, as_torch=False)
    fit_tree(train, test, 4)
    # check_data(train, test, n_feat, n_classes)
    print("\n### IRIS")
    train, test, n_feat, n_classes = get_iris(random_state, as_torch=False)
    fit_tree(train, test, 2)
    # check_data(train, test, n_feat, n_classes)
    print("\n### WINE")
    train, test, n_feat, n_classes = get_wine(random_state, as_torch=False)
    fit_tree(train, test, 3)
    # check_data(train, test, n_feat, n_classes)
    print("\n### BREAST CANCER")
    train, test, n_feat, n_classes = get_breast_cancer(random_state, as_torch=False)
    fit_tree(train, test, 3)
    # check_data(train, test, n_feat, n_classes)

    import sys; sys.exit()


    print("### LOAD FASHION-MNIST")
    train, test, n_feat, n_classes = get_fmnist(random_state)
    check_data(train, test, n_feat, n_classes)
    print("### LOAD EXTENDED-MNIST")
    train, test, n_feat, n_classes = get_emnist(random_state, batch_sizes=(64, 18800))
    check_data(train, test, n_feat, n_classes)
    print("### LOAD YELPREVIEWSFULL")
    train, test, size_vocab, n_classes = get_yelpreviewfull(random_state)
    check_data(train, test, size_vocab, n_classes)
    print("### LOAD AG NEWS")
    train, test, size_vocab, n_classes = get_agnews(random_state)
    check_data(train, test, size_vocab, n_classes)
    print("### LOAD DBPEDIA")
    train, test, size_vocab, n_classes = get_dbpedia(random_state)
    check_data(train, test, size_vocab, n_classes)
    print("DONE")
