from itertools import chain

import numpy as np

from lxg.models import DNFClassifier
from cfire.gely import ItemsetNode
from cfire.util import load_nn_dnfs

def greedy_itemsetnode_set_cover(X: set[int], F: list[tuple[set[int], ItemsetNode]], _lam=0.5):
    '''
    X are items to be coverd
    F contains tuples(items covered, ItemSetNode)
    '''
    U = X.copy()
    C = []  # represents picked instances, either via index or by pointer reference
    # def remove_parent_children(node, _F):
    #     # err 'argmax over empty sequence'
    #     # some children cover samples that the parent didn't cover.
    #     # that leads to U not being exhausted before F is empty
    #     to_be_removed = [node, node.parent] + node.get_children()
    #     F_new = [f for f in _F if f[1] not in to_be_removed]
    #     _F.remove(node)
    #     return F_new

    while len(U) > 0:
        # _intersects = [len(U.intersection(f[0])) for f in F]

        # _f_idx = np.argmax([f[1].score(_lam) for f in F])
        # _f = F[_f_idx]
        _f = sorted(F, key=lambda x: (-len(U.intersection(x[0])), -x[1].complexity))[0]
        _f = sorted(F, key=lambda x: (-x[1].complexity, -len(U.intersection(x[0]))))[0]

        # F = remove_parent_children(_f[1], F)
        F.remove(_f)
        U = U - _f[0]
        C.append(_f[1])

    return C

def greedy_compl_cover(X: set[int], F: list[tuple[set[int], ItemsetNode]], _lam=0.5):
    U = X.copy()
    C = []
    while len(U) > 0:
        _f = sorted(F, key=lambda x: (-x[1].complexity, -len(U.intersection(x[0]))))[0]
        F.remove(_f)
        U = U - _f[0]
        C.append(_f[1])
    return C


def greedy_cover_compl(X: set[int], F: list[tuple[set[int], ItemsetNode]], _lam=0.5):
    U = X.copy()
    C = []
    while len(U) > 0:
        _f = sorted(F, key=lambda x: (-len(U.intersection(x[0])), -x[1].complexity))[0]
        F.remove(_f)
        U = U - _f[0]
        C.append(_f[1])
    return C


def greedy_cover(X: set[int], F: list[tuple[set[int], ItemsetNode]]):
    U = X.copy()
    C = []
    while len(U) > 0:
        _f = sorted(F, key=lambda x: -len(U.intersection(x[0])))[0]
        F.remove(_f)
        U = U - _f[0]
        C.append(_f[1])
    return C


def greedy_score_cover(X: set[int], F: list[tuple[set[int], ItemsetNode]], _lam=0.5):
    U = X.copy()
    C = []
    while len(U) > 0:
        _f = sorted(F, key=lambda x: (-x[1].score(_lam), -len(U.intersection(x[0]))))[0]
        F.remove(_f)
        U = U - _f[0]
        C.append(_f[1])
    return C


def greedy_score(X: set[int], F: list[tuple[set[int], ItemsetNode]], _lam=0.5):
    U = X.copy()
    C = []
    while len(U) > 0:
        _f = sorted(F, key=lambda x: -x[1].score(_lam))[0]
        F.remove(_f)
        U = U - _f[0]
        C.append(_f[1])
    return C

def comp_variants_dnfs(_supp, _node_set, X_target, X_other):
    __items_covered = set(chain.from_iterable(_supp))
    nodes_compl_cover = greedy_compl_cover(__items_covered, list(zip(_supp, _node_set)), _lam=0.8)
    nodes_cover_compl = greedy_cover_compl(__items_covered, list(zip(_supp, _node_set)), _lam=0.8)
    nodes_cover = greedy_cover(__items_covered, list(zip(_supp, _node_set)))
    nodes_score_cover = greedy_score_cover(__items_covered, list(zip(_supp, _node_set)), _lam=0.5)
    nodes_score = greedy_score(__items_covered, list(zip(_supp, _node_set)), _lam=0.5)
    dnf_compl_cover = DNFClassifier([list(chain.from_iterable([n.dnf[0] for n in nodes_compl_cover]))])
    dnf_cover_compl = DNFClassifier([list(chain.from_iterable([n.dnf[0] for n in nodes_cover_compl]))])
    dnf_cover = DNFClassifier([list(chain.from_iterable([n.dnf[0] for n in nodes_cover]))])
    dnf_score_cover = DNFClassifier([list(chain.from_iterable([n.dnf[0] for n in nodes_score_cover]))])
    dnf_score = DNFClassifier([list(chain.from_iterable([n.dnf[0] for n in nodes_score]))])
    print('compl_cover\t\t', dnf_compl_cover.n_literals,
          (np.mean(dnf_compl_cover(X_target)), np.mean(dnf_compl_cover(X_other))))
    print('cover_compl\t\t', dnf_cover_compl.n_literals,
          (np.mean(dnf_cover_compl(X_target)), np.mean(dnf_cover_compl(X_other))))
    print('cover\t\t\t', dnf_cover_compl.n_literals, (np.mean(dnf_cover(X_target)), np.mean(dnf_cover(X_other))))
    print('score_cover\t\t', dnf_score_cover.n_literals,
          (np.mean(dnf_score_cover(X_target)), np.mean(dnf_score_cover(X_other))))
    print('score\t\t\t', dnf_score.n_literals, (np.mean(dnf_score(X_target)), np.mean(dnf_score(X_other))))
    return [dnf_compl_cover, dnf_cover_compl, dnf_cover, dnf_score_cover, dnf_score]

if __name__ == '__main__':
    support_idxs = None
    frequent_nodes = None
    r = comp_variants_dnfs(support_idxs, frequent_nodes)
    _lams = [0.1, 0.25, 0.5, 0.9, 1.]
    for _lam in _lams:
        print(f"lambda = {_lam}")
        idxs = np.argsort([-f.score(_lam) for f in frequent_nodes])[:int(0.25*len(frequent_nodes))]
        _filtered_nodes = [frequent_nodes[i] for i in idxs]
        _support_sets = [support_idxs[i] for i in idxs]
        r.extend(comp_variants_dnfs(_support_sets, _filtered_nodes))

        selection_params = dict(
            gt=None,  # gely_threshold=0.8,
            ex=None,  # expl_method='ig',
            km=None,  # k_means_max_bins=2,
            s=None,  # model_seed=None,
            fs=None,  # n_sam ples_percent=1.0,
            sc=None,  # setcover_reduction=True,
            st=None,#'topk0.5',  # significance_threshold='top0.5',
        )
