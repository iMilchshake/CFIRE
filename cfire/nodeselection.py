from itertools import chain

import numpy as np

from lxg.models import DNFClassifier
from .gely import ItemsetNode
from .util import load_nn_dnfs
from itertools import chain
import pulp

def greedy_itemsetnode_set_cover(X: set[int], F: list[tuple[set[int], ItemsetNode]]):
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
        _f = sorted(F, key=lambda x: (-len(U.intersection(x[0])), -x[1].complexity_factor))[0]

        _f = sorted(F, key=lambda x: (-x[1].complexity_factor, -len(U.intersection(x[0]))))[0]

        # F = remove_parent_children(_f[1], F)
        F.remove(_f)
        U = U - _f[0]
        C.append(_f[1])

    return C

def greedy_compl_cover(X: set[int], F: list[tuple[set[int], ItemsetNode]]):
    U = X.copy()
    C = []
    while len(U) > 0:
        _f = sorted(F, key=lambda x: (-x[1].complexity_factor, -len(U.intersection(x[0]))))[0]
        F.remove(_f)
        U = U - _f[0]
        C.append(_f[1])
    return C


def greedy_cover_compl(X: set[int], F: list[tuple[set[int], ItemsetNode]]):
    U = X.copy()
    C = []
    while len(U) > 0:
        _f = sorted(F, key=lambda x: (-len(U.intersection(x[0])), -x[1].complexity_factor))[0]
        F.remove(_f)
        U = U - _f[0]
        C.append(_f[1])
    return C

# covers universe by greedily selecting rules that cover the most uncovered items
def greedy_cover(universe_to_cover: set[int], candidate_rules: list[tuple[set[int], ItemsetNode]]):
    remaining_universe = universe_to_cover.copy()
    selected_cover_set = []
    while len(remaining_universe) > 0:
        # pick next rule by most samples of remaining covered
        _best_rule = sorted(candidate_rules, key=lambda rule: - len(remaining_universe.intersection(rule[0])))[0]
        candidate_rules.remove(_best_rule)
        remaining_universe = remaining_universe - _best_rule[0]
        selected_cover_set.append(_best_rule[1])
    return selected_cover_set


def greedy_score_cover(universe_to_cover: set[int], candidate_rules: list[tuple[set[int], ItemsetNode]], _lam=0.5, _lam2=0.5, _lam3=0):
    remaining_universe = universe_to_cover.copy()
    selected_cover_set = []
    while len(remaining_universe) > 0:
        _f = sorted(candidate_rules, key=lambda rule: (-rule[1].score(_lam, _lam2, _lam3), -len(remaining_universe.intersection(rule[0]))))[0]
        candidate_rules.remove(_f)
        U_new = remaining_universe - _f[0]
        if len(U_new) < len(remaining_universe):  # only add element if it increased coverage
            selected_cover_set.append(_f[1])
        remaining_universe = U_new
    return selected_cover_set

def greedy_dynamic_scoring(X: set[int], F: list[tuple[set[int], ItemsetNode]], _lam=0.5, _lam2=0.5, _lam3=0):
    U = X.copy()
    C = []

    def _scoring(node, support, uncovered, l1, l2, l3):
        p, r = node.precision, node.recall
        f1 = 2 * (p*r)/(p+r)
        _covering = len(uncovered-support)/len(uncovered)
        node_score = l1*f1 + l2*node.complexity_factor + l3*node.completeness_factor
        return _covering * node_score
    threshold = max(1, len(X)*0.02)

    while len(U) > threshold:
        _f = sorted(F, key=lambda x: -_scoring(x[1], x[0], U, _lam, _lam2, _lam3))[0]
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


def _comp_greedy_cover(_supp, _node_set):
    # create universe set of items to be covered
    __items_covered = set(chain.from_iterable(_supp))

    nodes = greedy_cover(__items_covered, list(zip(_supp, _node_set)))
    dnf = DNFClassifier([list(chain.from_iterable([n.dnf[0] for n in nodes]))],
                        tie_break="accuracy")
    return dnf

def _comp_score_cover(_supp, _node_set, acc_weight, cx_weight, cs_weight):
    __items_covered = set(chain.from_iterable(_supp))
    nodes = greedy_score_cover(__items_covered, list(zip(_supp, _node_set)),
                               acc_weight, cx_weight, cs_weight)
    dnf = DNFClassifier([list(chain.from_iterable([n.dnf[0] for n in nodes]))],
                        tie_break="accuracy")
    return dnf

def _comp_score_cover_weighted(_supp, _node_set, acc_weight, cx_weight, cs_weight):
    __items_covered = set(chain.from_iterable(_supp))
    nodes = greedy_dynamic_scoring(__items_covered, list(zip(_supp, _node_set)),
                               acc_weight, cx_weight, cs_weight)
    dnf = DNFClassifier([list(chain.from_iterable([n.dnf[0] for n in nodes]))],
                        tie_break="accuracy")
    return dnf


def comp_variants_dnfs(_supp, _node_set):
    __items_covered = set(chain.from_iterable(_supp))
    nodes_compl_cover = greedy_compl_cover(__items_covered, list(zip(_supp, _node_set)))
    # nodes_cover_compl = greedy_cover_compl(__items_covered, list(zip(_supp, _node_set)))
    nodes_cover = greedy_cover(__items_covered, list(zip(_supp, _node_set)))

    # _lambdas = np.linspace(0, 1, 10)
    nodes_score_cover = greedy_score_cover(__items_covered, list(zip(_supp, _node_set)), _lam=0.5)
    # nodes_score = greedy_score(__items_covered, list(zip(_supp, _node_set)), _lam=0.5)

    dnf_compl_cover = DNFClassifier([list(chain.from_iterable([n.dnf[0] for n in nodes_compl_cover]))])
    # dnf_cover_compl = DNFClassifier([list(chain.from_iterable([n.dnf[0] for n in nodes_cover_compl]))])
    dnf_cover = DNFClassifier([list(chain.from_iterable([n.dnf[0] for n in nodes_cover]))])
    dnf_score_cover = DNFClassifier([list(chain.from_iterable([n.dnf[0] for n in nodes_score_cover]))])
    # dnf_score = DNFClassifier([list(chain.from_iterable([n.dnf[0] for n in nodes_score]))])\
    return [dnf_cover, dnf_compl_cover, dnf_score_cover]


## after submission implementations of new variants

# currently complexity factor as cost
def _comp_ilp_optimal(supp, nodes):
    chosen_nodes = solve_ilp_set_cover(supp, nodes, cost_key=lambda n: n.complexity_factor)
    return DNFClassifier(
        [list(chain.from_iterable([n.dnf[0] for n in chosen_nodes]))],
        tie_break="accuracy",
    )

def solve_ilp_set_cover(support_sets, nodes,
                        cost_key=lambda n: 1,
                        time_limit=None):
    """
    Exact 0-1 set cover via MILP.
    Returns the *list of ItemsetNode* chosen by the solver.
    """
    print("Solving set cover via ILP..., TAKES TIME")
    m = pulp.LpProblem("SetCover", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{i}", 0, 1, cat="Binary")
         for i in range(len(nodes))]

    # objective: min sum(cost_i * x_i)
    m += pulp.lpSum(cost_key(nodes[i]) * x[i] for i in range(len(nodes)))

    # cover constraints
    universe = list(set(chain.from_iterable(support_sets)))
    for j in universe:
        covering_rules = [i for i, S in enumerate(support_sets) if j in S]
        m += pulp.lpSum(x[i] for i in covering_rules) >= 1

    if time_limit is not None:
        m.solve(pulp.PULP_CBC_CMD(timeLimit=time_limit))
    else:
        m.solve()

    return [nodes[i] for i in range(len(nodes)) if pulp.value(x[i]) > 0.5]

def _comp_greedy_plus_1opt(supp, nodes):
    first = greedy_cover(set().union(*supp), list(zip(supp, nodes)))
    cleaned = one_opt_prune(first, supp, nodes)   # implement the 1-opt loop
    return DNFClassifier([list(chain.from_iterable([n.dnf[0] for n in cleaned]))],
                         tie_break="accuracy")

def one_opt_prune(chosen_nodes,
                  support_sets,
                  all_nodes):          # mapping Node → support set
    """
    Removes any rule whose removal leaves full coverage intact.
    `chosen_nodes` is an *ordered list* output by a greedy selector.
    """
    # Build a mapping node → support for quick set ops
    supp_map = {n: s for n, s in zip(all_nodes, support_sets)}
    covered_universe = set.union(*(supp_map[n] for n in chosen_nodes))

    pruned = []
    for n in chosen_nodes:
        # Universe covered by others
        rest_support = set.union(*(supp_map[o] for o in chosen_nodes if o != n))
        if rest_support == covered_universe:
            continue  # n is redundant
        pruned.append(n)
    return pruned


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
