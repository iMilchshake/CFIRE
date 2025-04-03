from pathlib import Path


counter = 0

tasks = [
    "breastcancer",
    "ionosphere",
    'btsc',
    'spf',  # slow COMPosition rf
    'spambase', # slow eval, *relly* slow COMP rf
    'heloc',  # slow eval; slow comp on rf

    'autouniv',
    'abalone',
    'wine',
    'iris',
    'vehicle',
    'diggle',
    'beans',  # slow eval
    'breastw',  # slow eval
]

tasks_plotting_order = [
    'btsc',
    'spf',
    'breastw',
    'spambase',
    'heloc',
    "breastcancer",
    "ionosphere",
    'beans',
    'wine', #'vehicle',
    'diggle',
    'abalone',
    'iris', #'autouniv',
    'vehicle',
    'autouniv'

]



explanation_abbreviations = [
    'vg',
    'sg',
    'ig',
    'ks',
    'li',
    'grdpgi'
]

_explanation_abbreviations_paper = {e: e.upper() for e in explanation_abbreviations}

base_dir = './'

data_path = 'data/cfire'
expl_rule_dir = 'expl_rules'  # ItemsetNodes
dnfclassifier_dir ='dnfclassifier'  # our models
cega_dir = str(Path(data_path,'cega_rules'))  # CEGA models
nn_postfix = 'NN_dnfDiscriminative'

path_boxsystems = 'data/cfire'

results_prefix = 'rules_mar'


def __create_dir(dir):
    dir.mkdir(exist_ok=True, parents=True)
    return


def get_nn_dnf_dir(task):
    pth = Path(data_path, nn_postfix, task)
    __create_dir(pth)
    return pth


def get_expl_rule_dir(task, modelclass=None):
    # expl_rule_dir depends on data_path which includes modelclas
    if modelclass is not None:
        return Path(data_path, expl_rule_dir, modelclass, task)
    pth = Path(data_path, expl_rule_dir, task)
    return pth


def get_cega_dir(task, modelclass=None):
    # cega dir depends on data_path which includes modelclass
    if modelclass is not None:
        return Path(cega_dir, modelclass, task)
    pth = Path(cega_dir, task)
    return pth

def get_dnfclassifier_dir(task, modelclass=None):
    if modelclass is not None:
        return Path(data_path, dnfclassifier_dir, modelclass, task)
    pth = Path(data_path, dnfclassifier_dir, task)
    return pth

def get_data_dir(task, modelclass=None):
    if modelclass is None:
        dir = Path(base_dir, data_path, task)
    else:
        dir = Path(base_dir, data_path, modelclass, task)
    __create_dir(dir)
    return dir


def get_result_dir(task, custom_prefix=None):
    _results_prefix = results_prefix if custom_prefix is None else custom_prefix
    dir = Path(base_dir, '../results', _results_prefix, task)
    __create_dir(dir)
    return dir


def get_plot_dir(task):
    dir = Path(base_dir, '../plots', results_prefix, task)
    __create_dir(dir)
    return dir

def get_boxes_dir(task, modelclass='nn'):
    return Path(path_boxsystems, modelclass, task, 'boxes')