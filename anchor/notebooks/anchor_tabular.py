import numpy as np
np.random.seed(1)
import sys
import sklearn
import sklearn.ensemble

from anchor import utils
from anchor import anchor_tabular

from lime.lime_tabular import QuartileDiscretizer

from tqdm import tqdm

import lxg.datasets as lxgdata

from joblib import parallel_backend, Parallel, delayed

from lxg.models import DNFClassifier
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

    anchors_dataset = utils.Bunch(d)

    return anchors_dataset




# make sure you have adult/adult.data inside dataset_folder
dataset_folder = '/opt/mlfta/muellers/gits/inexp/anchor/datasets/'
# dataset = utils.load_dataset('adult', balance=True, dataset_folder=dataset_folder, discretize=True)
dataset = utils.load_dataset('adult', balance=False, dataset_folder=dataset_folder, discretize=False)

(X_tr, Y_tr), (X_te, Y_te), (X_val, Y_val), n_dim, n_classes = lxgdata.get_beans(11880, as_torch=False, root='../../datasets')
X = np.vstack([X_tr, X_te, X_val])
Y = np.hstack([Y_tr, Y_te, Y_val])
tr_idxs = np.arange(len(X_tr))
te_idxs = np.arange(len(X_te)) + len(X_tr)
validation_idxs = np.arange(len(X_val)) + len(X_tr) + len(X_te)
dataset = anchors_dataset(X, Y, tr_idxs, te_idxs, validation_idxs, n_classes, n_dim)

c = sklearn.ensemble.RandomForestClassifier(n_estimators=50, n_jobs=5)
c.fit(dataset.train, dataset.labels_train)
print('Train', sklearn.metrics.accuracy_score(dataset.labels_train, c.predict(dataset.train)))
print('Test', sklearn.metrics.accuracy_score(dataset.labels_test, c.predict(dataset.test)))




explainer = anchor_tabular.AnchorTabularExplainer(
    dataset.class_names,
    dataset.feature_names,
    dataset.train,
    dataset.categorical_names)





idx = 0
np.random.seed(1)
print('Prediction: ', explainer.class_names[c.predict(dataset.test[idx].reshape(1, -1))[0]])

exps = []
idxs = np.random.choice(dataset.test.shape[0], 20, replace=False)
# for i in tqdm(idxs):
#     exp = explainer.explain_instance(dataset.test[i], c.predict, threshold=0.90)
#     exps.append(exp.exp_map)
print(len(dataset.test))
with parallel_backend(backend='loky', n_jobs=12):  # 14
    # Parallel()(delayed(rename)(task=a[0], expl_method=a[1], model_set=a[2], gely_threshold=a[3],
    #                                         significance_threshold=a[4], k_means_max_bins=a[5]) for a in arg_sets)
    exps = Parallel(verbose=10, batch_size=2)(delayed(explainer.explain_instance)(i, c.predict, threshold=0.9) for i in dataset.test[:500])
exps = [e.exp_map for e in exps]

for k in range(5, 40):
    chosen_idxs = utils.greedy_pick_anchor(exps, dataset.test[:500], k=k)
    chosen_anchors = [exps[i] for i in chosen_idxs]

    dnf: DNFClassifier = anchors_to_rule_model(chosen_anchors, n_classes)
    print(f"{k} -> acc {np.mean(dnf(X_val) == c.predict(X_val)):.3f}, n_rules={dnf.n_rules}, n_unique_literals={dnf.n_literals}")
print("done")

# print('Anchor: %s' % (' AND '.join(exp.names())))
# print('Precision: %.2f' % exp.precision())
# print('Coverage: %.2f' % exp.coverage())


# fit_anchor = np.where(np.all(dataset.test[:, exp.features()] == dataset.test[idx][exp.features()], axis=1))[0]
# print('Anchor test precision: %.2f' % (np.mean(c.predict(dataset.test[fit_anchor]) == c.predict(dataset.test[idx].reshape(1, -1)))))
# print('Anchor test coverage: %.2f' % (fit_anchor.shape[0] / float(dataset.test.shape[0])))