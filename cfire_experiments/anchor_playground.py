import numpy as np
np.random.seed(1)
import sys
import sklearn
import sklearn.ensemble
from sklearn.datasets import fetch_openml

from anchor import utils
from anchor import anchor_tabular

import cfire._variables_cfire as _variables

import pandas as pd

# 1590 is adult dataset
dataset = fetch_openml(data_id=1590, data_home='./', as_frame=True, cache=True)
X = pd.get_dummies(dataset.data)

c = sklearn.ensemble.RandomForestClassifier(n_estimators=50, n_jobs=5)
c.fit(X, dataset.target)
print('Train', sklearn.metrics.accuracy_score(dataset.target, c.predict(X)))
categorical_feature_names = dataset.data.select_dtypes(include=['object', 'category']).columns.to_numpy()
categorical_features = {k:v for k, v in zip([1, 3, 5, 6, 7, 8, 9, 10, 11, 13], categorical_feature_names)}
explainer = anchor_tabular.AnchorTabularExplainer(
    dataset.target_names,
    dataset.feature_names,
    dataset.data,
    None)
