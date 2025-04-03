#!/bin/sh

export PYTHONPATH="${PYTHONPATH}:../`pwd`:../`pwd`/cfire/:`pwd`/lxg/"

echo "STARTING WITH CFIRE"

python 01_calc_expl_rules.py --modelclass NN &

echo "STARTING WITH ANCHORS"

python compute_anchors.py

echo "STARTING WITH CEGA"

python 01_calc_itemsets.py --cega True --modelclass NN

echo "All models completed"

