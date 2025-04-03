#!/bin/sh

export PYTHONPATH="${PYTHONPATH}:../`pwd`:../`pwd`/cfire/:`pwd`/lxg/"

python 03_eval_rules.py --composednf True

python 03_eval_rules.py --evaldnf True

python 03_eval_rules.py --evalanchors True

python 03_eval_rules.py --evalcega True
