# CFIRE

Tested with Python 3.11.

Install dependencies using `pip install -r requirements.txt`.


Had no time to test from a clean install before submission:

0. File to set up conda environment:`cfire_requirements`
1. Train neural networks: `run_small.sh` # some datasets may have to be downloaded manually
	Trains 50 neural networks per task, network architectures are defined in script
2. Calculcate explanations: `calc_expls.sh`
	calculates KS, IG and LI explanations for all models
3. Calculate Rules for all methods: `cfire_experiments/calc_expl_rules.sh`
	Calculation for Anchors will take some time
4. Evaluate: `cfire_experiments/run_full_eval.sh`
	Evaluation of CEGA takes a long time
5. Print LaTeX Tables: `python cfire_experiments/plot_rule_results.py`
	Also produces plots
