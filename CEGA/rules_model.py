from tqdm import tqdm
from .helper_func import *
import sys, math
import numpy as np
from multiprocessing import Pool, cpu_count
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class RulesModel:

    def __init__(self, ohe_df, rules, y_dev, pos_label, neg_label, prb_pos=None, intervals_dict=None):
        self.ohe_df = ohe_df
        self.oh_np_arr = np.array(ohe_df)

        self.pos_label = pos_label
        self.neg_label = neg_label
        self.ind_neg = list(ohe_df.columns).index(neg_label)
        self.ind_pos = list(ohe_df.columns).index(pos_label)
        self.rules = rules
        self.alpha = 0.1
        self.beta = len(rules)
        if prb_pos is None:
            self.prb_pos = sum(y_dev) / len(y_dev)
        else:
            self.prb_pos = prb_pos
        self.prb_neg = 1 - self.prb_pos
        self.intervals_dict = intervals_dict
        self.X = None

    def compute_intervals_dict(self, X_train, num_bins=5):
        names = X_train.columns
        self.intervals_dict = {}
        for name in names:
            unique_values = X_train[name].unique()
            if len(unique_values) > 2 or max(unique_values) != 1 or min(unique_values) != 0:
                intervals = pd.cut(X_train[name], num_bins)
                self.intervals_dict[name] = intervals
    @staticmethod
    def from_RulesModel(rulesmodel):
        return RulesModel(
            ohe_df=rulesmodel.ohe_df,
            rules=rulesmodel.rules,
            y_dev=None,
            pos_label=rulesmodel.pos_label,
            neg_label=rulesmodel.neg_label,
            prb_pos=rulesmodel.prb_pos
        )

    def compute_scores(self, record, eval_mode=False):
        # if eval mode, not only return probas but also every rule that matched

        _, row = record
        positive_score = self.prb_pos
        negative_score = self.prb_neg

        num_pos_rules = 0
        num_neg_rules = 0

        example_set = set()
        # print("rec", record)
        # print("row", row)
        for name, val in row.items():
            if val > 0:
                example_set.add(format_name(name, val, intervals=self.intervals_dict))

        # print("example_set", example_set)
        for i, (_, rule) in enumerate(self.rules.iterrows()):
            label = rule['label']
            itemset = rule['itemset']

            if itemset.issubset(example_set):

                if label == self.pos_label:
                    num_pos_rules += 1
                    positive_score *= ((rule['support'] + self.alpha) / (self.prb_pos + self.alpha * self.beta))

                    neg_count = sum(self.oh_np_arr[:, self.ind_neg])
                    features = [list(self.ohe_df.columns).index(item) for item in list(itemset)]
                    features_support = list(np.sum(np.array(self.ohe_df[self.ohe_df[self.neg_label] == 1]
                                                            )[:, features], axis=1)).count(len(itemset))

                    opp_class_score = (features_support + self.alpha) / (neg_count + self.alpha * self.beta)
                    if opp_class_score > 0:
                        negative_score *= opp_class_score
                    else:
                        negative_score *= (self.alpha / (neg_count + self.alpha * self.beta))

                else:
                    num_neg_rules += 1
                    negative_score *= ((rule['support'] + self.alpha) / (self.prb_neg + self.alpha * self.beta))

                    pos_count = sum(self.oh_np_arr[:, self.ind_pos])
                    features = [list(self.ohe_df.columns).index(item) for item in list(itemset)]
                    features_support = list(np.sum(np.array(self.ohe_df[self.ohe_df[self.pos_label] == 1]
                                                            )[:, features], axis=1)).count(len(itemset))

                    opp_class_score = (features_support + self.alpha) / (pos_count + self.alpha * self.beta)
                    if opp_class_score > 0:
                        positive_score *= opp_class_score
                    else:
                        positive_score *= (self.alpha / (pos_count + self.alpha * self.beta))

        return (positive_score, negative_score, num_pos_rules, num_neg_rules)

    def eval_rules(self, data, y, alpha=20, beta=1, decision_thr=0.5):
        found_sol = []
        probas = []

        self.alpha = alpha
        self.beta = beta
        self.X = data

        num_cores = cpu_count() - 2
        pool = Pool(num_cores)

        scores = list(tqdm(pool.imap(self.compute_scores, data.iterrows()),
                           total=len(data), file=sys.stdout, position=0, leave=True))

        app_pos_dict = {}
        app_neg_dict = {}
        sum_dict = {}
        applicable_rules = []
        for score_tup in scores:
            positive_score, negative_score, num_pos_rules, num_neg_rules = score_tup
            # _appl_rules = None
            # applicable_rules.append(_appl_rules)
            if negative_score == 0 and positive_score == 0: print('not applicable')
            if negative_score == 0 and positive_score == 0 or num_pos_rules + num_neg_rules == 0:
                # applicable_rules.append([])  # no rules applicable
                found_sol.append(-1)
                probas.append(0)
            elif negative_score == 0:
                # applicable_rules.append(filter_rules_by_label(_appl_rules, self.pos_label))
                found_sol.append(1)
                probas.append(1)
            else:  # neither positive_score nor negative_score is zero:
                if negative_score < 0: print("negative score smaller 0, sus")
                try:
                    if sigmoid(math.log(positive_score / negative_score)) > decision_thr:
                        # applicable_rules.append(filter_rules_by_label(_appl_rules, self.pos_label))
                        found_sol.append(1)
                        probas.append(sigmoid(math.log(positive_score / negative_score)))
                    else:
                        # applicable_rules.append(filter_rules_by_label(_appl_rules, self.neg_label))
                        found_sol.append(0)
                        probas.append(sigmoid(math.log(positive_score / negative_score)))
                except ValueError:
                    if negative_score < positive_score:
                        # applicable_rules.append(filter_rules_by_label(_appl_rules, self.pos_label))
                        found_sol.append(1)
                        probas.append(1)
                    elif positive_score < negative_score:
                        # applicable_rules.append(filter_rules_by_label(_appl_rules, self.neg_label))
                        found_sol.append(0)
                        probas.append(0)

            if (num_pos_rules in app_pos_dict):
                app_pos_dict[num_pos_rules] += 1
            else:
                app_pos_dict[num_pos_rules] = 1

            if (num_neg_rules in app_neg_dict):
                app_neg_dict[num_neg_rules] += 1
            else:
                app_neg_dict[num_neg_rules] = 1

            applicability = num_pos_rules + num_neg_rules

            if (applicability in sum_dict):

                sum_dict[applicability] += 1
            else:
                sum_dict[applicability] = 1

        pool.close(); pool.terminate(); pool.join()
        return found_sol


        rules_rec_micro = recall_score(y, found_sol, average='micro')  # ACCURACY
        print(f'mirco rules recall: {rules_rec_micro}')
        rules_acc = recall_score(y, found_sol, average='weighted')
        print(f'rules acc: {rules_acc}')

        rules_rec = recall_score(y, found_sol, average='macro')  # recall
        print(f'macro rules recall: {rules_rec}')

        # rules_prec = precision_score(y, found_sol, average='macro')  # prec mac
        # print(f'macro rules prec: {rules_prec}')

        rules_prec = precision_score(y, found_sol, average='weighted')  # prec mac
        print(f'weight rules prec: {rules_prec}')

        rules_f1 = f1_score(y, found_sol, average='macro')  # F1
        print(f'macro rules f1_score: {rules_f1}')

        rules_roc_auc = roc_auc_score(y, found_sol)

        print(sum(y), sum(found_sol))

        # self.plot(y, probas)
        # self.draw_hist(app_pos_dict, app_neg_dict, sum_dict)
        coverage = 1 - sum_dict[0] / len(data)

        results = dict(
            found_sol=found_sol,
            rules_rec_micro=rules_rec_micro,
            rules_acc=rules_acc,
            rules_rec=rules_rec,
            rules_prec=rules_prec,
            rules_f1=rules_f1,
            rules_roc_auc=rules_roc_auc,
            coverage=coverage,
            applied_rules=applicable_rules
        )
        # return found_sol, rules_rec_micro, rules_acc, rules_rec, rules_prec, rules_f1, rules_roc_auc, coverage
        return results


    def predict_proba(self, data):

        found_sol = []

        num_cores = cpu_count() - 2
        pool = Pool(num_cores)

        scores = list(pool.imap(self.compute_scores, data.iterrows()))

        for score_tup in scores:
            print(score_tup)
            positive_score, negative_score = score_tup
            if negative_score == 0 and positive_score == 0: print('not applicable')
            if negative_score == 0 and positive_score == 0:
                found_sol.append(0)
            elif negative_score == 0:
                found_sol.append(1)
            else:
                found_sol.append(sigmoid(math.log(positive_score / negative_score)))
        return found_sol

    def predict_explain(self, data, alpha=20, beta=1, decision_thr=0.5, explain=False):
        # NOT IN ORIGINAL CEGA REPO
        self.alpha = alpha
        self.beta = beta
        self.X = data

        found_sol = []
        applicable_rules = []

        num_cores = 1
        if len(data) > 500:
            num_cores = 2#cpu_count() - 2
        pool = Pool(num_cores)

        scores = list(tqdm(pool.imap(self.compute_scores_explain, data.iterrows()),
                       total=len(data), file=sys.stdout, position=0, leave=True))

        def filter_rules_by_label(rules, lab):
            return [r for r in rules if r['label'] == lab]

        for score_tup in scores:
            positive_score, negative_score, num_pos_rules, num_neg_rules, _appl_rules = score_tup
            # if (negative_score == 0 and positive_score == 0 or
            #         negative_score==positive_score or len(_appl_rules) == 0): print('not applicable')
            if negative_score == 0 and positive_score == 0 or len(_appl_rules) == 0:
                found_sol.append(-1)
                applicable_rules.append(_appl_rules)  # no rules applicable
            elif negative_score == 0:
                found_sol.append(1)
                applicable_rules.append(filter_rules_by_label(_appl_rules, self.pos_label))
            else:  # neither positive_score nor negative_score is zero:
                if negative_score < 0: print("negative score smaller 0, sus")
                try:
                    if sigmoid(math.log(positive_score / negative_score)) > decision_thr:
                        found_sol.append(1)
                        applicable_rules.append(filter_rules_by_label(_appl_rules, self.pos_label))
                    else:
                        found_sol.append(0)
                        applicable_rules.append(filter_rules_by_label(_appl_rules, self.neg_label))
                except ValueError:
                    if negative_score < positive_score:
                        found_sol.append(1)
                        applicable_rules.append(filter_rules_by_label(_appl_rules, self.pos_label))
                    elif positive_score < negative_score:
                        found_sol.append(0)
                        applicable_rules.append(filter_rules_by_label(_appl_rules, self.neg_label))
        return found_sol, applicable_rules


    def compute_scores_explain(self, record):
        # if eval mode, not only return probas but also every rule that matched

        _, row = record
        positive_score = self.prb_pos
        negative_score = self.prb_neg

        num_pos_rules = 0
        num_neg_rules = 0

        example_set = set()

        for name, val in row.items():
            if val > 0:
                example_set.add(format_name(name, val, intervals=self.intervals_dict))
        _applicability = []

        for i, (_, rule) in enumerate(self.rules.iterrows()):
            label = rule['label']
            itemset = rule['itemset']


            if itemset.issubset(example_set):
                _applicability.append(rule)

                if label == self.pos_label:
                    num_pos_rules += 1
                    positive_score *= ((rule['support'] + self.alpha) / (self.prb_pos + self.alpha * self.beta))

                    neg_count = sum(self.oh_np_arr[:, self.ind_neg])
                    features = [list(self.ohe_df.columns).index(item) for item in list(itemset)]
                    features_support = list(np.sum(np.array(self.ohe_df[self.ohe_df[self.neg_label] == 1]
                                                            )[:, features], axis=1)).count(len(itemset))

                    opp_class_score = (features_support + self.alpha) / (neg_count + self.alpha * self.beta)
                    if opp_class_score > 0:
                        negative_score *= opp_class_score
                    else:
                        negative_score *= (self.alpha / (neg_count + self.alpha * self.beta))

                else:
                    num_neg_rules += 1
                    negative_score *= ((rule['support'] + self.alpha) / (self.prb_neg + self.alpha * self.beta))

                    pos_count = sum(self.oh_np_arr[:, self.ind_pos])
                    features = [list(self.ohe_df.columns).index(item) for item in list(itemset)]
                    features_support = list(np.sum(np.array(self.ohe_df[self.ohe_df[self.pos_label] == 1]
                                                            )[:, features], axis=1)).count(len(itemset))

                    opp_class_score = (features_support + self.alpha) / (pos_count + self.alpha * self.beta)
                    if opp_class_score > 0:
                        positive_score *= opp_class_score
                    else:
                        positive_score *= (self.alpha / (pos_count + self.alpha * self.beta))

        return (positive_score, negative_score, num_pos_rules, num_neg_rules, _applicability)


    def plot(self, y, found_sol):
        import sklearn.metrics as metrics
        import matplotlib.pyplot as plt
        fpr, tpr, thresholds = metrics.roc_curve(y, found_sol)
        auc = metrics.roc_auc_score(y, found_sol)
        plt.plot(fpr, tpr, label="RulesClassification(AUC={:.2f})".format(auc))
        plt.legend(loc=4)
        plt.show()
        print(find_Optimal_Cutoff(fpr, tpr, thresholds))

    def plot_auc(self, data, y, alpha=None, beta=None):

        if alpha is not None and beta is not None:
            self.alpha = alpha
            self.beta = beta
        found_sol = self.predict_proba(data, y)
        self.plot(y, found_sol)

    def draw_hist(self, app_pos_dict, app_neg_dict, sum_dict, axis='x'):
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        if axis == 'x':
            gs = gridspec.GridSpec(1, 3)
            x1, y1, x2, y2, x3, y3 = 0, 0, 0, 1, 0, 2
            fig = plt.figure(figsize=(15, 3))
        else:
            gs = gridspec.GridSpec(5, 1)
            x1, y1, x2, y2, x3, y3 = 0, 0, 2, 0, 4, 0
            fig = plt.figure(figsize=(11, 11))

        ax1 = fig.add_subplot(gs[x1, y1])
        ax1.bar(app_pos_dict.keys(), app_pos_dict.values(), width=0.5, color='mediumslateblue')
        ax1.set_title("Applicable Rules Histogram (Class 1)")
        ax1.set_xlabel('Count of Applicable Rules')
        ax1.set_ylabel('Number of Examples')

        ax2 = fig.add_subplot(gs[x2, y2])
        ax2.bar(app_neg_dict.keys(), app_neg_dict.values(), width=0.5, color='dodgerblue')
        ax2.set_title("Applicable Rules Histogram (Class 2)")
        ax2.set_xlabel('Count of Applicable Rules')
        ax2.set_ylabel('Number of Examples')

        ax3 = fig.add_subplot(gs[x3, y3])
        ax3.bar(sum_dict.keys(), sum_dict.values(), width=0.5, color='darkgreen')
        ax3.set_title("All Applicable Rules Histogram")
        ax3.set_xlabel('Count of Applicable Rules')
        ax3.set_ylabel('Number of Examples')

        if 0 in sum_dict:
            print(f'coverage: {1 - sum_dict[0] / len(self.X)}')
        else:
            print('coverage: 1.00')