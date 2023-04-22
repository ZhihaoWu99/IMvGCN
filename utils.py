"""
@Project   : IMvGCN
@Time      : 2021/10/4
@Author    : Zhihao Wu
@File      : utils.py
"""
import torch
import random
from texttable import Texttable
from sklearn import metrics


def tab_printer(args):
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]] + [[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())


def get_evaluation_results(labels_true, labels_pred):
    ACC = metrics.accuracy_score(labels_true, labels_pred)
    F1 = metrics.f1_score(labels_true, labels_pred, average='macro')
    return ACC, F1


def criteria(num_view, output, w_list, feature_list, flt_list, Lambda):
    loss_rec = 0.
    loss_reg = 0.
    for v in range(num_view):
        loss_rec += torch.norm(-output.mm(w_list[num_view*2+v].t()).mm(w_list[v*2+1].t()).mm(w_list[v*2].t())+feature_list[v]) ** 2
        loss_reg += torch.trace(output.t().mm(torch.spmm(flt_list[v], output)))
        torch.cuda.empty_cache()
    return loss_rec + Lambda * loss_reg


def data_split(labels, ratio):
    each_class_num = {}
    for label in labels:
        if label in each_class_num.keys():
            each_class_num[label] += 1
        else:
            each_class_num[label] = 1
    labeled_each_class_num = {}  # number of labeled samples for each class
    total_num = round(ratio * len(labels))
    for label in each_class_num.keys():
        labeled_each_class_num[label] = max(round(each_class_num[label] * ratio), 1)

    # index of labeled and unlabeled samples
    idx_labeled = []
    idx_unlabeled = []
    index = range(len(labels))
    random.shuffle(index)
    labels = labels[index]
    for idx, label in enumerate(labels):
        if (labeled_each_class_num[label] > 0):
            labeled_each_class_num[label] -= 1
            idx_labeled.append(index[idx])
            total_num -= 1
        else:
            idx_unlabeled.append(index[idx])
    return idx_labeled, idx_unlabeled
