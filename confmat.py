import numpy as np
from sklearn import datasets, svm, metrics


def confusion_matrix(expect, pred, labels=None):
    if (labels == None):
        labels = np.unique(expect + pred)
    m = [[0] * len(labels) for l in labels]
    index = {v: i for i, v in enumerate(labels)}
    for p, t in zip(expect, pred):
        m[index[p]][index[t]] += 1

    return m


def calc_accuracy(conf_matrix):
    t = sum(sum(l) for l in conf_matrix)
    return sum(conf_matrix[i][i] for i in range(len(conf_matrix))) / t


def calc_recall(cm):
    tp = [cm[i][i] for i in range(len(cm))]
    sm = [sum(l) for l in cm]
    return [t / s if s > 0 else 0. for t, s in zip(tp, sm)]


def calc_precision(cm):
    tp = [cm[i][i] for i in range(len(cm))]
    t = [[row[i] for row in cm] for i in range(len(cm[0]))]
    sm = [sum(l) for l in t]
    return [t / s if s > 0 else 0. for t, s in zip(tp, sm)]


def calc_f1(cm):
    p = np.asarray(calc_precision(cm))
    r = np.asarray(calc_recall(cm))
    return np.nan_to_num(2 * (r * p) / (r + p), )


def support(cm):
    c = np.sum(cm, axis=1)
    return c


# _true = ['e', 'e', 'x', 'c', 'd', 'd', 'e']
# _pred = ['e', 'x', 'e', 'e', 'd', 'e', 'x']

# mat = confusion_matrix(_true, _pred)
# print(np.asarray(mat))

# mat2 = metrics.confusion_matrix(_true, _pred)
# print(mat2)

# print(mat == mat2)

# print("Classification report for classifier :\n%s\n"
#       % (metrics.classification_report(_true, _pred)))

# print(calc_accuracy(mat))
# print(calc_precision(mat))
# print(calc_recall(mat))
# print(calc_f1(mat))
# print(support(mat))
