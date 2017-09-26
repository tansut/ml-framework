import numpy as np


class Analyse:

    def confusion_matrix(expect, pred, labels=None):
        if (labels == None):
            labels = np.union1d(expect, pred)
        m = [[0] * len(labels) for l in labels]
        index = {v: i for i, v in enumerate(labels)}
        for e, p in zip(expect, pred):
            m[index[e]][index[p]] += 1

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
        return np.nan_to_num(2 * (r * p) / (r + p))

    def support(cm):
        c = np.sum(cm, axis=1)
        return c

    def report(expect, pred):
        c
