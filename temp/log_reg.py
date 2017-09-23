import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ml import LearningAlgorithm


class LogisticRegression(LearningAlgorithm):

    def activate(w, b, x):
        return LearningAlgorithm.sigmoid(w.T.dot(x) + b)

    def loss(a, y):
        return - y * np.log(a) - (1 - y) * np.log(1 - a)

    def cost(a, y):
        losses = LogisticRegression.loss(a, y)
        m = y.shape[1]
        return (1.0 / m) * (np.sum(losses, axis=1)).reshape(y.shape[0], 1)

    def grads(a, y, x, lr):
        m = y.shape[1]
        return (1. / m) * ((a - y).dot(x.T)).T

    def biasgrad(a, y):
        return np.sum(a - y, axis=1).reshape(y.shape[0], 1)

    def predict(self, w, b, x):
        res = LogisticRegression.activate(w, b, x.T)
        return res

    def test(self, w, b, x, y):
        res = self.predict(w, b, x)
        maxindexes = np.argmax(res[:, :], axis=0)
        predictedValues = np.array([self.yclasses[v]
                                    for v in maxindexes]).reshape((y.shape[0], 1))
        predResult = np.sum(predictedValues == y)

        return dict(success=predResult, total=y.shape[0])

    def _init(self):
        pass

    def __init__(self, x, y, lr=0.001):
        self.initial_x = x
        self.initial_y = y
        self.lr = lr
        self.yclasses = np.unique(self.initial_y).T
        self.c = self.yclasses.shape[0]

        self.x = self.initial_x.T
        self.y = (self.initial_y == self.yclasses).astype(int).T

        self.n = self.x.shape[0]
        self.m = self.x.shape[1]

        self.initial_w = np.zeros((self.n, self.c))
        self.initial_b = np.zeros((self.c, 1))

    def train(self, iterate=25):
        costs = []
        w = self.initial_w[:]
        b = self.initial_b[:]
        for i in range(iterate):
            a = LogisticRegression.activate(w, b, self.x)
            cv = LogisticRegression.cost(a, self.y)
            gv = LogisticRegression.grads(a, self.y, self.x, self.lr)
            bg = LogisticRegression.biasgrad(a, self.y)
            w = w - self.lr * gv
            b = b - self.lr * bg
            costs.append(cv)

        return dict(costs=costs, w=w, b=b)
