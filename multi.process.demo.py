import concurrent.futures
import time

import pandas as pd
import numpy as np

from aibrite.ml.neuralnet import NeuralNet
from aibrite.ml.neuralnetwithmomentum import NeuralNetWithMomentum
from aibrite.ml.neuralnetwithrmsprop import NeuralNetWithRMSprop
from aibrite.ml.neuralnetwithadam import NeuralNetWithAdam
from aibrite.ml.analyser import NeuralNetAnalyser
from aibrite.ml.loggers import CsvLogger


df = pd.read_csv("./data/winequality-red.csv", sep=";")

np.random.seed(5)
data = df.values

train_set, test_set, dev_set = NeuralNet.split(
    data, 0.6, 0.20, 0.20, shuffle=True)

train_x, train_y = (train_set[:, 0:-1]), train_set[:, -1]
dev_x, dev_y = (dev_set[:, 0:-1]), dev_set[:, -1]
test_x, test_y = (test_set[:, 0:-1]), test_set[:, -1]

labels = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

normalize_inputs = [True]
iteration_count = [5, 10]
learning_rate = [0.005, 0.002]
hidden_layers = [(32, 64, 128), (4, 4)]
lambds = [0.4, 0.6, 0.8, 0.9]
learnin_rate_decay = [0.5]
epoch = [1]
shuffle = [True]
minibatch_size = [0]

test_sets = {'dev': (dev_x, dev_y),
             'test': (test_x, test_y),
             'train': (train_x, train_y)}


def jb(analyser, results):
    pass


analyser = NeuralNetAnalyser(
    "Red Wine Analysis", job_completed=jb)

train_set = (train_x, train_y)

for it in iteration_count:
    for lr in learning_rate:
        for hl in hidden_layers:
            for lambd in lambds:
                for ni in normalize_inputs:
                    for lrd in learnin_rate_decay:
                        for ep in epoch:
                            for shl in shuffle:
                                for mbs in minibatch_size:
                                    analyser.submit(NeuralNetWithAdam, train_set, test_sets,
                                                    hidden_layers=hl,
                                                    learning_rate=lr,
                                                    learning_rate_decay=lrd,
                                                    iteration_count=it,
                                                    lambd=lambd,
                                                    normalize_inputs=ni,
                                                    epochs=ep,
                                                    shuffle=shl,
                                                    minibatch_size=mbs,
                                                    labels=labels)

analyser.join()
analyser.print_summary()
