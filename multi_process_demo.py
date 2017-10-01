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
from aibrite.ml.loggers import MongodbLogger


df = pd.read_csv("./data/winequality-red.csv", sep=";")

# df = df[df['quality'] != 8.0]
# df = df[df['quality'] != 3.0]

np.random.seed(5)
data = df.values

train_set, test_set, dev_set = NeuralNet.split(
    data, 0.6, 0.20, 0.20, shuffle=True)

train_x, train_y = (train_set[:, 0:-1]), train_set[:, -1]
dev_x, dev_y = (dev_set[:, 0:-1]), dev_set[:, -1]
test_x, test_y = (test_set[:, 0:-1]), test_set[:, -1]

labels = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

normalize_inputs = [True]
iteration_count = [30, 90, 120]
learning_rate = [0.005, 0.003]
hidden_layers = [(12, 6, 24, 12)]
lambds = [0.2, 0.4]
learnin_rate_decay = [0.5]
epoch = [1, 2]
shuffle = [True]
minibatch_size = [0]

test_sets = {'dev_set': (dev_x, dev_y),
             'test_set': (test_x, test_y),
             'train_set': (train_x, train_y)}
# test_sets = {'train_set': (train_x, train_y)}
# test_sets = {'test_set': (test_x, test_y)}


def jb(analyser, results):
    pass


# logger = CsvLogger("Coursera Ex2 Analysis", overwrite=True,
#                    base_dir='./analyserlogs')

analyser = NeuralNetAnalyser(
    "Coursera Ex2 Analysis", job_completed=jb)

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

if len(test_sets) > 1:
    selected = analyser.get_testset_from_user()
    analyser.print_summary(selected)
else:
    analyser.print_summary()
