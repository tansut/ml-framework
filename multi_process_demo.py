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
data = NeuralNet.shuffle(data)

train_set, test_set, dev_set = NeuralNet.split(data, 0.7, 0.15, 0.15)

train_x, train_y = NeuralNet.zscore(train_set[:, 0:-1]), train_set[:, -1]
dev_x, dev_y = (dev_set[:, 0:-1]), dev_set[:, -1]
test_x, test_y = NeuralNet.zscore(test_set[:, 0:-1]), test_set[:, -1]

labels = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

iterations = [1600]
learning_rates = [0.002]
hidden_layers = [(12, 24)]
lambds = [0.4]
test_sets = {'dev': (dev_x, dev_y),
             'test': (test_x, test_y),
             'train': (train_x, train_y)}
test_sets = {'train': (train_x, train_y)}
# test_sets = {'test': (test_x, test_y)}


def jb(analyser, results):
    pass


# logger = CsvLogger("Coursera Ex2 Analysis", overwrite=True,
#                    base_dir='./analyserlogs')

analyser = NeuralNetAnalyser(
    "Coursera Ex2 Analysis", job_completed=jb)

train_set = (train_x, train_y)

for it in iterations:
    for lr in learning_rates:
        for hl in hidden_layers:
            for lambd in lambds:
                analyser.submit(NeuralNetWithAdam, train_set, test_sets,
                                hidden_layers=hl,
                                learning_rate=lr,
                                iteration_count=it,
                                lambd=lambd,
                                epochs=1,
                                shuffle=True,
                                minibatch_size=0,
                                labels=labels)

analyser.join()
analyser.print_summary()
