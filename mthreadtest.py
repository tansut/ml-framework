import concurrent.futures
import time

import pandas as pd
import numpy as np

from aibrite.ml.neuralnet import NeuralNet
from aibrite.ml.neuralnetwithmomentum import NeuralNetWithMomentum
from aibrite.ml.neuralnetwithrmsprop import NeuralNetWithRMSprop
from aibrite.ml.neuralnetwithadam import NeuralNetWithAdam
from aibrite.ml.analyser import NeuralNetAnalyser


df = pd.read_csv("./data/ex2data1.csv", sep=",")

# df = df[df['quality'] != 8.0]
# df = df[df['quality'] != 3.0]

np.random.seed(5)
data = df.values
data = NeuralNet.shuffle(data)

train_set, test_set, dev_set = NeuralNet.split(data, 0.7, 0.15, 0.15)

train_x, train_y = train_set[:, 0:-1], train_set[:, -1]
train_x, train_y = (train_set[:, 0:-1]), train_set[:, -1]
dev_x, dev_y = (dev_set[:, 0:-1]), dev_set[:, -1]
test_x, test_y = (test_set[:, 0:-1]), test_set[:, -1]

labels = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

iterations = [10, 20]
learning_rates = [0.03]
hidden_layers = [(4, 6, 4, 12, 6)]
test_sets = {'dev': (dev_x, dev_y),
             'test': (test_x, test_y),
             'train': (train_x, train_y)}
# test_sets = {'train': (train_x, train_y)}


def jb(analyser, results):
    pass


analyser = NeuralNetAnalyser(log_dir='./analyserlogs', job_completed=jb)

start_time = time.time()

train_set = (train_x, train_y)

for it in iterations:
    for lr in learning_rates:
        for hl in hidden_layers:
            analyser.submit(NeuralNetWithAdam, train_set, test_sets,
                            hidden_layers=hl,
                            learning_rate=lr,
                            iteration_count=it,
                            lambd=0.4,
                            epochs=3,
                            shuffle=True,
                            minibatch_size=0)
            analyser.foo = "tansu"

analyser.start()
analyser.print_summary()
