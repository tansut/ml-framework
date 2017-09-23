import numpy as np
import pandas as pd

from aibrite.ml.neuralnet import NeuralNet
from aibrite.ml.neuralnetwithmomentum import NeuralNetWithMomentum

df = pd.read_csv("./data/winequality-red.csv", sep=";")

train_set, test_set, valid_set = NeuralNet.split(df.values, 0.8, 0.1, 0.1)

train_x, train_y = train_set[:, 0:-1].T, train_set[:, -1:].T
test_x, test_y = test_set[:, 0:-1].T, test_set[:, -1:].T
val_x, val_y = valid_set[:, 0:-1].T, valid_set[:, -1:].T

nn = NeuralNet(train_x, train_y,
               hidden_layers=[6],
               iteration_count=30000,
               learning_rate=0.02,
               minibatch_size=100,
               epochs=3)

nn.train(lambda i, cost: print(
    "{0:<6} {1:8.4f}".format(i, cost)) if (i % 100 == 0) else None)

predict = nn.predict_and_test(test_x, test_y)

print("succ%: {0:.2f}".format(predict['rate']))
