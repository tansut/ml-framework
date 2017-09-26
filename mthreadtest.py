import concurrent.futures
import time

import pandas as pd

from aibrite.ml.neuralnet import NeuralNet
from aibrite.ml.neuralnetwithmomentum import NeuralNetWithMomentum
from aibrite.ml.neuralnetwithrmsprop import NeuralNetWithRMSprop
from aibrite.ml.neuralnetwithadam import NeuralNetWithAdam

df = pd.read_csv("./data/winequality-red.csv", sep=";")

train_set, dev_set, test_set = NeuralNet.split(df.values, 0.8, 0.1, 0.1)

train_x, train_y = train_set[:, 0:-1], train_set[:, -1]
dev_x, dev_y = dev_set[:, 0:-1], dev_set[:, -1]
test_x, test_y = test_set[:, 0:-1], test_set[:, -1]


# test configurations
iterations = [500, 700]
learning_rates = [0.02]
hidden_layers = [(6,), (6, 9)]


def test_it(nn_class, train_x, train_y, test_x, test_y, **kvargs):
    nn = nn_class(train_x, train_y, **kvargs)
    train_result = nn.train()
    prediction_result = nn.predict(test_x)

    return nn, train_result, prediction_result


executor_list = []

with concurrent.futures.ThreadPoolExecutor() as executor:
    print("Starting test with {0} workers...", .format(executor._max_workers))
    for it in iterations:
        for lr in learning_rates:
            for hl in hidden_layers:
                e = executor.submit(test_it, NeuralNetWithRMSprop, train_x, train_y, test_x, test_y,
                                    hidden_layers=hl,
                                    learning_rate=lr,
                                    iteration_count=it
                                    )
                executor_list.append(e)
    for test in concurrent.futures.as_completed(executor_list):
        try:
            result = test.result()
        except Exception as exc:
            print(exc)
        else:
            nn, train_result, prediction_result = result
            report = NeuralNet.score_report(
                test_y, prediction_result.predicted)
            print("{0}:\n{1}\n".format(
                nn, NeuralNet.format_score_report(report)))
