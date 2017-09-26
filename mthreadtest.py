import concurrent.futures
import time

import pandas as pd
import numpy as np

from aibrite.ml.neuralnet import NeuralNet
from aibrite.ml.neuralnetwithmomentum import NeuralNetWithMomentum
from aibrite.ml.neuralnetwithrmsprop import NeuralNetWithRMSprop
from aibrite.ml.neuralnetwithadam import NeuralNetWithAdam

df = pd.read_csv("./data/winequality-red.csv", sep=";")

# df = df[df['quality'] != 8.0]
# df = df[df['quality'] != 3.0]

# print(df.values.shape)

# np.random.seed(1)

data = df.values
# data = NeuralNet.shuffle(data)

train_set, dev_set, test_set = NeuralNet.split(data, 0.6, 0.2, 0.2)

train_x, train_y = train_set[:, 0:-1], train_set[:, -1]
train_x, train_y = NeuralNet.zscore(train_set[:, 0:-1]), train_set[:, -1]
dev_x, dev_y = NeuralNet.zscore(dev_set[:, 0:-1]), dev_set[:, -1]
test_x, test_y = NeuralNet.zscore(test_set[:, 0:-1]), test_set[:, -1]

labels = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

iterations = [1000]
learning_rates = [0.008]
hidden_layers = [(24, 36, 24, 12, 6)]
test_sets = {'dev': (dev_x, dev_y), 'test': (
    test_x, test_y), 'train': (train_x, train_y)}

# test_sets = {'train': (train_x, train_y)}

future_list = []
start_time = time.time()
best_nns, best_reports, best_test_sets, best_f1 = [], [], [], -1


def predict_it(train_result, test_id, test_set):
    nn, tr, train_time = result

    predict_start = time.time()
    prediction_result = nn.predict(test_set[0])
    pred_time = time.time() - predict_start

    report = NeuralNet.score_report(test_set[1], prediction_result.predicted)
    print("[{id}][{train_time:3.1f}/{pred_time:3.1f}]:{nn}:\n{report}\n".format(id=test_id,
                                                                                train_time=train_time,
                                                                                pred_time=pred_time,
                                                                                nn=nn,
                                                                                report=NeuralNet.format_score_report(report)))

    return prediction_result, report, pred_time


def test_it(nn_class, train_x, train_y, **kvargs):
    nn = nn_class(train_x, train_y, **kvargs)

    train_start = time.time()
    train_result = nn.train()
    training_time = time.time() - train_start

    return nn, train_result, training_time


with concurrent.futures.ProcessPoolExecutor() as executor:
    print(
        "Starting test with {0} max-workers...".format(executor._max_workers))
    for it in iterations:
        for lr in learning_rates:
            for hl in hidden_layers:
                e = executor.submit(test_it, NeuralNetWithAdam, train_x, train_y,
                                    hidden_layers=hl,
                                    learning_rate=lr,
                                    iteration_count=it,
                                    lambd=0.001,
                                    epochs=3,
                                    shuffle=True,
                                    minibatch_size=64
                                    # labels=labels
                                    )
                future_list.append(e)
    for future in concurrent.futures.as_completed(future_list):
        try:
            result = future.result()
        except Exception as exc:
            print(exc)
            future_list.remove(future)
        else:

            for i, v in test_sets.items():
                pred_result, report, pred_time = predict_it(result, i, v)
                if (best_f1 <= report.totals[2]):
                    best_f1 = report.totals[2]
                    best_nns.append(result[0])
                    best_reports.append(report)
                    best_test_sets.append(i)

            future_list.remove(future)

            if len(future_list) == 0:
                print("-" * 40)
                print("RESULTS")
                print("-" * 40)
                print("Completed: {0:.2f} seconds\n".format(
                    time.time() - start_time))
                print(
                    "It seems best f1 is {f1:.2f} with {nn} configuration(s)\n".format(f1=best_f1, nn=len(best_nns)))
                for i, rep in enumerate(best_reports):
                    print("[{0}]{1}\n{2}".format(best_test_sets[i],
                                                 best_nns[i], NeuralNet.format_score_report(rep)))
                print("-" * 40)
