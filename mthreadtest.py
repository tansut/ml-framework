import concurrent.futures
import time

import pandas as pd
import numpy as np

from aibrite.ml.neuralnet import NeuralNet
from aibrite.ml.neuralnetwithmomentum import NeuralNetWithMomentum
from aibrite.ml.neuralnetwithrmsprop import NeuralNetWithRMSprop
from aibrite.ml.neuralnetwithadam import NeuralNetWithAdam

from aibrite.ml.analyser import Analyser

df = pd.read_csv("./data/winequality-red.csv", sep=";")

df = df[df['quality'] != 8.0]
df = df[df['quality'] != 3.0]

np.random.seed(5)
data = df.values
data = NeuralNet.shuffle(data)

train_set, test_set, dev_set = NeuralNet.split(data, 0.7, 0.15, 0.15)

train_x, train_y = train_set[:, 0:-1], train_set[:, -1]
train_x, train_y = (train_set[:, 0:-1]), train_set[:, -1]
dev_x, dev_y = (dev_set[:, 0:-1]), dev_set[:, -1]
test_x, test_y = (test_set[:, 0:-1]), test_set[:, -1]

labels = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

iterations = [2500]
learning_rates = [0.003, 0.002, 0.001]
hidden_layers = [(24, 36, 24, 12, 6)]
test_sets = {'dev': (dev_x, dev_y),
             'test': (test_x, test_y),
             'train': (train_x, train_y)}


analyser = Analyser()

# test_sets = {'train': (train_x, train_y)}

future_list = []
start_time = time.time()
best_nns, best_reports, best_test_sets, best_f1 = [], [], [], -1


def print_result(df):
    df = df[['classifier', 'test_set', 'f1', 'iteration_count', 'hidden_layers', 'learning_rate']].sort_values(
        ['f1'], ascending=False)
    with pd.option_context('expand_frame_repr', False):
        print(df)


def predict(neuralnet, test_id, test_set):
    test_input, expected = test_set
    neuralnet.predict(test_input)
    predicted = neuralnet.prediction_result.predicted
    hyper_parameters = neuralnet.get_hyperparameters()

    analyser.add_to_prediction_log(neuralnet.__class__.__name__, test_id, expected, predicted, hyper_parameters,
                                   extra_data={
                                       'train_time': neuralnet.train_result.elapsed(),
                                       'pred_time': neuralnet.prediction_result.elapsed()
                                   })


def train(neuralnet_class, train_x, train_y, **kvargs):
    neuralnet = neuralnet_class(train_x, train_y, **kvargs)
    neuralnet.train()
    return neuralnet


with concurrent.futures.ProcessPoolExecutor() as executor:
    print("Starting with {0} max-workers...".format(executor._max_workers))
    for it in iterations:
        for lr in learning_rates:
            for hl in hidden_layers:
                e = executor.submit(train, NeuralNetWithAdam, train_x, train_y,
                                    hidden_layers=hl,
                                    learning_rate=lr,
                                    iteration_count=it,
                                    lambd=0.4,
                                    epochs=3,
                                    shuffle=True,
                                    minibatch_size=0
                                    # labels=labels
                                    )
                future_list.append(e)
    for future in concurrent.futures.as_completed(future_list):
        try:
            neuralnet = future.result()
        except Exception as exc:
            print(exc)
            future_list.remove(future)
        else:
            for i, v in test_sets.items():
                predict(neuralnet, i, v)
                # if (best_f1 <= report.totals[2]):
                #     best_f1 = report.totals[2]
                #     best_nns.append(result[0])
                #     best_reports.append(report)
                #     best_test_sets.append(i)

            future_list.remove(future)

            if (len(future_list) == 0):
                analyser.to_csv('./analyse_results.csv')
<< << << < HEAD
    print_result(analyser.df)
== == == =
    df = analyser.prediction_log[['classifier', 'test_set', 'f1', 'iteration_count', 'hidden_layers', 'learning_rate']].sort_values(
        ['f1'], ascending=False)
    print(df)
>>>>>> > 2a3dce3cfdd6e468b595346e61fb11ef0ac59871

# if len(future_list) == 0:

#     print(
#         analyser.df[['classifier', 'test_set', 'f1']].sort_values(['f1'], ascending=False))
#     print("-" * 40)
#     print("RESULTS")
#     print("-" * 40)
#     print("Completed: {0:.2f} seconds\n".format(
#         time.time() - start_time))
#     print(
#         "It seems best f1 is {f1:.2f} with {nn} configuration(s)\n".format(f1=best_f1, nn=len(best_nns)))
#     for i, rep in enumerate(best_reports):
#         print("[{0}]{1}\n{2}".format(best_test_sets[i],
#                                      best_nns[i], NeuralNet.format_score_report(rep)))
#     print("-" * 40)
