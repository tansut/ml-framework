from deep_nn import DeepNN
from deep_nn_momentum import DeepNNMomentum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ml import LearningAlgorithm
from nn_test_thread import NNTestThread
import concurrent.futures
import time


df = pd.read_csv("winequality.txt", sep=";")
# df = pd.read_csv("ex2data1.txt", sep=",")

# split as train, test, validation (%80, %10, %10)
split_data = LearningAlgorithm.split(df.values, 0.8, 0.1, 0.1)

train_x, train_y = split_data[0][:, 0:-1].T, split_data[0][:, -1:].T
test_x, test_y = split_data[1][:, 0:-1].T, split_data[1][:, -1:].T
val_x, val_y = split_data[2][:, 0:-1].T, split_data[2][:, -1:].T

iterations = [15000]
learning_rates = [0.04, 0.05, 0.06]
hidden_layers = [[6], [12]]


def test_it(nn_class, train_x, train_y, test_x, test_y, **kvargs):
    nn = nn_class(train_x, train_y, **kvargs)
    train_result = nn.train()
    pred_result = nn.predict_and_test(test_x, test_y)
    return {
        'pred_result': pred_result,
        'train_result': train_result,
        'nn': nn
    }


thread_list = []
result_list = []
start_time = time.time()

with concurrent.futures.ThreadPoolExecutor() as executor:
    for it in iterations:
        for lr in learning_rates:
            for hl in hidden_layers:
                e = executor.submit(test_it, DeepNNMomentum, train_x, train_y, test_x, test_y,
                                    hidden_layers=hl,
                                    learning_rate=lr,
                                    iteration_count=it
                                    )
                thread_list.append(e)
    for future in concurrent.futures.as_completed(thread_list):
        try:
            data = future.result()
        except Exception as exc:
            print(exc)
        else:
            result_list.append(data)
            pred_result = data['pred_result']
            nn = data['nn']
            print ("it:{}, lr:{}, nn:{} success: {:.2f}".format(
                nn.iteration_count, nn.learning_rate, nn.hidden_layers, pred_result['rate']))


print("%f seconds" % (time.time() - start_time))
