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

# train, test, validation
split_data = LearningAlgorithm.split(df.values, 0.8, 0.1, 0.1)

train_x, train_y = split_data[0][:, 0:-1].T, split_data[0][:, -1:].T
test_x, test_y = split_data[1][:, 0:-1].T, split_data[1][:, -1:].T
val_x, val_y = split_data[2][:, 0:-1].T, split_data[2][:, -1:].T


iterations = [20000, 25000]
learning_rates = [0.05, 0.005]
hidden_layers = [[2], [6]]

# threads = []
# thread_counter = 0

# plt.ion()
# fig = plt.figure(1)
# ax1 = fig.add_subplot(1, 1, 1)

# costs = []
# cost_iterations = []


def test_it(nn_class, train_x, train_y, test_x, test_y, **kvargs):
    nn = nn_class(train_x, train_y, **kvargs)
    train_result = nn.train()
    pred_result = nn.predict_and_test(test_x, test_y)
    return {
        'pred_result': pred_result,
        'train_result': train_result,
        'nn': nn
    }

# def train_cb(i, cost):
#     print("it: {}, cost: {}".format(i, cost)) if i % 1000 == 0 else None
#     # costs.append(cost)
#     # ax1.plot(costs)

#     # plt.pause(0.0005)  # if i % 100 == 0 else None


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


# for it in iterations:
#     for lr in learning_rates:
#         for hl in hidden_layers:
#             # run_params.append({
#             #     'class': DeepNNMomentum,
#             #     'train_x': train_x,
#             #     'train_y': train_y,
#             #     'test_x': test_x,
#             #     'test_y': test_y,
#             #     'kvargs': {}
#             # })
#             nn_thread = NNTestThread(thread_counter, DeepNNMomentum, train_x, train_y, test_x, test_y,
#                                      hidden_layers=hl,
#                                      learning_rate=lr,
#                                      iteration_count=it
#                                      )

#             threads.append(nn_thread)
#             thread_counter += 1
#             nn_thread.start()

# nn = DeepNNMomentum(train_x, train_y, hidden_layers=hl,
#                     learning_rate=lr, iteration_count=it)
# train_result = nn.train(train_cb=train_cb)
# pred = nn.predict_and_test(test_x, test_y)
# print('It: {} Lr: {} Layers: {} Success Rate: {:.2f}'.format(
#     it, lr, hl, pred['rate']))


# for t in threads:
#     t.join()

print("%f seconds" % (time.time() - start_time))
