import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from aibrite.ml.neuralnet import NeuralNet
from aibrite.ml.neuralnetwithmomentum import NeuralNetWithMomentum
from aibrite.ml.neuralnetwithrmsprop import NeuralNetWithRMSprop
from aibrite.ml.neuralnetwithadam import NeuralNetWithAdam

from multiprocessing import Process
from matplotlib.pyplot import plot, show

from mnist import get_datasets


df = pd.read_csv("./data/winequality-red.csv", sep=";")

train_set, dev_set, test_set = NeuralNet.split(df.values, 0.8, 0.1, 0.1)

# train_x, train_y = train_set[:, 0:-1].T, train_set[:, -1:].T
# dev_x, dev_y = dev_set[:, 0:-1].T, dev_set[:, -1:].T
# test_x, test_y = test_set[:, 0:-1].T, test_set[:, -1:].T


(train_x, train_y), (test_x, test_y), (dev_x, dev_y) = get_datasets()


costs = {}


def train_cb(cost, epoch, current_batch_index, total_batch_index, iteration):
    if (iteration % 10 == 0):
        # print("{epoch:<4} {current_batch:<6} {iteration:<6} {cost:8.4f}".format(
        #     epoch=epoch,
        #     current_batch=current_batch_index,
        #     cost=cost, iteration=iteration))
        if (costs.get(epoch) == None):
            costs[epoch] = {current_batch_index: ([], [])}
        elif costs[epoch].get(current_batch_index) == None:
            (costs[epoch])[current_batch_index] = ([], [])
        (costs[epoch][current_batch_index])[0].append(iteration)
        (costs[epoch][current_batch_index])[1].append(cost)


def display_costs(costs, predict):
    for epoch, batches in costs.items():
        plt.figure(epoch)
        plt.title("epoch: {epoch:<10} Succ: {rate:.2f}".format(
            epoch=epoch, rate=predict['rate']))
        for batch, xy in batches.items():
            plt.plot(xy[0], xy[1], label="batch {batch}".format(batch=batch))
            plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
            # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
            #            fancybox=True, shadow=True, ncol=len(costs[epoch]))
    show()
    # plt.show(block=False)


nn = NeuralNetWithAdam(train_x, train_y,
                       hidden_layers=[18],
                       iteration_count=300,
                       learning_rate=0.001,
                       minibatch_size=25000,
                       epochs=1,
                       learning_rate_decay=0.2,
                       # beta1=0.8,
                       shuffle=True)

nn.train(train_cb)


predict = nn.predict_and_test(test_x, test_y)

print("succ%: {0:.2f}".format(predict['rate']))


# def plot_graph(*args):
#     # print(args[0])
#     display_costs(args[0])
#     show()


# p = Process(target=plot_graph, args=([costs]))
# p.start()


# p.join()


display_costs(costs, predict)
