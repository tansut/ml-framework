import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from aibrite.ml.neuralnet import NeuralNet
from aibrite.ml.neuralnetwithmomentum import NeuralNetWithMomentum
from aibrite.ml.neuralnetwithrmsprop import NeuralNetWithRMSprop
from aibrite.ml.neuralnetwithadam import NeuralNetWithAdam

from multiprocessing import Process
# from matplotlib.pyplot import plot, show

from mnist import get_datasets
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import datasets, svm, metrics
from sklearn.neural_network import MLPClassifier

df = pd.read_csv("./data/winequality-red.csv", sep=";")

train_set, dev_set, test_set = NeuralNet.split(df.values, 0.8, 0.1, 0.1)

train_x, train_y = train_set[:, 0:-1], train_set[:, -1]
dev_x, dev_y = dev_set[:, 0:-1], dev_set[:, -1]
test_x, test_y = test_set[:, 0:-1], test_set[:, -1]

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


classifier = NeuralNetWithAdam(train_x, train_y,
                               hidden_layers=(6,),
                               iteration_count=1000,
                               learning_rate=0.005,
                               minibatch_size=64,
                               epochs=3,
                               learning_rate_decay=0.2,
                               # beta1=0.8,
                               shuffle=True)

train_result = classifier.train(train_cb)
prediction = classifier.predict(test_x)
report = NeuralNet.score_report(test_y, prediction.predicted)
print("Classification report for {0}:\n{1}\n".format(
    classifier, NeuralNet.format_score(report)))


classifier = svm.SVC(gamma=0.001)
classifier.fit(train_x, train_y)
predicted = classifier.predict(test_x)
report = NeuralNet.score_report(test_y, predicted)
print("Classification report for {0}:\n{1}\n".format(
    classifier, NeuralNet.format_score(report)))


# analyse(classifier, expected, predicted)

# print("train MLp")

# classifier = MLPClassifier(hidden_layer_sizes=(9))

# classifier.fit(np.squeeze(train_x.T), np.squeeze(train_y.T))

# predicted = classifier.predict(np.squeeze(test_x.T))

# analyse(classifier, expected, predicted)


#print("succ%: {0:.2f}".format(predict['rate']))


# def plot_graph(*args):
#     # print(args[0])
#     display_costs(args[0])
#     show()


# p = Process(target=plot_graph, args=([costs]))
# p.start()


# p.join()


# display_costs(costs, predict)
