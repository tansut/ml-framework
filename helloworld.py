import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# train_set, dev_set, test_set = NeuralNet.split(df.values, 0.8, 0.1, 0.1)

# train_x, train_y = train_set[:, 0:-1].T, train_set[:, -1:].T
# dev_x, dev_y = dev_set[:, 0:-1].T, dev_set[:, -1:].T
# test_x, test_y = test_set[:, 0:-1].T, test_set[:, -1:].T


def confusion_matrix(expect, pred, labels=None):
    if (labels == None):
        labels = np.union1d(expect, pred)
    m = [[0] * len(labels) for l in labels]
    index = {v: i for i, v in enumerate(labels)}
    for p, t in zip(expect, pred):
        m[index[p]][index[t]] += 1

    return m


def calc_accuracy(conf_matrix):
    t = sum(sum(l) for l in conf_matrix)
    return sum(conf_matrix[i][i] for i in range(len(conf_matrix))) / t


def calc_recall(cm):
    tp = [cm[i][i] for i in range(len(cm))]
    sm = [sum(l) for l in cm]
    return [t / s if s > 0 else 0. for t, s in zip(tp, sm)]


def calc_precision(cm):
    tp = [cm[i][i] for i in range(len(cm))]
    t = [[row[i] for row in cm] for i in range(len(cm[0]))]
    sm = [sum(l) for l in t]
    return [t / s if s > 0 else 0. for t, s in zip(tp, sm)]


def calc_f1(cm):
    p = np.asarray(calc_precision(cm))
    r = np.asarray(calc_recall(cm))
    return np.nan_to_num(2 * (r * p) / (r + p), )


def support(cm):
    c = np.sum(cm, axis=1)
    return c


(train_x, train_y), (test_x, test_y), (dev_x, dev_y) = get_datasets()


def analyse(classifier, expected, predicted):
    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" %
          metrics.confusion_matrix(expected, predicted))
    print("Accuracy: {0}".format(accuracy_score(expected, predicted)))
    print("------------------")
    mat = confusion_matrix(expected, predicted)
    print(mat)
    print(calc_accuracy(mat))
    print(calc_precision(mat))
    print(calc_recall(mat))
    print(calc_f1(mat))
    print(support(mat))


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
                               hidden_layers=[9],
                               iteration_count=30,
                               learning_rate=0.001,
                               minibatch_size=25000,
                               epochs=1,
                               learning_rate_decay=0.2,
                               # beta1=0.8,
                               shuffle=True)
print("train NN")
classifier.train(train_cb)

predict = classifier.predict_and_test(test_x, test_y)

expected = np.squeeze(test_y.T)
predicted = np.squeeze(predict["pred"].T)

analyse(classifier, expected, predicted)

# classifier = svm.SVC(gamma=0.001)

# classifier.fit(np.squeeze(train_x.T), np.squeeze(train_y.T))

# predicted = classifier.predict(np.squeeze(test_x.T))

# analyse(classifier, expected, predicted)

# print("train MLp")

# classifier = MLPClassifier(hidden_layer_sizes=(9))

# classifier.fit(np.squeeze(train_x.T), np.squeeze(train_y.T))

# predicted = classifier.predict(np.squeeze(test_x.T))

# analyse(classifier, expected, predicted)


# print("succ%: {0:.2f}".format(predict['rate']))


# def plot_graph(*args):
#     # print(args[0])
#     display_costs(args[0])
#     show()


# p = Process(target=plot_graph, args=([costs]))
# p.start()


# p.join()


# display_costs(costs, predict)
