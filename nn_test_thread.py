import threading
import time
import matplotlib.pyplot as plt

exitFlag = 0


class NNTestThread (threading.Thread):
    def __init__(self, threadID, nn_class, train_x, train_y, test_x, test_y, **kvargs):
        threading.Thread.__init__(self)
        self.threadID = threadID

        self.nn_obj = nn_class(train_x, train_y, **kvargs)

        self.test_x = test_x
        self.test_y = test_y

        # self.nn_class = nn_class
        # self.parameters = parameters

    def plotCost(self, costs, figNum=None, title='', show=True):
        # plt.figure(figNum)
        plt.title(title)
        plt.plot(costs)
        plt.xlabel("iteration")
        plt.show() if show == True else None

    def train_cb(self, i, cost):
        pass

    def run(self):
        print ("Starting train ..." + str(self.threadID))
        train_result = self.nn_obj.train(
            train_cb=lambda i, cost: self.train_cb(i, cost))
        pred_result = self.nn_obj.predict_and_test(
            self.test_x, self.test_y)
        self.prediction_result = pred_result
        print ("tread: {}, it:{}, lr:{}, nn:{} success: {:.2f}".format(self.threadID, self.nn_obj.iteration_count,
                                                                       self.nn_obj.learning_rate, self.nn_obj.hidden_layers, pred_result['rate']))
