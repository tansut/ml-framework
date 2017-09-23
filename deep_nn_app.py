from deep_nn import DeepNN
from deep_nn_momentum import DeepNNMomentum
from deep_nn_rmsprop import DeepNNRMSprop
from deep_nn_adam import DeepNNAdam
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
from ml import LearningAlgorithm
from nn_test_thread import NNTestThread


from nntest import NNTester


df = pd.read_csv("./data/winequality-red.csv", sep=";")
# df = pd.read_csv("ex2data1.txt", sep=",")

# split as train, test, validation (%80, %10, %10)
split_data = LearningAlgorithm.split(df.values, 0.8, 0.1, 0.1)

train_x, train_y = split_data[0][:, 0:-1].T, split_data[0][:, -1:].T
test_x, test_y = split_data[1][:, 0:-1].T, split_data[1][:, -1:].T
val_x, val_y = split_data[2][:, 0:-1].T, split_data[2][:, -1:].T


iterations = [16500]
learning_rates = [0.02]
hidden_layers = [[6]]


result_list = []
start_time = time.time()

tester = NNTester()

with tester.getExecutor() as executor:
    for it in iterations:
        for lr in learning_rates:
            for hl in hidden_layers:
                tester.submit("it:{}, lr:{}, hl:{}".format(it, lr, hl), DeepNNMomentum, train_x, train_y, test_x, test_y,
                              hidden_layers=hl,
                              learning_rate=lr,
                              iteration_count=it
                              )
    for test in tester.as_completed():
        try:
            data = test.result()
        except Exception as exc:
            print(exc)
        else:
            result_list.append(data)
            pred_result = data['pred_result']
            nn = data['nn']
            print("id:{} success: {:.2f}".format(
                data['id'], pred_result['rate']))


print("%f seconds" % (time.time() - start_time))
