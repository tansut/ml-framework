from deep_nn import DeepNN
from deep_nn_momentum import DeepNNMomentum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ml import LearningAlgorithm


def process_input(data_list):
    inp, out = [], []
    for i, v in enumerate(data_list):
        inp.append(np.array(v[:, 0:v.shape[1] - 1]).T)
        out.append(np.array(v[:, v.shape[1] - 1:v.shape[1]]).T)

    return inp, out


df = pd.read_csv("winequality.txt", sep=";")
#data_orig = pd.read_csv("ex2data1.txt", sep=",")

split_data = LearningAlgorithm.split(df.values, 0.8, 0.1, 0.1)


input_data, output_data = process_input(split_data)

train_x, train_y = input_data[0], output_data[0]
test_x, test_y = input_data[1], output_data[1]
val_x, val_y = input_data[2], output_data[2]

iterations = [16000]
learning_rates = [0.005]
hidden_layers = [[6, 12]]


for it in iterations:
    for lr in learning_rates:
        for hl in hidden_layers:
            nn = DeepNNMomentum(train_x, train_y, hidden_layers=hl,
                                learning_rate=lr, iteration_count=it)
            train_result = nn.train()
            prediction_result = nn.predict(test_x)
            success = np.sum(
                prediction_result['predictions'] == test_y)
            print('Iterations: {0} Lr: {1} Layers: {2} Success Rate: {3:.2f}, Last Cost: {4:.5f}'.format(
                it, lr, hl, (success / test_y.shape[1]) * 100, np.sum(train_result['costs'][-1])))


LearningAlgorithm.plotCost(train_result['costs'])

exit()
