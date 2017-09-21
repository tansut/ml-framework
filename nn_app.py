from neural_network import NeuralNetwork
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


data_orig = pd.read_csv("winequality.txt", sep=";")
#data_orig = pd.read_csv("ex2data1.txt", sep=",").values

df = data_orig.copy()

# df['fixed acidity'] = np.power(df['fixed acidity'], 0.4)
# df['volatile acidity'] = np.power(df['volatile acidity'], 0.3)
# df['citric acid'] = np.power(df['citric acid'], 0.35)
# df['residual sugar'] = np.power(df['residual sugar'], 0.2)
# df['chlorides'] = np.power(df['chlorides'], 0.5)
# df['free sulfur dioxide'] = np.power(df['free sulfur dioxide'], 0.5)
# df['total sulfur dioxide'] = np.power(df['total sulfur dioxide'], -0.5)
# del df['alcohol']  # = np.power(df['alcohol'], -0.5)

split_data = NeuralNetwork.split(df.values, 0.8, 0.1, 0.1)
input_data, output_data = process_input(split_data)

train_x, train_y = input_data[0], output_data[0]
test_x, test_y = input_data[1], output_data[1]
validation_x, validation_y = input_data[2], output_data[2]

assert(train_x.shape == (split_data[0].shape[1] - 1, split_data[0].shape[0]))
assert(train_y.shape == (1, split_data[0].shape[0]))

assert(test_x.shape == (split_data[1].shape[1] - 1, split_data[1].shape[0]))
assert(test_y.shape == (1, split_data[1].shape[0]))

assert(validation_x.shape == (
    split_data[2].shape[1] - 1, split_data[2].shape[0]))
assert(validation_y.shape == (1, split_data[2].shape[0]))

analyse_data_x, analyse_data_y = test_x, test_y

iterations = [4000, 6000, 8000]
learning_rates = [0.005]
hidden_layers = [[2]]

norm_train_x = train_x  # LearningAlgorithm.zscore(train_x.T).T

for it in iterations:
    for lr in learning_rates:
        for hl in hidden_layers:
            nn = NeuralNetwork(norm_train_x, train_y, hidden_layers=hl,
                               learning_rate=lr, iteration_count=it)
            train_result = nn.train()
            prediction_result = nn.predict(analyse_data_x)
            success = np.sum(
                prediction_result['predictions'] == analyse_data_y)
            caption = 'It: {0} Lr: {1} l: {2} Succ: {3:.2f}, Cost: {4:.5f}'.format(
                it, lr, hl, (success / analyse_data_y.shape[1]) * 100, np.sum(train_result['costs'][-1]))
            print(caption)

            LearningAlgorithm.plotCosts(
                train_result['costs'], title=caption, show=False)

plt.show()

exit()
