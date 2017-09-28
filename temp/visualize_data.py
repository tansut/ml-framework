import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def mean_norm(np_arr):
    avgs = np.sum(np_arr, axis=0, keepdims=True) / np_arr.shape[0]
    return np_arr - avgs


def var_norm(np_arr):
    varss = np.sum(np.square(np_arr), axis=0,
                   keepdims=True) / np_arr.shape[0]
    return np_arr / varss


def zscore(np_arr):
    avgs = np.mean(np_arr, axis=0, keepdims=True)
    return (np_arr - avgs) / np.std(np_arr, axis=0, keepdims=True)


data_frame = pd.read_csv('./data/winequality-red.csv', sep=';')
# data_frame.drop('density', axis=1, inplace=True)
#del data_frame['density']


data = data_frame.values

n = data.shape[1] - 1
m = data.shape[0]

ox = data[:, 0:n]
y = data[:, n]

x = zscore(ox)
x = ox
# x = var_norm(x)


plt.figure(1)


for i, c in enumerate(data_frame.columns[0:n]):
    plt.subplot(4, 3, i + 1)
    plt.scatter(x[:, i], y)
    plt.xlabel(c)
    plt.ylabel("quality")

# plt.figure(2)


# for i, c in enumerate(data_frame.columns[0:n]):
#     plt.subplot(4, 3, i + 1)
#     plt.scatter(x[:, i], y)
#     plt.xlabel(c)
#     plt.ylabel("quality")

plt.show()
