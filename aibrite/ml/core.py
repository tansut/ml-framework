import numpy as np


class MlBase:

    def zscore(np_arr):
        avgs = np.sum(np_arr, axis=0, keepdims=True) / np_arr.shape[0]
        return (np_arr - avgs) / np.std(np_arr, axis=0, keepdims=True)

    def hyperbolic_tangent(z):
        return np.tanh(z)

    def relu(data, epsilon=0.1):
        return np.maximum(epsilon * data, data)

    def relu_d(data, epsilon=0.1):
        gradients = 1. * (data > 0)
        gradients[gradients == 0] = epsilon
        return gradients

    def hyperbolic_tangent_d(z):
        return (1 - np.power(z, 2))

    def sigmoid(z):
        """sigmoid"""
        return 1. / (1 + np.exp(-z))

    def sigmoid_d(z):
        return a * (1. - a)

    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def split(arr, *ratios):
        sizes = (np.array(ratios) * len(arr))
        sizes = np.round(sizes).astype(int)
        sizes[0] = sizes[0] + len(arr) - np.sum(sizes)
        assert np.sum(sizes) == len(arr)
        res = []
        for i, v in enumerate(ratios):
            j = 0 if i == 0 else sizes[i - 1]
            res.append(arr[j:sizes[i] + j])
        return res

    def __init__(self):
        pass
