from ml import LearningAlgorithm
import numpy as np


class NeuralNetwork(LearningAlgorithm):

    def _compute_cost(self):
        A = self._last_layer['A']
        Y = self._yvalues_binary
        m = self.m
        logprobs = np.multiply(np.log(A), Y) + \
            np.multiply(np.log(1 - A), (1 - Y))
        cost = - (1. / m) * np.sum(logprobs, axis=1).reshape(Y.shape[0], 1)
        return cost

    def _forward(self, _layers):
        for i, layer in enumerate(_layers):
            if (i == 0):
                continue
            prev_layer = _layers[i - 1]
            layer["Z"] = layer["W"].dot(prev_layer["A"]) + layer["b"]
            layer["A"] = layer["G"](layer["Z"])

    def _backward(self):
        self._last_layer['dZ'] = self._last_layer['A'] - self._yvalues_binary
        m = self.m
        for i, v in reversed(list(enumerate(self._layers))):
            if i == len(self._layers) - 1:
                v['dW'] = (1. / m) * v['dZ'].dot(self._layers[i - 1]['A'].T)
                v['db'] = (1. / m) * np.sum(v['dZ'], axis=1, keepdims=True)
                continue

            if (i == 0):
                continue

            v['dZ'] = self._layers[i +
                                   1]['W'].T.dot(self._layers[i + 1]['dZ']) * v['G_d'](v['A'])
            v['dW'] = (1. / m) * v['dZ'].dot(self._layers[i - 1]['A'].T)
            v['db'] = (1. / m) * np.sum(v['dZ'], axis=1, keepdims=True)

    def _grads(self):
        for i, layer in enumerate(self._layers):
            if (i == 0):
                continue
        layer['W'] = layer['W'] - self.learning_rate * layer['dW']
        layer['b'] = layer['b'] - self.learning_rate * layer['db']

    def _init_layers(self, _layers):
        self._layers = []
        np.random.seed(1)
        for i, v in enumerate(_layers):
            if (i == 0):
                self._layers.insert(0, {
                    "n": _layers[0],
                    "A": self.train_x_orig
                })
            else:
                rand_fac = 0.01  # np.sqrt(2. / _layers[i - 1])
                self._layers.append({
                    "n": _layers[i],
                    "W": np.random.randn(_layers[i], _layers[i - 1]) * rand_fac,
                    "b": np.zeros((_layers[i], 1)),
                    "G": LearningAlgorithm.relu if i < len(_layers) - 1 else LearningAlgorithm.sigmoid,
                    "G_d": LearningAlgorithm.relu_d if i < len(_layers) - 1 else LearningAlgorithm.sigmoid_d
                })
        self._first_layer = self._layers[0]
        self._last_layer = self._layers[-1]

    def _init(self):
        unique_y = np.unique(self.train_y_orig)
        self._yvalues_dict = {v: i for i, v in enumerate(unique_y)}
        self._yindices_dict = {i: v for i, v in enumerate(unique_y)}
        _yvalues = np.eye(len(self._yvalues_dict))
        self.m = self.train_x_orig.shape[1]
        self._yvalues_binary = np.array([_yvalues[self._yvalues_dict[v]]
                                         for v in self.train_y_orig[0, :]]).T

    def __init__(self, train_x, train_y, hidden_layers=None, learning_rate=0.01, iteration_count=1000):
        self.train_x_orig = train_x
        self.train_y_orig = train_y
        self.learning_rate = learning_rate
        self.iteration_count = iteration_count
        self.hidden_layers = hidden_layers[:] if (
            hidden_layers != None) else [4]
        self._init()

    def train(self):
        _layers = np.concatenate(
            ([self.train_x_orig.shape[0]], self.hidden_layers, [len(self._yvalues_dict)]))
        self._init_layers(_layers)
        cost_history = []
        for i, layer in enumerate(range(self.iteration_count)):
            self._forward(self._layers)
            costs = self._compute_cost()
            if (i % 1000 == 0):
                pass
                # print("Iteration: {0}, Costs: {1}".format(i, costs))
            cost_history.append(costs)
            self._backward()
            self._grads()

        return {
            'costs': cost_history,
            'layers': self._layers
        }

    def predict(self, X):
        prediction_layers = [{'A': X}]

        for i, layer in enumerate(self._layers):
            if (i == 0):
                continue
            prediction_layers.append({
                'W': layer["W"],
                'b': layer["b"],
                'G': layer['G']
            })

        self._forward(prediction_layers)
        A = prediction_layers[-1]['A']
        maxindexes = np.argmax(A, axis=0)
        prediction_values = np.array(
            [self._yindices_dict[maxindexes[i]] for i, v in enumerate(maxindexes)]).reshape(1, A.shape[1])
        return {
            'probs': A,
            'predictions': prediction_values
        }
