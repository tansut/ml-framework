from ml import LearningAlgorithm
import numpy as np


class DeepNN(LearningAlgorithm):

    def _compute_cost(self):
        A = self._last_layer['A']
        Y = self._yvalues_binary
        m = self.m
        logprobs = np.multiply(np.log(A), Y)
        losses = -np.sum(logprobs, axis=0)
        cost = (1. / m) * np.sum(losses)
        return cost

    def _forward(self, _layers):
        for i, layer in enumerate(_layers):
            if (i == 0):
                continue
            prev_layer = _layers[i - 1]
            layer["Z"] = layer["W"].dot(prev_layer["A"]) + layer["b"]
            layer["A"] = layer["G"](layer["Z"])

    def _backward_for_layer(self, layer_num, layer):
        m = self.m

        if layer_num == len(self._layers) - 1:
            layer['dW'] = (1. / m) * \
                layer['dZ'].dot(self._layers[layer_num - 1]['A'].T)
            layer['db'] = (1. / m) * np.sum(layer['dZ'],
                                            axis=1, keepdims=True)
        elif (layer_num != 0):
            layer['dZ'] = self._layers[layer_num +
                                       1]['W'].T.dot(self._layers[layer_num + 1]['dZ']) * layer['G_d'](layer['A'])
            layer['dW'] = (1. / m) * \
                layer['dZ'].dot(self._layers[layer_num - 1]['A'].T)
            layer['db'] = (1. / m) * np.sum(layer['dZ'],
                                            axis=1, keepdims=True)

    def _backward(self):
        m = self.m
        self._last_layer['dZ'] = self._last_layer['A'] - self._yvalues_binary
        for i, v in reversed(list(enumerate(self._layers))):
            self._backward_for_layer(i, v)

    def _grad_layer(self, layer_num, layer):
        if (layer_num > 0):
            layer['W'] = layer['W'] - self.learning_rate * layer['dW']
            layer['b'] = layer['b'] - self.learning_rate * layer['db']

    def _grads(self):
        for i, layer in enumerate(self._layers):
            self._grad_layer(i, layer)

    def _init_layer(self, layer_num, layer):
        self._layers.insert(layer_num, layer)

    def _generate_layers(self, _layers):
        self._layers = []
        np.random.seed(1)
        for i, v in enumerate(_layers):
            if (i == 0):
                self._init_layer(0, {
                    "n": _layers[0],
                    "A": self.train_x_orig
                })
            else:
                rand_fac = np.sqrt(2. / _layers[i - 1])
                self._init_layer(i, {
                    "n": _layers[i],
                    "W": np.random.randn(_layers[i], _layers[i - 1]) * rand_fac,
                    "b": np.zeros((_layers[i], 1)),
                    "G": LearningAlgorithm.relu if i < len(_layers) - 1 else LearningAlgorithm.softmax,
                    "G_d": LearningAlgorithm.relu_d if i < len(_layers) - 1 else LearningAlgorithm.softmax_d
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
        self._generate_layers(_layers)
        cost_history = []
        for i, layer in enumerate(range(self.iteration_count)):
            self._forward(self._layers)
            cost = self._compute_cost()
            if (i % 1000 == 0):
                pass
                print("It: {0}, Cost: {1}".format(i, cost))
            cost_history.append(cost)
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
