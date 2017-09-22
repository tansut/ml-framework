from ml import LearningAlgorithm
import numpy as np


class DeepNN(LearningAlgorithm):

    def _compute_cost(self):
        """Computes Softmax cost"""
        A = self._layers[-1]['A']
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

        # skip first (input) layer
        if layer_num > 0:

            # compute dZ if this is not output layer
            if layer_num < len(self._layers) - 1:
                next_layer = self._layers[layer_num + 1]
                g_function_derivative = layer['G_d']
                layer['dZ'] = next_layer['W'].T.dot(
                    next_layer['dZ']) * g_function_derivative(layer['A'])

            prev_layer = self._layers[layer_num - 1]
            layer['dW'] = (1. / m) * layer['dZ'].dot(prev_layer['A'].T)
            layer['db'] = (1. / m) * np.sum(layer['dZ'],
                                            axis=1, keepdims=True)

    def _backward(self):
        self._layers[-1]['dZ'] = self._layers[-1]['A'] - self._yvalues_binary
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

    def train(self, train_cb=None):
        _layers = np.concatenate(
            ([self.train_x_orig.shape[0]], self.hidden_layers, [len(self._yvalues_dict)]))
        self._generate_layers(_layers)
        cost_history = []
        for i, layer in enumerate(range(self.iteration_count)):
            self._forward(self._layers)
            cost = self._compute_cost()
            train_cb(i, cost) if train_cb != None else None
            if (i % 1000 == 0):
                pass
                #print("It: {0}, Cost: {1}".format(i, cost))
            cost_history.append(cost)
            self._backward()
            self._grads()

        return {
            'costs': cost_history,
            'layers': self._layers
        }

    def predict_and_test(self, test_x, test_y):
        prediction_result = self.predict(test_x)
        successes = prediction_result['predictions'] == test_y
        total_success = np.sum(successes)
        return {
            'result': successes,
            'total_success': total_success,
            'rate': (total_success / successes.shape[1]) * 100
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
