from aibrite.ml.core import MlBase
import numpy as np


class NeuralNetLayer:
    def __init__(self, n):
        self.n = n


class InputLayer(NeuralNetLayer):
    def __init__(self, n, next_layer=None):
        super().__init__(n)
        self.A = None
        self.next_layer = next_layer


class HiddenLayer(NeuralNetLayer):
    def __init__(self, n,  activation_fn, activaion_fn_derivative, prev_layer, next_layer=None):
        super().__init__(n)
        self.W = None
        self.b = None
        self.activation_fn = activation_fn
        self.activaion_fn_derivative = activaion_fn_derivative
        self.prev_layer = prev_layer
        self.next_layer = next_layer

    def init_weight_bias(self):
        rand_fac = np.sqrt(2. / self.prev_layer.n)
        self.W = np.random.randn(self.n, self.prev_layer.n) * rand_fac
        self.b = np.zeros((self.n, 1))


class OutputLayer(HiddenLayer):
    def __init__(self, n, activation_fn, prev_layer):
        super().__init__(n, activation_fn, None, prev_layer)


class NeuralNet(MlBase):

    def l2_regularization_cost(self, m):
        regulariozation = 0.0
        for i, v in enumerate(self.hidden_layers + [self.output_layer]):
            ws = np.sum(np.square(v.W))
            regulariozation += ws
        regulariozation = (
            1. / m) * (self.lambd / 2.) * regulariozation
        return regulariozation

    def compute_cost(self, Y):
        """Computes Softmax cost"""
        A = self.output_layer.A
        m = Y.shape[1]
        logprobs = np.multiply(np.log(A), Y)
        losses = -np.sum(logprobs, axis=0)
        cost = (1. / m) * np.sum(losses)
        return cost + self.l2_regularization_cost(m)

    def _forward(self, _layers):
        for i, layer in enumerate(_layers):
            if (i == 0):
                continue
            layer.Z = layer.W.dot(layer.prev_layer.A) + layer.b
            layer.A = layer.activation_fn(layer.Z)

    def _backward_for_layer(self, layer, Y, epoch, current_batch_index, total_batch_index):
        m = Y.shape[1]

        # compute dZ if this is not output layer
        if not isinstance(layer, OutputLayer):
            layer.dZ = layer.next_layer.W.T.dot(
                layer.next_layer.dZ) * layer.activaion_fn_derivative(layer.A)

        layer.dW = (1. / m) * layer.dZ.dot(layer.prev_layer.A.T)

        # regularization
        layer.dW += (self.lambd / m) * layer.W

        layer.db = (1. / m) * np.sum(layer.dZ,
                                     axis=1, keepdims=True)

    def _backward(self, Y, epoch, current_batch_index, total_batch_index):
        self.output_layer.dZ = self.output_layer.A - Y
        for i, v in reversed(list(enumerate(self.hidden_layers + [self.output_layer]))):
            self._backward_for_layer(
                v, Y, epoch, current_batch_index, total_batch_index)

    def _grad_layer(self, layer, Y):
        layer.W = layer.W - self.learning_rate * layer.dW
        layer.b = layer.b - self.learning_rate * layer.db

    def _grads(self, Y):
        for i, layer in enumerate(self.hidden_layers + [self.output_layer]):
            self._grad_layer(layer, Y)

    def initialize_layers(self, hiddens):
        self.layers = []
        self.hidden_layers = []

        input_layer = InputLayer(self.n)
        input_layer.A = self.train_x_orig

        self.layers.append(input_layer)

        np.random.seed(1)

        prev_layer = input_layer

        for i, v in enumerate(hiddens):
            layer = HiddenLayer(
                v, MlBase.relu, MlBase.relu_d, prev_layer=prev_layer)
            layer.init_weight_bias()
            self.layers.append(layer)
            self.hidden_layers.append(layer)
            prev_layer = layer

        output_layer = OutputLayer(
            len(self.labels), MlBase.softmax, prev_layer=prev_layer)
        output_layer.init_weight_bias()

        self.layers.append(output_layer)

        for i, v in enumerate(self.layers):
            v.next_layer = self.layers[i +
                                       1] if i < len(self.layers) - 1 else None

        self.input_layer = input_layer
        self.output_layer = output_layer

    def prepare_data(self, train_x, train_y, labels):

        self.m = train_x.shape[1]
        self.n = train_x.shape[0]

        self.train_x_orig = train_x
        self.train_y_orig = train_y

        if labels == None:
            labels = np.unique(train_y)

        self.label_value_map = {v: i for i, v in enumerate(labels)}
        self.label_binary_matrix = np.eye(len(labels))

        self._yvalues_binary = self.y_to_binary(self.train_y_orig)

        self.labels = labels

    def label_to_binary(self, label):
        index = self.label_value_map[label]
        return self.label_binary_matrix[index][:]

    def y_to_binary(self, y):
        return np.array([self.label_to_binary(
            v) for v in y[0, :]]).T

    def __init__(self, train_x, train_y,
                 hidden_layers=None,
                 learning_rate=0.01,
                 iteration_count=1000,
                 lambd=0.1,
                 minibatch_size=0,
                 epochs=1,
                 labels=None,
                 shuffle=False):

        self.learning_rate = learning_rate
        self.iteration_count = iteration_count
        hiddens = hidden_layers[:] if (
            hidden_layers != None) else [4]
        self.lambd = lambd
        self.minibatch_size = minibatch_size
        self.epochs = epochs
        self.shuffle = shuffle
        self.prepare_data(train_x, train_y, labels)
        self.initialize_layers(hiddens)

    def iterate_minibatches(self, inputs, targets, batchsize, shuffle=False):
        assert inputs.shape[1] == targets.shape[1]
        if shuffle:
            indices = np.arange(inputs.shape[1])
            np.random.shuffle(indices)
        for start_idx in range(0, inputs.shape[1] - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[:, excerpt], targets[:, excerpt]

    def train(self, cb=None):
        minibatch_size = self.minibatch_size
        if minibatch_size <= 0:
            minibatch_size = self.train_x_orig.shape[1]
        total_batch_index = 0
        for epoch in range(self.epochs):
            current_batch_index = 0

            for batch in self.iterate_minibatches(self.train_x_orig, self.train_y_orig, minibatch_size, shuffle=self.shuffle):
                x_batch, y_batch = batch
                y_values_binary = self.y_to_binary(y_batch)
                self.input_layer.A = x_batch
                for i, v in enumerate(range(self.iteration_count)):
                    self._forward(self.layers)
                    cost = self.compute_cost(y_values_binary)
                    cb(cost, epoch, current_batch_index,
                       total_batch_index, i) if cb != None else None
                    self._backward(y_values_binary, epoch,
                                   current_batch_index, total_batch_index)
                    self._grads(y_values_binary)
                current_batch_index += 1
                total_batch_index += 1

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
        self.input_layer.pA = X

        for i, layer in enumerate(self.hidden_layers + [self.output_layer]):
            layer.pZ = layer.W.dot(layer.prev_layer.pA) + layer.b
            layer.pA = layer.activation_fn(layer.pZ)

        A = self.output_layer.pA
        maxindexes = np.argmax(A, axis=0)
        prediction_values = np.array(
            [self.labels[maxindexes[i]] for i, v in enumerate(maxindexes)]).reshape(1, A.shape[1])
        return {
            'probs': A,
            'predictions': prediction_values
        }
