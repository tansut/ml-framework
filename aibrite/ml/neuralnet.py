from aibrite.ml.core import MlBase, PredictionResult, TrainIteration, TrainResult, NeuralNetLayer, InputLayer, OutputLayer, HiddenLayer

import numpy as np
import time
import datetime
import uuid


class NeuralNet(MlBase):

    def __repr__(self):
        return "NeuralNet[it={iteration_count},lr={learning_rate:6.4f},hl={hidden_layers},lrd={learning_rate_decay:6.4f},lambd={lambd:6.4f},batch={minibatch_size},epochs={epochs}, shuffle={shuffle}]".format(iteration_count=self.iteration_count,
                                                                                                                                                                                                               learning_rate=self.learning_rate,
                                                                                                                                                                                                               hidden_layers=self.hidden_layers,
                                                                                                                                                                                                               learning_rate_decay=self.learning_rate_decay,
                                                                                                                                                                                                               lambd=self.lambd,
                                                                                                                                                                                                               minibatch_size=self.minibatch_size,
                                                                                                                                                                                                               epochs=self.epochs,
                                                                                                                                                                                                               shuffle=self.shuffle)

    def l2_regularization_cost(self, m):
        regulariozation = 0.0
        for i, v in enumerate(self._hidden_layers + [self.output_layer]):
            ws = np.sum(np.square(v.W))
            regulariozation += ws
        regulariozation = (
            1. / m) * regulariozation
        return regulariozation

    def compute_cost(self, Y):
        """Computes Softmax cost"""
        A = self.output_layer.A
        m = Y.shape[1]
        try:
            logprobs = np.multiply(np.log(A), Y)
        except Exception as exc:
            print(exc, A)
        losses = -np.sum(logprobs, axis=0)
        cost = (1. / m) * np.sum(losses)
        return cost + (self.lambd / 2.) * self.l2_regularization_cost(m)

    def _forward(self, _layers):
        for i, layer in enumerate(_layers):
            if (i == 0):
                continue
            layer.Z = layer.W.dot(layer.prev_layer.A) + layer.b
            layer.A = layer.activation_fn(layer.Z)

    def _backward_for_layer(self, layer, Y, iteration_data):
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

    def _backward(self, Y, iteration_data):
        self.output_layer.dZ = self.output_layer.A - Y
        for i, v in reversed(list(enumerate(self._hidden_layers + [self.output_layer]))):
            self._backward_for_layer(
                v, Y, iteration_data)

    def _grad_layer(self, layer, Y, iteration_data):
        # self.learning_rate / (1 + self.learning_rate_decay * epoch)
        lr = iteration_data.calculated_learning_rate

        layer.W = layer.W - lr * layer.dW
        layer.b = layer.b - lr * layer.db

    def _grads(self, Y, iteration_data):
        for i, layer in enumerate(self._hidden_layers + [self.output_layer]):
            self._grad_layer(
                layer, Y, iteration_data)

    def initialize_layers(self):
        hiddens = self.hidden_layers
        self.layers = []
        self._hidden_layers = []

        input_layer = InputLayer(self.n)
        input_layer.A = self.train_x

        self.layers.append(input_layer)

        np.random.seed(1)

        prev_layer = input_layer

        for i, v in enumerate(hiddens):
            layer = HiddenLayer(
                v, MlBase.relu, MlBase.relu_d, prev_layer=prev_layer)
            layer.init_weight_bias()
            self.layers.append(layer)
            self._hidden_layers.append(layer)
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

        self.train_x = np.asarray(train_x).T
        self.train_y = np.asarray(train_y).reshape(len(train_y), 1).T

        self.m = self.train_x.shape[1]
        self.n = self.train_x.shape[0]

        self.train_x_orig = train_x
        self.train_y_orig = train_y

        if labels == None:
            labels = np.unique(train_y)

        self.label_value_map = {v: i for i, v in enumerate(labels)}
        self.label_binary_matrix = np.eye(len(labels))

        self._yvalues_binary = self.y_to_binary(self.train_y)

        self.labels = labels

    def label_to_binary(self, label):
        index = self.label_value_map[label]
        return self.label_binary_matrix[index][:]

    def y_to_binary(self, y):
        return np.array([self.label_to_binary(
            v) for v in y[0, :]]).T

    def get_hyperparameters(self):
        return {
            'learning_rate': self.learning_rate,
            'hidden_layers': self.hidden_layers,
            'iteration_count': self.iteration_count,
            'learning_rate_decay': self.learning_rate_decay,
            'lambd': self.lambd,
            'minibatch_size': self.minibatch_size,
            'shuffle': self.shuffle
        }

    def __init__(self, train_x, train_y,
                 hidden_layers=None,
                 learning_rate=0.01,
                 learning_rate_decay=0,
                 iteration_count=2000,
                 lambd=0.0001,
                 minibatch_size=0,
                 epochs=1,
                 labels=None,
                 shuffle=False):
        self.instance_id = str(uuid.uuid4())
        self.learning_rate = learning_rate
        self.iteration_count = iteration_count
        self.hidden_layers = hidden_layers if (
            hidden_layers != None) else (4,)
        self.lambd = lambd
        self.minibatch_size = minibatch_size
        self.epochs = epochs
        self.shuffle = shuffle
        self.learning_rate_decay = learning_rate_decay
        self.prepare_data(train_x, train_y, labels)
        self.initialize_layers()

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

    def train(self, call_back=None):
        self.train_result = TrainResult()
        minibatch_size = self.minibatch_size
        if minibatch_size <= 0:
            minibatch_size = self.train_x.shape[1]
        total_batch_index = 0
        total_iteration_count = 0
        for epoch in range(self.epochs):
            current_batch_index = 0

            for batch in self.iterate_minibatches(self.train_x, self.train_y, minibatch_size, shuffle=self.shuffle):
                x_batch, y_batch = batch
                y_values_binary = self.y_to_binary(y_batch)
                self.input_layer.A = x_batch
                for i, v in enumerate(range(self.iteration_count)):
                    self._forward(self.layers)
                    cost = self.compute_cost(y_values_binary)
                    calculated_learning_rate = self.learning_rate / \
                        (1 + self.learning_rate_decay * epoch)
                    iteration_data = TrainIteration(cost=cost, epoch=epoch, current_batch_index=current_batch_index,
                                                    total_batch_index=total_batch_index, total_iteration_count=total_iteration_count, current_batch_iteration=i, calculated_learning_rate=calculated_learning_rate)
                    call_back(
                        self, iteration_data) if call_back != None else None
                    self._backward(y_values_binary, iteration_data)
                    self._grads(y_values_binary, iteration_data)
                    total_iteration_count += 1
                current_batch_index += 1
                total_batch_index += 1
        return self.train_result.complete()

    def predict(self, X, expected=None):
        self.prediction_result = PredictionResult()
        self.input_layer.pA = np.asarray(X).T

        for i, layer in enumerate(self._hidden_layers + [self.output_layer]):
            layer.pZ = layer.W.dot(layer.prev_layer.pA) + layer.b
            layer.pA = layer.activation_fn(layer.pZ)

        A = self.output_layer.pA
        maxindexes = np.argmax(A, axis=0)
        pred = [self.labels[maxindexes[i]]
                for i, v in enumerate(maxindexes)]
        if expected is None:
            score = None
        else:
            score = NeuralNet.score_report(expected, pred, labels=self.labels)
        return self.prediction_result.complete(pred, A, score)
