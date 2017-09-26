from aibrite.ml.neuralnet import NeuralNet
import numpy as np


class NeuralNetWithRMSprop(NeuralNet):

    def __repr__(self):
        return ("NeuralNetWithRMSprop[it={iteration_count},lr={learning_rate:6.4f},lrd={learning_rate_decay:6.4f},lambd={lambd:6.4f},batch={minibatch_size},epochs={epochs},shuffle={shuffle},beta={beta:6.4f},epslion={epsilon:6.4f}]").format(iteration_count=self.iteration_count,
                                                                                                                                                                                                                                                learning_rate=self.learning_rate,
                                                                                                                                                                                                                                                learning_rate_decay=self.learning_rate_decay,
                                                                                                                                                                                                                                                lambd=self.lambd,
                                                                                                                                                                                                                                                minibatch_size=self.minibatch_size,
                                                                                                                                                                                                                                                epochs=self.epochs,
                                                                                                                                                                                                                                                shuffle=self.shuffle,
                                                                                                                                                                                                                                                beta=self.beta,
                                                                                                                                                                                                                                                epsilon=self.epsilon)

    def initialize_layers(self, hiddens):
        super().initialize_layers(hiddens)
        for i, layer in enumerate(self.hidden_layers + [self.output_layer]):
            layer.SdW = np.zeros(layer.W.shape)
            layer.Sdb = np.zeros(layer.b.shape)

    def _backward_for_layer(self, layer, Y, epoch, current_batch_index, total_batch_index):
        super()._backward_for_layer(layer, Y, epoch,
                                    current_batch_index, total_batch_index)
        layer.SdW = self.beta * layer.SdW + \
            (1.0 - self.beta) * np.square(layer.dW)
        layer.Sdb = self.beta * layer.Sdb + \
            (1.0 - self.beta) * np.square(layer.db)

    def _grad_layer(self, layer, Y, epoch, current_batch_index, total_batch_index):
        lr = self.learning_rate / (1 + self.learning_rate_decay * epoch)

        layer.W = layer.W - lr * \
            (layer.dW / (np.sqrt(layer.SdW) + self.epsilon))
        layer.b = layer.b - lr * \
            (layer.db / (np.sqrt(layer.Sdb) + self.epsilon))

    def __init__(self, train_x, train_y, beta=0.9, epsilon=0.00000001, *args, **kwargs):
        super().__init__(train_x, train_y, *args, **kwargs)
        self.beta = beta
        self.epsilon = epsilon
