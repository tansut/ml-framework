from aibrite.ml.neuralnet import NeuralNet
import numpy as np


class NeuralNetWithMomentum(NeuralNet):

    def initialize_layers(self, hiddens):
        super().initialize_layers(hiddens)
        for i, layer in enumerate(self.hidden_layers + [self.output_layer]):
            layer.VdW = np.zeros(layer.W.shape)
            layer.Vdb = np.zeros(layer.b.shape)

    def _backward_for_layer(self, layer, Y, epoch, current_batch_index, total_batch_index):
        super()._backward_for_layer(layer, Y, epoch,
                                    current_batch_index, total_batch_index)
        layer.VdW = self.beta * layer.VdW + \
            (1.0 - self.beta) * layer.dW
        layer.Vdb = self.beta * layer.Vdb + \
            (1.0 - self.beta) * layer.db

    def _grad_layer(self, layer, Y, epoch, current_batch_index, total_batch_index):
        lr = self.learning_rate / (1 + self.learning_rate_decay * epoch)

        layer.W = layer.W - lr * layer.VdW
        layer.b = layer.b - lr * layer.Vdb

    def __init__(self, train_x, train_y, beta=0.9, *args, **kwargs):
        super().__init__(train_x, train_y, *args, **kwargs)
        self.beta = beta
