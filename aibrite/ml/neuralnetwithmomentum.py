from aibrite.ml.neuralnet import NeuralNet
import numpy as np


class NeuralNetWithMomentum(NeuralNet):

    def __repr__(self):
        return ("NeuralNetWithMomentum[it={iteration_count},lr={learning_rate:6.4f},hl={hidden_layers},lrd={learning_rate_decay:6.4f},lambd={lambd:6.4f},batch={minibatch_size},epochs={epochs},shuffle={shuffle},beta={beta:6.4f}]").format(iteration_count=self.iteration_count,
                                                                                                                                                                                                                                             learning_rate=self.learning_rate,
                                                                                                                                                                                                                                             hidden_layers=self.hidden_layers,
                                                                                                                                                                                                                                             learning_rate_decay=self.learning_rate_decay,
                                                                                                                                                                                                                                             lambd=self.lambd,
                                                                                                                                                                                                                                             minibatch_size=self.minibatch_size,
                                                                                                                                                                                                                                             epochs=self.epochs,
                                                                                                                                                                                                                                             shuffle=self.shuffle,
                                                                                                                                                                                                                                             beta=self.beta)

    def initialize_layers(self):
        super().initialize_layers()
        for i, layer in enumerate(self._hidden_layers + [self.output_layer]):
            layer.VdW = np.zeros(layer.W.shape)
            layer.Vdb = np.zeros(layer.b.shape)

    def _backward_for_layer(self, layer, Y, iteration_data):
        super()._backward_for_layer(layer, Y, iteration_data)
        layer.VdW = self.beta * layer.VdW + \
            (1.0 - self.beta) * layer.dW
        layer.Vdb = self.beta * layer.Vdb + \
            (1.0 - self.beta) * layer.db

    def _grad_layer(self, layer, Y, iteration_data):
        lr = iteration_data.calculated_learning_rate

        layer.W = layer.W - lr * layer.VdW
        layer.b = layer.b - lr * layer.Vdb

    def get_hyperparameters(self):
        hp = super().get_hyperparameters()
        hp['beta'] = self.beta

        return hp

    def __init__(self, train_x, train_y, beta=0.9, *args, **kwargs):
        super().__init__(train_x, train_y, *args, **kwargs)
        self.beta = beta
