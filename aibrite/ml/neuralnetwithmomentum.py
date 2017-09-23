from aibrite.ml.neuralnet import NeuralNet
import numpy as np


class NeuralNetWithMomentum(NeuralNet):
    def _init_layer(self, layer_num, layer):
        if (layer_num > 0):
            layer['VdW'] = np.zeros(layer['W'].shape)
            layer['Vdb'] = np.zeros(layer['b'].shape)

        super()._init_layer(layer_num, layer)

    def _backward_for_layer(self, layer_num, layer):
        super()._backward_for_layer(layer_num, layer)
        if (layer_num > 0):
            layer['VdW'] = self.beta * layer['VdW'] + \
                (1.0 - self.beta) * layer['dW']
            layer['Vdb'] = self.beta * layer['Vdb'] + \
                (1.0 - self.beta) * layer['db']

    def _grad_layer(self, layer_num, layer):
        if (layer_num > 0):
            layer['W'] = layer['W'] - self.learning_rate * layer['VdW']
            layer['b'] = layer['b'] - self.learning_rate * layer['Vdb']

    def __init__(self, train_x, train_y, beta=0.9, *args, **kwargs):
        super().__init__(train_x, train_y, *args, **kwargs)
        self.beta = beta
