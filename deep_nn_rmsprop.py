from ml import LearningAlgorithm
from deep_nn import DeepNN
import numpy as np


class DeepNNRMSprop(DeepNN):

    def _init_layer(self, layer_num, layer):
        if (layer_num > 0):
            layer['SdW'] = np.zeros(layer['W'].shape)
            layer['Sdb'] = np.zeros(layer['b'].shape)

        super()._init_layer(layer_num, layer)

    def _backward_for_layer(self, layer_num, layer):
        super()._backward_for_layer(layer_num, layer)
        if (layer_num > 0):
            layer['SdW'] = self.beta * layer['SdW'] + \
                (1.0 - self.beta) * np.square(layer['dW'])
            layer['Sdb'] = self.beta * layer['Sdb'] + \
                (1.0 - self.beta) * np.square(layer['db'])

    def _grad_layer(self, layer_num, layer):
        if (layer_num > 0):
            layer['W'] = layer['W'] - self.learning_rate * \
                (layer['dW'] / (np.sqrt(layer['SdW']) + self.epsilon))
            layer['b'] = layer['b'] - self.learning_rate * \
                (layer['db'] / (np.sqrt(layer['Sdb']) + self.epsilon))

    def __init__(self, train_x, train_y, beta=0.9, epsilon=0.00000001, *args, **kwargs):
        super().__init__(train_x, train_y, *args, **kwargs)
        self.beta = beta
        self.epsilon = epsilon
