from aibrite.ml.neuralnet import NeuralNet
import numpy as np


class NeuralNetWithAdam(NeuralNet):

    def initialize_layers(self, hiddens):
        super().initialize_layers(hiddens)
        for i, layer in enumerate(self.hidden_layers + [self.output_layer]):
            layer.VdW = np.zeros(layer.W.shape)
            layer.Vdb = np.zeros(layer.b.shape)
            layer.SdW = np.zeros(layer.W.shape)
            layer.Sdb = np.zeros(layer.b.shape)

    def _backward_for_layer(self, layer, Y, epoch, current_batch_index, total_batch_index):
        super()._backward_for_layer(layer, Y, epoch,
                                    current_batch_index, total_batch_index)
        layer.VdW = self.beta1 * layer.VdW + \
            (1.0 - self.beta1) * layer.dW
        layer.Vdb = self.beta1 * layer.Vdb + \
            (1.0 - self.beta1) * layer.db
        layer.SdW = self.beta2 * layer.SdW + \
            (1.0 - self.beta2) * np.square(layer.dW)
        layer.Sdb = self.beta2 * layer.Sdb + \
            (1.0 - self.beta2) * np.square(layer.db)

        t = total_batch_index
        layer.VdWCorrected = layer.VdW / (1 - self.beta1)**t
        layer.VdbCorrected = layer.Vdb / (1 - self.beta1)**t

        layer.SdWCorrected = layer.SdW / (1 - self.beta2)**t
        layer.SdbCorrected = layer.Sdb / (1 - self.beta2)**t

    def _grad_layer(self, layer, Y):
        layer.W = layer.W - self.learning_rate * \
            (layer.VdWCorrected /
             (np.sqrt(layer.SdWCorrected) + self.epsilon))

        layer.b = layer.b - self.learning_rate * \
            (layer.VdbCorrected /
             (np.sqrt(layer.SdbCorrected) + self.epsilon))

    def __init__(self, train_x, train_y, beta1=0.9, beta2=0.999, epsilon=0.00000001, *args, **kwargs):
        super().__init__(train_x, train_y, *args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
