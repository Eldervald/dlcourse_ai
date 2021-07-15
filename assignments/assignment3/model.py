import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization,
    softmax
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        width, height, n_channels = input_shape
        conv = 3
        pad = 1
        stride = 4
        pool = 4
        
        hidden_layer_size = (width // stride // stride) * (height // stride // stride) * conv2_channels
        
        self.layers = [
            ConvolutionalLayer(n_channels, conv1_channels, conv, pad),
            ReLULayer(),
            MaxPoolingLayer(pool, stride),
            ConvolutionalLayer(conv1_channels, conv2_channels, conv, pad),
            ReLULayer(),
            MaxPoolingLayer(pool, stride),
            Flattener(),
            FullyConnectedLayer(hidden_layer_size, n_output_classes)        
        ]

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        for _, param in self.params().items():
            param.reset_grad()
        
        out = X.copy()
        for layer in self.layers:
            out = layer.forward(out)
        
        loss, d_loss = softmax_with_cross_entropy(out, y)
        
        for layer in reversed(self.layers):
            d_loss = layer.backward(d_loss)
            
        return loss

    def predict(self, X):
        out = X.copy()
        for layer in self.layers:
            out = layer.forward(out)
        
        out = softmax(out)
        pred = np.argmax(out, axis=1)
        
        return pred

    def params(self):
        result = {}

        for i, layer in enumerate(self.layers):
            for name, par in layer.params().items():
                result[f'{i}_{name}'] = par

        return result
