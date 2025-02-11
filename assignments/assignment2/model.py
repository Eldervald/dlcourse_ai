import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, softmax, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.layers = [
            FullyConnectedLayer(n_input, hidden_layer_size),
            ReLULayer(),
            FullyConnectedLayer(hidden_layer_size, n_output)
        ]

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
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
        
        for _, param in self.params().items():
            reg_loss, reg_grad = l2_regularization(param.value, self.reg)
            loss += reg_loss
            param.grad += reg_grad
            
        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        
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
