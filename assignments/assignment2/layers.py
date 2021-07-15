import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    
    if W.ndim == 1:
        loss = np.dot(W, W)
    else:
        loss = np.trace(np.matmul(W.T, W))
    
    loss *= reg_strength
    
    grad = 2 * reg_strength * W     # dL2(W)/dW = 2 * λ * W

    return loss, grad   # L2(W), dL2(W)/dW

def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''

    single = (predictions.ndim == 1)
    
    probs = predictions.copy()
    
    if single:
        probs = probs.reshape(1, -1)

    probs -= np.max(probs, axis=1).reshape(-1, 1)

    probs = np.exp(probs)
    sums = np.sum(probs, axis=1).reshape(-1, 1)
    result = probs / sums

    if single:
        result = result.reshape(-1)

    return result


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''

    single = (probs.ndim == 1)

    if single:
        probs = probs.reshape(1, -1)
        target_index = np.array([target_index])

    return np.mean(-np.log(probs[np.arange(target_index.shape[0]), target_index]))


def softmax_with_cross_entropy(preds, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    single = (preds.ndim == 1)
    
    logits = preds.copy()

    if single:
        logits = preds.reshape(1, -1)
        target_index = np.array([target_index])

    probs = softmax(logits)
    loss = cross_entropy_loss(probs, target_index)
    
#     print(f'probs : \n{probs}')
#     print(f'loss : \n{loss}')
#     print(f'target : \n{target_index}')

    grad = np.zeros(probs.shape)
    grad[np.arange(probs.shape[0]), target_index] = 1
    dprediction = (probs - grad) / len(probs)

    if single:
        dprediction = dprediction.reshape(-1)
#     print(f'grad : \n{grad}')
#     print(f'pred : \n{dprediction}')

    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)
    
    def reset_grad(self):
        self.grad = np.zeros_like(self.value)


class ReLULayer:
    def __init__(self):
        self.grad = None

    def forward(self, X):
        self.grad = (X > 0).astype(np.float64)
        res = X.copy()
        res[res < 0] = 0
        return res

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        d_result = d_out * self.grad
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        return np.matmul(self.X, self.W.value) + self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        
        d_res = np.matmul(d_out, self.W.value.T)
        d_W = np.matmul(self.X.T, d_out)
        d_B = np.sum(d_out, axis=0)
        
        self.W.grad = d_W
        self.B.grad = d_B

        return d_res

    def params(self):
        return {'W': self.W, 'B': self.B}
