import numpy as np


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    
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
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
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
        d_result = d_out * self.grad
        return d_result

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = Param(X)
        return np.matmul(self.X.value, self.W.value) + self.B.value

    def backward(self, d_out):
        d_res = np.matmul(d_out, self.W.value.T)
        d_W = np.matmul(self.X.value.T, d_out)
        d_B = np.sum(d_out, axis=0)
        
        self.W.grad = d_W
        self.B.grad = d_B

        return d_res

    def params(self):
        return { 'W': self.W, 'B': self.B }

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding


    def forward(self, X):
        batch_size, height, width, channels = X.shape

        out_height = height - self.filter_size + 1 + 2 * self.padding
        out_width = width - self.filter_size + 1 + 2 * self.padding
        
        self.X = Param(np.pad(X.copy(), ((0,0), (self.padding,self.padding), (self.padding,self.padding), (0,0)), 'constant'))
        
        out = np.zeros((batch_size, out_height, out_width, self.out_channels))
        W_flat = self.W.value.reshape(-1, self.out_channels)
        
        for y in range(out_height):
            for x in range(out_width):
                X_slice_flat = self.X.value[:, y:y + self.filter_size, x:x + self.filter_size, :].reshape(batch_size, -1)
                out[:, y, x, :] = np.matmul(X_slice_flat, W_flat) + self.B.value
                
        return out


    def backward(self, d_out):
        batch_size, height, width, channels = self.X.value.shape
        _, out_height, out_width, out_channels = d_out.shape
        
        d_input = np.zeros_like(self.X.value)
        W_flat = self.W.value.reshape(-1, self.out_channels)

        for y in range(out_height):
            for x in range(out_width):
                X_slice_flat = self.X.value[:, y:y + self.filter_size, x:x + self.filter_size, :].reshape(batch_size, -1)
                d_input[:, y:y + self.filter_size, x:x + self.filter_size, :] += np.dot(d_out[:, y, x, :], W_flat.T).reshape(batch_size, self.filter_size, self.filter_size, channels)
                
                self.W.grad += np.matmul(X_slice_flat.T, d_out[:, y, x, :]).reshape(self.filter_size, self.filter_size, self.in_channels, out_channels)
                self.B.grad += np.sum(d_out[:, y, x, :], axis=0)
                
        return d_input[:, self.padding:height - self.padding, self.padding:width - self.padding, :]
                

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X = X.copy()
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1
        
        out = np.zeros([batch_size, out_height, out_width, channels])
        
        for y in range(out_height):
            for x in range(out_width):
                out[:, y, x, :] = np.amax(self.X[:,y*self.stride:y*self.stride + self.pool_size,x*self.stride:x*self.stride + self.pool_size, :], axis=(1, 2))
        
        return out

    def backward(self, d_out):
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, _ = d_out.shape
        
        d_res = np.zeros_like(self.X)
        
        for y in range(out_height):
            for x in range(out_width):
                stride_X = self.X[:,
                                  y*self.stride:y*self.stride + self.pool_size,
                                  x*self.stride:x*self.stride + self.pool_size,
                                  :].reshape(batch_size, -1, channels)
                
                max_idxs = np.argmax(stride_X, axis= 1)
                
                stride_d_res = d_res[:,
                                     y * self.stride:y * self.stride + self.pool_size,
                                     x * self.stride:x * self.stride + self.pool_size,
                                     :].reshape(batch_size, -1, channels)
                
                for batch in range(batch_size):
                    for channel in range(channels):
                        stride_d_res[batch, max_idxs[batch, channel], channel] += d_out[batch, y, x, channel]
                
                d_res[:,y * self.stride:y * self.stride + self.pool_size,x * self.stride:x * self.stride + self.pool_size,:] = stride_d_res.reshape(batch_size, self.pool_size, self.pool_size, channels)     
                  
        return d_res   

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X_shape = batch_size, height, width, channels
        return X.reshape(batch_size, -1)
        
    def backward(self, d_out):
        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}
