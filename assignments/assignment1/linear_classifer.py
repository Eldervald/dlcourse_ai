import numpy as np


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


def softmax_with_cross_entropy(predictions, target_index):
    '''
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
    '''

    single = (predictions.ndim == 1)
    
    logits = predictions.copy()

    if single:
        logits = predictions.reshape(1, -1)
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

    loss = reg_strength * np.trace(np.matmul(W.T, W))   # L2(W) = λ * tr(W.T * W)
    grad = 2 * reg_strength * W                         # dL2(W)/dW = 2 * λ * W

    return loss, grad   # L2(W), dL2(W)/dW


def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)
    loss, dprediction = softmax_with_cross_entropy(predictions, target_index)

    dW = np.matmul(dprediction.T, X).T
    return loss, dW


class LinearSoftmaxClassifier:
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5, epochs=1):
        '''
        Trains linear classifier

        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y) + 1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            loss = np.nan
            for batch_indices in batches_indices:
                batch_X = X[batch_indices]
                batch_y = y[batch_indices]

                fn_loss, fn_dW = linear_softmax(batch_X, self.W, batch_y)
                reg_loss, reg_dW = l2_regularization(self.W, reg)

                loss = fn_loss + reg_loss
                dW = fn_dW + reg_dW

                self.W = self.W - learning_rate * dW

            loss_history.append(loss)

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''

        Z = np.dot(X, self.W)
        S = softmax(Z)

        return np.argmax(S, axis=1)
