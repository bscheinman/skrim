import numpy as np

pad_ones = lambda x: np.append(np.ones([x.shape[0], 1]), x, 1)

sigmoid = lambda x: 1 / (1 + exp(-x))
sigmoid_curry = np.vectorize(sigmoid)

def sigmoid_gradient(x):
    x_sig = sigmoid(x)
    return x * (1 - x)
sigmoid_gradient_curry = np.vectorize(sigmoid_gradient)