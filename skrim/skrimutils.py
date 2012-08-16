import numpy as np
from math import exp, sqrt

pad_ones = lambda x: np.append(np.ones([x.shape[0], 1]), x, 1)

sigmoid = lambda x: 1 / (1 + exp(-x))
sigmoid_curry = np.vectorize(sigmoid)


def sigmoid_gradient(x):
    x_sig = sigmoid(x)
    return x_sig * (1 - x_sig)
sigmoid_gradient_curry = np.vectorize(sigmoid_gradient)

euclidean_distance = lambda x1, x2: sqrt(sum((x1 - x2) ** 2))
manhattan_distance = lambda x1, x2: sum(abs(x1 - x2))
