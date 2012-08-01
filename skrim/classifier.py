"""
    This file contains classes that perform logistic regression on data sets with arbitrary features
    Data should already be cleaned and structured before they are passed here
"""

import numpy as np

sigmoid = lambda x: 1 / (1 + exp(-x))


class GradientDescent(object):

    def __init__(x, y, cost_function, alpha, max_iter = 1000, min_change = None):
        """
            x: feature array (this should NOT already include leading 1s for x_0)
            y: classification vector
            cost_function: a function to evaluate the cost of theta after each step
                it will be called as:
                cost_function(x, y, theta)
                and should return a tuple whose first value is the cost and whose second
                value is an array of gradients for each feature.
                this parameter can also be a string naming one of a number of predefined functions
            alpha: gradient descent step size
            max_iter: maximum number of iterations to perform
            min_change: if this is specified, descent will stop whenever
                the cost decrease is below this value
        """
        self.x = x
        self.y = y
        self.cost_function = cost_function
        self.alpha = alpha
        self.max_iter = max_iter
        self.min_change = min_change

        cost_history = []
        theta = np.zeros([x.shape[1]])
        for i in xrange(max_iter):
            cost, gradients = cost_function(x, y, theta)
            cost_history.append(cost)
            if cost == 0 or min_change and cost_history and cost_history[-1] - cost <= min_change:
                break
            
            theta = theta - alpha * gradients

        self.cost_history = cost_history
        self.theta = theta
