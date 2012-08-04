"""
    This file contains classes that perform logistic regression on data sets with arbitrary features
    Data should already be cleaned and structured before they are passed here
"""

import numpy as np

class Classifier(object):

    def __init__(self, generator, normalizer = None):
        """
            generator: an instance of a subclass of ThetaGenerator that will calculate theta
            normalize: an instance of a subclass of Normalizer that will be applied
        """
        self.reset()
        self.generator = generator
        self.normalizer = normalizer


    def train(self, x, y):
        """
            x, y: features and results of training data
            there is no return value but this method must be called before the classifier can make any predictions
        """
        if x.shape[1] != self.x.shape[1]:
            raise ValueError('all training inputs must have the same number of features')
        if y.shape[1] != 1:
            # If y is provided as a row instead of a column, just transpose it into a column
            if y.shape[0] == 1:
                y = y.T
            else:
                raise ValueError('all training results must have one column')
        if x.shape[0] != y.shape[0]:
            raise ValueError('you must provide the same number of input features as input results')

        self.x = np.append(self.x, x, 0)
        self.y = np.append(self.y, y, 0)
        if normalizer:
            normalizer.set_basis(self.x)

        self.theta = generator.calculate(normalizer.normalize(self.x) if normalizer else self.x, self.y)


    def predict(self, x):
        """
            x: features of testing data
            returns a vector of predicted values (or classes) for each row of feature data
        """
        if x.shape[1] != self.x.shape[1]:
            raise ValueError('this classifier was trained using inputs with %s features but this input has %s features'
                % (str(self.x.shape[1]), str(x.shape[1])))

        if normalizer:
            x = normalizer.normalize(x)
        m = x.shape[0]
        x = np.append(np.ones(m).reshape([m, 1]), x, 1)        

        # TODO: need a generalized way to apply sigmoid here for logistic regression
        return np.dot(x, self.theta)


    def reset(self):
        self.x = np.array()
        self.y = np.array()
        self.theta = None
        if self.normalizer:
            self.normalizer.reset()



class ThetaGenerator(object):

    def calculate(self, x, y):
        """
            x: feature array (this should NOT already include leading 1s for x_0)
            y: classification vector

            returns the resulting theta vector
        """
        raise NotImplementedError('all subclasses of ThetaGenerator must implement calculate')


class GradientDescent(ThetaGenerator):

    """
        cost_history: a list of the computed cost after each step of gradient descent
    """

    def __init__(self, cost_function, alpha, max_iter = 1000, min_change = None):
        """
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
        self.cost_function = cost_function
        self.alpha = alpha
        self.max_iter = max_iter
        self.min_change = min_change


    def calculate(self, x, y):
        
        # Pad x with leading zeros to act as x_0
        m = x.shape[0]

        x = np.append(np.ones(m).reshape([m, 1]), x, 1)
        self.x = x
        m, n = self.x.shape

        self.cost_history = []
        theta = np.zeros([n])
        for i in xrange(max_iter):
            cost, gradients = cost_function(x, y, theta)
            self.cost_history.append(cost)
            if cost == 0 or min_change and len(self.cost_history) > 1 and self.cost_history[-2] - cost <= min_change:
                break
            
            theta = theta - alpha * gradients

        return theta
