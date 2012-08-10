"""
    This file contains classes that perform classification on data sets with arbitrary features
    Data should already be cleaned and structured before they are passed here
"""

from abc import abstractmethod
import numpy as np

import cost_functions as cost
from skrimutils import pad_ones, sigmoid_curry


class Classifier(object):

    def __init__(self, generator, normalizer=None):
        """
            generator: an instance of a subclass of ThetaGenerator that will calculate theta
            normalize: an instance of a subclass of Normalizer that will be applied (optional)

            many subclasses of Classifier will want to define self.cost_function
        """
        self.generator = generator
        self.normalizer = normalizer
        self.cost_function = None
        self.reset()

    def train(self, x, y):
        """
            x, y: features and results of training data
            
            there is no return value but this method must be called before the classifier can make any predictions
        """
        if self.x.size and x.shape[1] != self.x.shape[1]:
            raise ValueError(
                'all training inputs must have the same number of features')
        if y.shape[1] != 1:
            # If y is provided as a row instead of a column,
            # just transpose it into a column
            if y.shape[0] == 1:
                y = y.T
            else:
                raise ValueError('all training results must have one column')
        if x.shape[0] != y.shape[0]:
            raise ValueError('you must provide the same number of input features as input results')

        self.x = np.append(self.x, x, 0) if self.x.size else x
        self.y = np.append(self.y, y, 0) if self.y.size else y

        if self.normalizer:
            self.normalizer.set_basis(self.x)

        self.theta = self.generator.calculate(self.cost_function,
            self.normalizer.normalize(self.x) if self.normalizer else self.x, self.y)
        self.theta = self.theta.reshape([self.theta.shape[0], 1])

    def predict(self, x):
        """
            x: features of testing data
            returns a vector of predicted values (or classes) for each row of feature data
        """
        if not self.x.size:
            raise Exception('this classifier has not been trained yet')

        if self.normalizer:
            x = self.normalizer.normalize(x)
        x = pad_ones(x)

        return self._predict_impl(x)

    @abstractmethod
    def _predict_impl(self, x):
        """
            performs the prediction logic on the test input and returns a matrix with a single prediction in each row
        """

    def reset(self):
        self.x = np.array([])
        self.y = np.array([])
        self.theta = None
        if self.normalizer:
            self.normalizer.reset()


class LinearClassifier(Classifier):

    def __init__(self, generator, normalizer=None, regular_coeff=0):
        super(LinearClassifier, self).__init__(generator, normalizer)
        self.cost_function = cost.LinearRegression(regular_coeff=regular_coeff)

    def _predict_impl(self, x):
        n = x.shape[1]
        if n != self.theta.shape[0]:
            raise ValueError('this classifier was trained using inputs with %s features but this input has %s features'
                % (str(self.x.shape[1]), str(n)))

        return np.dot(x, self.theta)


class LogisticClassifier(LinearClassifier):

    def __init__(self, generator, normalizer=None, regular_coeff=0, threshold=0.5):
        super(LogisticClassifier, self).__init__(generator, normalizer)
        self.cost_function = cost.LogisticRegression(regular_coeff=regular_coeff)
        self.threshold = threshold

    def _predict_impl(self, x):
        return np.vectorize(lambda x: 1 if x >= self.threshold else 0)\
            (sigmoid_curry(super(LogisticClassifier, self)._predict_impl(x)))



class ThetaGenerator(object):

    @abstractmethod
    def calculate(self, cost_function, x, y):
        """
            cost_function: an instance of a subclass of CostFunction that will
                be used to evaluate the cost of theta after each step.
                It will be called as:
                cost_function.calculate(x, y, theta)
                and should return a tuple whose first value is the cost and
                whose second value is an array of gradients for each feature.
            x: feature array
                (this should NOT already include leading 1s for x_0)
            y: classification vector

            returns the resulting theta vector
        """


class GradientDescent(ThetaGenerator):

    """
        cost_history: a list of the computed cost
            after each step of gradient descent
    """

    def __init__(self, alpha, max_iter, min_change=0):
        """
            alpha: gradient descent step size
            max_iter: maximum number of iterations to perform
            min_change: if this is specified, descent will stop
                whenever the cost decrease is below this value
        """

        if alpha <= 0 or max_iter <= 0:
            raise ValueError(
                'you must provide positive values for alpha and max_iter')

        if min_change < 0:
            raise ValueError('negative values are invalid for min_change')

        self.alpha = alpha
        self.max_iter = max_iter
        self.min_change = min_change

    def calculate(self, cost_function, x, y):

        m = x.shape[0]
        x = pad_ones(x)
        m, n = x.shape

        self.cost_history = []
        theta = np.zeros([n, 1])
        for i in xrange(self.max_iter):
            cost, gradients = cost_function.calculate(x, y, theta)
            self.cost_history.append(cost)
            if cost == 0.0 or self.min_change and len(self.cost_history) > 1 and self.cost_history[-2] - cost <= self.min_change:
                break

            theta = theta - self.alpha * gradients

        return theta


class NormalEquation(ThetaGenerator):
    """
        uses the normal equation to solve for the exact value of theta.  this is only suitable for smaller data sets
    """

    # cost_function isn't used in this class and will be ignored
    def calculate(self, cost_function, x, y):
        x_calc = pad_ones(x)
        x_calc_t = x_calc.T
        return np.dot(np.dot(np.linalg.inv((np.dot(x_calc_t, x_calc))), x_calc_t), y)
