import numpy as np
from math import exp, log

sigmoid = lambda x: 1 / (1 + exp(-x))


class CostFunction(object):
    
    def calculate(self, x, y, theta):
        """
            Calculates the cost and feature gradients using the following arguments:
            x: the feature array
            y: the result vector
            theta: the current feature weights
            return value: a tuple whose first value is the cost and second value is a list of feature gradients
        """
        raise NotImplementedError('all subclasses of CostFunction must implement calculate')


class RegressionCost(CostFunction):

    def __init__(self, row_value_getter, row_cost_getter, regular_coeff=0):
        """
            reg_type: the type of regression to run.  options are:
                linear
                logistic
            regular_coeff: the coefficient to use for regularization.  Usually referred to as lambda,
                but for obvious reasons can't be named that.  If set to 0, regularization won't be used.
        """
        self.get_row_value = row_value_getter
        self.get_cost = row_cost_getter
        self.regular_coeff = regular_coeff


    def calculate(self, x, y, theta):
        """
            Calculates the regression cost and feature gradients using the following arguments:
            x: the feature array
            y: the result vector
            theta: the current feature weights
            return value: a tuple whose first value is the cost and second value is a list of feature gradients
        """

        m, n = x.shape
        if n != theta.shape[0]:
            raise ValueError('theta must have the same size as each feature vector')
        if m != y.shape[0]:
            raise ValueError('features and results must have the same number of observations')

        cost = 0
        gradients = np.zeros(n)

        # TODO: vectorize this
        for i in xrange(m):
            row = x[i,:]
            row_value = self.get_row_value(row)
            cost += self.get_cost(row_value, y[i])
            gradients += row * (row_value - y[i])

        if regular_coeff:
            cost += (regular_coeff / 2) * sum(theta[1:] ^ 2)
            for i in xrange(theta.shape[0] - 1):
                gradients[i + 1] += self.regular_coeff * theta[i + 1][0]

        cost /= m
        gradients /= m
        return cost, gradients


class LinearRegression(RegressionCost):
    def __init__(self, regular_coeff = 0):
        super(LinearRegression, self).__init__(
            (lambda row: np.dot(row, theta)),
            (lambda predicted, actual: ((predicted - actual) ** 2) * 2),
            regular_coeff)


class LogisticRegression(RegressionCost):
    def __init__(self, regular_coeff = 0):
        super(LogisticRegression, self).__init__(
            (lambda row: sigmodid(np.dot(row, theta))),
            (lambda predicted, actual: -log(predicted if actual else (1 - predicted))),
            regular_coeff)