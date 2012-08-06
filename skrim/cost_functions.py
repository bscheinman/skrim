from abc import abstractmethod
import numpy as np
from math import log
import skrimutils as util


class CostFunction(object):
    
    @abstractmethod
    def calculate(self, x, y, theta):
        """
            Calculates the cost and feature gradients using the following arguments:
            x: the feature array
            y: the result vector
            theta: the current feature weights
            return value: a tuple whose first value is the cost and second value is a list of feature gradients
        """


class RegressionCost(CostFunction):

    def __init__(self, regular_coeff=0):
        """
            regular_coeff: the coefficient to use for regularization.  Usually referred to as lambda,
                but for obvious reasons can't be named that.  If set to 0, regularization won't be used.
        """
        self.regular_coeff = float(regular_coeff)


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

        predicted = self._get_values(x, theta)
        cost = sum(self._get_costs(predicted, y))
        gradients = sum(x * (predicted - y), 0).reshape([n, 1])

        if self.regular_coeff:
            cost += (self.regular_coeff / 2) * sum(theta[1:] ** 2)
            for i in xrange(theta.shape[0]):
                if not i:
                    continue
                gradients[i] += self.regular_coeff * theta[i][0]

        cost /= m
        gradients /= m
        return cost, gradients


    @abstractmethod
    def _get_values(self, x, theta):
        """
            calculates the predicted values for each observation given the observed values and current feature weights

            x: the feature array
            theta: the current feature weights
        """


    @abstractmethod
    def _get_costs(self, predicted, actual):
        """
            calculates the costs for each prediction

            predicted: an array of predicted output values
            actual: an array of actual observed output values
        """


class LinearRegression(RegressionCost):
    
    def _get_values(self, x, theta):
        return np.dot(x, theta)

    def _get_costs(self, predicted, actual):
        return ((predicted - actual) ** 2) * 2


class LogisticRegression(RegressionCost):
    
    def _get_values(self, x, theta):
        return util.sigmoid_curry(np.dot(x, theta))

    def _get_costs(self, predicted, actual):
        return -np.log(predicted * actual + (1 - predicted) * (1 - actual))


class NeuralNetCost(CostFunction):

    # TODO: refactor to work with an arbitrary number of levels
    def __init__(self, n_input, n_hidden, n_labels, regular_coeff = 0):
        """
            n_input: the number of input features
            n_hidden: the number of nodes in the (first) hidden layer of the neural net
            n_labels: the number of possible output labels
            regular_coeff: the value to use for lambda (optional)
        """
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_labels = n_labels
        self.regular_coeff = float(regular_coeff)


    def calculate(self, x, y, theta):

        theta_1 = theta[0 : self.n_hidden * (self.n_input + 1)].reshape([self.n_hidden, self.n_input + 1])
        theta_2 = theta[self.n_hidden * (self.n_input + 1) : theta.size].reshape([self.n_labels, self.n_hidden + 1])

        m = x.shape[0]
        x = util.pad_ones(x)
        cost = 0

        z2 = np.dot(x, theta_1.T)
        a2 = util.sigmoid_curry(z2)
        a2_input = util.pad_ones(a2)
        # a3 represents the predicted values
        a3 = util.sigmoid_curry(np.dot(a2_input, theta_2.T))
        actual = np.zeros(a3.shape)
        for i in xrange(actual.shape[1]):
            actual[:,i] = (y == i).reshape(10)
        cost = -sum(sum(actual * np.log(a3) + (1 - actual) * np.log(1 - a3)))

        d3 = a3 - actual
        d2 = np.dot(d3, theta_2[:,1:]) * util.sigmoid_gradient_curry(z2)

        theta_1_grad = np.dot(d2.T, x)
        theta_2_grad = np.dot(d3.T, util.pad_ones(a2))

        # adjust for regularization
        if self.regular_coeff:
            cost += (self.regular_coeff / 2) * (sum(theta_1[:,1:] ** 2) + sum(theta_2[:,1:] ** 2))
            theta_1_grad[:,1:] += self.regular_coeff * theta_1[:,1:]
            theta_2_grad[:,1:] += self.regular_coeff * theta_2[:,1:]

        cost /= m
        theta_1_grad /= m
        theta_2_grad /= m

        grad = np.append(theta_1_grad.reshape(theta_1_grad.size, 1),
            theta_2_grad.reshape(theta_2_grad.size, 1), 0)

        return cost, grad
