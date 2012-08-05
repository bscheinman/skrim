import numpy as np
from math import exp, log

sigmoid = lambda x: 1 / (1 + exp(-x))
sigmoid_curry = np.vectorize(sigmoid)



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
            row_value_getter: a function that calculates the predicted value of an observation
                given the feature values and theta
            row_cost_getter: a function that calculates the cost of an observation given the
                predicted value and the observed value
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
        gradients = np.zeros([n,1])

        # TODO: vectorize this
        for i in xrange(m):
            row = x[i,:].reshape([1, n])
            row_value = self.get_row_value(row, theta)
            cost += self.get_cost(row_value, y[i])
            gradients += (row * (row_value - y[i])).T

        if self.regular_coeff:
            cost += (regular_coeff / 2) * sum(theta[1:] ^ 2)
            for i in xrange(theta.shape[0]):
                if not i:
                    continue
                gradients[i] += self.regular_coeff * theta[i][0]

        cost /= m
        gradients /= m
        return cost, gradients


class LinearRegression(RegressionCost):
    def __init__(self, regular_coeff = 0):
        super(LinearRegression, self).__init__(
            (lambda row, theta: np.dot(row, theta)),
            (lambda predicted, actual: ((predicted - actual) ** 2) * 2),
            regular_coeff)


class LogisticRegression(RegressionCost):
    def __init__(self, regular_coeff = 0):
        super(LogisticRegression, self).__init__(
            (lambda row, theta: sigmoid(np.dot(row, theta))),
            (lambda predicted, actual: -log(predicted if actual else (1 - predicted))),
            regular_coeff)


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
        self.regular_coeff = regular_coeff


    def calculate(self, x, y, theta):

        theta_1 = theta[0 : n_hidden * (n_input + 1) - 1].reshape([n_hidden, n_input + 1])
        theta_2 = theta[n_hidden * (n_input + 1) : theta.size - 1]\
            .reshape([n_labels, n_hidden + 1])

        m = x.shape[0]
        x = np.append(np.ones(m, 1), x, 1)
        cost = 0

        for i in xrange(m):
            a1 = x[i,:]
            a2 = sigmoid_curry(np.dot(a1, theta_1.T))
            a2_input = np.append(np.ones(n_hidden, 1), a2, 1)
            # a3 represents the predicted values
            a3 = sigmoid_curry(np.dot(a2_input, theta_2.T))
            actual = np.zeros(1, num_labels);
            actual[y[i]] = 1;
            cost -= sum(actual * log(a3) + (1 - actual) * log(1 - a3))

        if self.regular_coeff:
            cost += (regular_coeff / 2) * (sum(theta_1[:,1:] ** 2) + sum(theta_2[:,1:] ** 2))

        cost /= m
        return cost
