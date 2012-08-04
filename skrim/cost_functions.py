import numpy as np
from math import exp, log

sigmoid = lambda x: 1 / (1 + exp(-x))

def regression_cost(x, y, theta, reg_type='linear', regular_coeff=0):
    """
        Calculates the regression cost and feature gradients using the following arguments:
        x: the feature array
        y: the result vector
        theta: the current feature weights
        reg_type: the type of regression to run.  options are:
            linear
            logistic
        regular_coeff: the coefficient to use for regularization.  Usually referred to as lambda,
            but for obvious reasons can't be named that.  If set to 0, regularization won't be used.
        return value: a tuple whose first value is the cost and second value is a list of feature gradients
    """

    m, n = x.shape
    if n != theta.shape[0]:
        raise ValueError('theta must have the same size as each feature vector')
    if m != y.shape[0]:
        raise ValueError('features and results must have the same number of observations')

    cost = 0
    gradients = np.zeros(n)

    if reg_type == 'linear':
        get_row_value = lambda row: np.dot(row, theta)
        get_cost = lambda predicted, actual: ((predicted - actual) ** 2) * 2
    elif reg_type == 'logistic':
        get_row_value = lambda row: sigmoid(np.dot(row, theta))
        get_cost = lambda predicted, actual: -log(predicted if actual else (1 - predicted))
    else:
        raise ValueError('unknown regression type %s' % reg_type)

    for i in xrange(m):
        row = x[i,:]
        row_value = get_row_value(row)
        cost += get_cost(row_value, y[i])
        gradients += row * (row_value - y[i])

    if regular_coeff:
        cost += (regular_coeff / 2) * sum(theta[1:] ^ 2)
        for i in xrange(theta.shape[0] - 1):
            gradients[i + 1] += regular_coeff * theta[i + 1][0];

    cost /= m
    gradients /= m
    return cost, gradients


lin_reg = lambda x, y, theta: regression_cost(x, y, theta, 'linear')
log_reg = lambda x, y, theta: regression_cost(x, y, theta, 'logistic')
