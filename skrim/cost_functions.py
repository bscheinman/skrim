import numpy as np

sigmoid = lambda x: 1 / (1 + exp(-x))

def regression_cost(x, y, theta, reg_type='linear'):
    """
        Calculates the regression cost and feature gradients using the following arguments:
        x: the feature array
        y: the result vector
        theta: the current feature weights
        reg_type: the type of regression to run.  options are:
            linear
            logistic
        return value: a tuple whose first value is the cost and second value is a list of feature gradients
    """

    n, m = x.shape
    if n != theta.shape[0]:
        raise ValueError('theta must have the same size as each feature vector')
    if m != y.shape[0]:
        raise ValueError('features and results must have the same number of observations')

    cost = 0
    gradients = np.zeros(n)

    if reg_type == 'linear':
        get_row_value = lambda row: np.dot(row, theta)
        get_cost = lambda predicted, actual: (predicted - actual) ** 2
    elif reg_type == 'logistic':
        get_row_value = lambda row: sigmoid(np.dot(row, theta))
        get_cost = lambda predicted, actual: -log(predicted if actual else (1 -predicted))
    else:
        raise ValueError('unknown regression type %s' % reg_type)

    for i in xrange(m):
        row = x[i,:]
        row_value = get_row_value(row)
        print row_value, y[i]
        cost += get_cost(row_value, y[i])
        gradients += row * (row_value - y[i])

    cost /= m
    gradients /= m
    return cost, gradients


lin_reg = lambda x, y, theta: regression_cost(x, y, theta, 'linear')
log_reg = lambda x, y, theta: regression_cost(x, y, theta, 'logistic')
