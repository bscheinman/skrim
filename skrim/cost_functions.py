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
    cost = 0
    gradients = np.zeros(theta.shape[0])

    if reg_type == 'linear':
        get_row_value = lambda row: row * theta
        get_cost = lambda predicted, actual: (predicted - actual) ** 2
    elif reg_type == 'logistic':
        get_row_value = lambda row: sigmoid(row * theta)
        get_cost = lambda predicted, actual: -log((1 - predicted) if actual else predicted)
    else:
        raise ValueError('unknown regression type %s' % reg_type)

    m = x.shape[1]
    for i in xrange(m):
        row = x[i,:]
        row_value = get_row_value(row)
        cost += get_cost(row_value, y[i])
        gradients += row * (row_value - y[i])

    gradients = gradients / m
    return cost, gradients
