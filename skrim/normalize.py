import numpy as np

def normalize(x):
    """
        Takes in a feature array and normalizes each column using
        mean and standard deviation so that each feature's values will be centered
        around 0 and shouldn't have overly large values

        x: the feature array to be processed
        return value: a 3-tuple with the following values:
            0: the normalized feature array
            1: a vector containing the means of each feature
            2: a vector containing the standard deviations of each feature

        The mean and standard deviation vectors should be kept in order to transform
        the features of any testing data being passed to the resulting model
    """

    m, n = x.shape
    means = np.ones(n)
    stds = np.ones(n)
    x_norm = np.zeros(x.shape)

    for i in range(n):
        feature = x[:,i]
        means[i] = np.mean(feature)
        stds[i] = np.std(feature)
        x_norm[:,i] = (feature - means[i]) / stds[i]

    return x_norm, means, stds
