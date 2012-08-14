import numpy as np
from math import exp


def gaussian_kernel_impl(x1, x2, sigma):
    if x1.shape != x2.shape:
        raise ValueError('x1 and x2 must have the same dimensionality')

    # note: the distance is actually the square root of this value,
    # but we end up squaring it again for the Gaussian function,
    # so it's quicker to just do neither
    dist = np.sum((x1 - x2) ** 2)

    return -exp(dist / (2 * sigma ** 2))


gaussian_kernel = lambda sigma: (lambda x1, x2: gaussian_kernel_impl(x1, x2, sigma))
