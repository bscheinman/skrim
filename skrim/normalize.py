import numpy as np

class Normalizer(object):
    """

    """

    def set_basis(self, x):
        """
            x: training data to use as the basis for all normalizing
            this method must be called before normalizing any data
        """
        raise NotImplementedError('all subclasses of Normalizer must implement set_basis')


    def normalize(self, x):
        """
            x: testing (or training) data to be normalized
            returns input data normalized to the scale of the basis data
        """
        raise NotImplementedError('all subclasses of Normalizer must implement normalize')




class StandardNormalizer(Normalizer):

    """
        Normalizes each column of the feature array using mean and standard deviation so that
        each feature's values will be centered around 0 and shouldn't have overly large values
    """

    def __init__(self):
        self.means = np.array()
        self.stds = np.array()

    def set_basis(self, x):
        self.means = np.mean(x, 0)
        self.stds = np.std(x, 0)
        return self.normalize(x)


    def normalize(self, x):
        n = x.shape[1]
        if n != self.means.shape[0]:
            raise ValueError('this normalizer is initialized to take input with %s features but you provided input with %s features'
                % (str(self.means.shape[0])), x.shape[1])

        return (x - self.means) / self.stds
