from abc import abstractmethod
import numpy as np

class Normalizer(object):
    """

    """

    @abstractmethod
    def set_basis(self, x):
        """
            x: training data to use as the basis for all normalizing
            this method must be called before normalizing any data
        """


    @abstractmethod
    def normalize(self, x):
        """
            x: testing (or training) data to be normalized
            returns input data normalized to the scale of the basis data
        """


    @abstractmethod
    def reset(self):
        """
            resets the normalizer to be in its initial state
        """




class StandardNormalizer(Normalizer):

    """
        Normalizes each column of the feature array using mean and standard deviation so that
        each feature's values will be centered around 0 and shouldn't have overly large values
    """

    def __init__(self):
        self.reset()


    def set_basis(self, x):
        self.means = np.mean(x, 0)
        self.stds = np.std(x, 0)


    def normalize(self, x):
        n = x.shape[1]
        if n != self.means.shape[0]:
            raise ValueError('this normalizer is initialized to take input with %s features but you provided input with %s features'
                % (str(self.means.shape[0])), x.shape[1])

        return (x - self.means) / self.stds


    def reset(self):
        self.means = np.array([])
        self.stds = np.array([])
