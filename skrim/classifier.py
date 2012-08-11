"""
    This file contains classes that perform classification on data sets with arbitrary features
    Data should already be cleaned and structured before they are passed here
"""

from abc import abstractmethod
import numpy as np

import cost_functions as cost
import normalize
from skrimutils import pad_ones, sigmoid_curry


class Classifier(object):

    @abstractmethod
    def train(self, x, y):
        """
            x, y: features and results of training data

            there is no return value but this method must be called before the classifier can make any predictions
        """

    @abstractmethod
    def predict(self, x):
        """
            x: features of testing data
            returns a vector of predicted values (or classes) for each row of feature data
        """

    @abstractmethod
    def reset(self):
        pass


class LinearClassifier(Classifier):
    """
        This class performs linear regression
    """

    def __init__(self, generator, normalizer=None, regular_coeff=0):
        """
            generator: an instance of a subclass of ThetaGenerator that will calculate theta
            normalize: an instance of a subclass of Normalizer that will be applied (optional)
            regular_coeff: the value of lambda to use for the cost function.  This can be useful
                to avoid overfitting the training set
        """
        self.generator = generator
        self.normalizer = normalizer
        self.cost_function = cost.LinearRegression(regular_coeff=regular_coeff)
        self.reset()

    def train(self, x, y):
        if self.x.size and x.shape[1] != self.x.shape[1]:
            raise ValueError('all training inputs must have the same number of features')
        if y.shape[1] != 1:
            # If y is provided as a row instead of a column,
            # just transpose it into a column
            if y.shape[0] == 1:
                y = y.T
            else:
                raise ValueError('all training results must have one column')
        if x.shape[0] != y.shape[0]:
            raise ValueError('you must provide the same number of input features as input results')

        self.x = np.append(self.x, x, 0) if self.x.size else x
        self.y = np.append(self.y, y, 0) if self.y.size else y

        if self.normalizer:
            self.normalizer.set_basis(self.x)

        self.theta = self.generator.calculate(self.cost_function,
            self.normalizer.normalize(self.x) if self.normalizer else self.x, self.y)
        self.theta = self.theta.reshape([self.theta.shape[0], 1])

    def predict(self, x):
        if not self.x.size:
            raise Exception('this classifier has not been trained yet')

        if self.normalizer:
            x = self.normalizer.normalize(x)
        x = pad_ones(x)

        n = x.shape[1]
        if n != self.theta.shape[0]:
            raise ValueError('this classifier was trained using inputs with %s features but this input has %s features'
                % (str(self.x.shape[1]), str(n)))

        return x.dot(self.theta)

    def reset(self):
        self.x = np.array([])
        self.y = np.array([])
        self.theta = None
        if self.normalizer:
            self.normalizer.reset()


class LogisticValueClassifier(LinearClassifier):
    """
        This class performs logistic regression and will return P(class=1) as its predictions
    """

    def __init__(self, generator, normalizer=None, regular_coeff=0):
        super(LogisticValueClassifier, self).__init__(generator, normalizer)
        self.cost_function = cost.LogisticRegression(regular_coeff=regular_coeff)

    def train(self, x, y):
        if not np.vectorize(lambda x: x in (0, 1))(y).all():
            raise ValueError('all training classes must be 0 or 1')
        super(LogisticValueClassifier, self).train(x, y)

    def predict(self, x):
        return sigmoid_curry(super(LogisticValueClassifier, self).predict(x))


class LogisticClassifier(LogisticValueClassifier):
    """
        This class is the same as LogisticValueClassifier except that it returns 0 or 1
        depending on whether the predicted probability is above a certain threshold
    """

    def __init__(self, generator, normalizer=None, regular_coeff=0, threshold=0.5):
        super(LogisticClassifier, self).__init__(generator, normalizer, regular_coeff)
        self.threshold = threshold

    def predict(self, x):
        return np.vectorize(lambda x: 1 if x >= self.threshold else 0)(
            super(LogisticClassifier, self).predict(x))


class OneVsAllClassifier(Classifier):
    """
        This class uses other classifiers to perform one-vs-all classification.
        It will create an instance of the provided classifier for each unique
        class in the training set provided and train each one using that training
        data.  For each testing record, it will predict the class whose classifier
        provides the highest prediction value.
    """

    def __init__(self, classifier_generator=None):
        """
            classifier_generator: a function that takes no arguments and returns an instance
                of another classifier to use for a single observation class.  For example, the
                default value is:
                    lambda: LogisticValueClassifier(GradientDescent(1, 1000), StandardNormalizer())
        """
        self.classifier_generator = classifier_generator or\
            (lambda: LogisticValueClassifier(GradientDescent(1, 1000), normalize.StandardNormalizer()))
        self.classifiers = {}

    def train(self, x, y):
        self.reset()
        for y_class in np.unique(y):
            y_classifier = self.classifier_generator()
            y_train = np.vectorize(lambda x: 1 if x == y_class else 0)(y)
            y_classifier.train(x, y_train)
            self.classifiers[y_class] = y_classifier

    def predict(self, x):
        # predictions has a row for each test record, and a column for each possible y class
        # predictions[i, j] represents the probability that the classifier for class j
        # assigned to x_i
        predictions = np.array([])
        classes = []
        for y_class, y_classifier in self.classifiers.iteritems():
            y_prediction = y_classifier.predict(x)
            predictions = np.append(predictions, y_prediction, 1) if predictions.size else y_prediction
            #predictions = np.append(predictions, y_classifier.predict(x), 1)
            classes.append(y_class)
        return np.vectorize(lambda x: classes[x])(predictions.argmax(1).reshape((x.shape[0], 1)))

    def reset(self):
        self.classifiers.clear()


class ThetaGenerator(object):

    @abstractmethod
    def calculate(self, cost_function, x, y):
        """
            cost_function: an instance of a subclass of CostFunction that will
                be used to evaluate the cost of theta after each step.
                It will be called as:
                cost_function.calculate(x, y, theta)
                and should return a tuple whose first value is the cost and
                whose second value is an array of gradients for each feature.
            x: feature array
                (this should NOT already include leading 1s for x_0)
            y: classification vector

            returns the resulting theta vector
        """


class GradientDescent(ThetaGenerator):

    """
        cost_history: a list of the computed cost
            after each step of gradient descent
    """

    def __init__(self, alpha, max_iter, min_change=0):
        """
            alpha: gradient descent step size
            max_iter: maximum number of iterations to perform
            min_change: if this is specified, descent will stop
                whenever the cost decrease is below this value
        """

        if alpha <= 0 or max_iter <= 0:
            raise ValueError(
                'you must provide positive values for alpha and max_iter')

        if min_change < 0:
            raise ValueError('negative values are invalid for min_change')

        self.alpha = alpha
        self.max_iter = max_iter
        self.min_change = min_change

    def calculate(self, cost_function, x, y):

        m = x.shape[0]
        x = pad_ones(x)
        m, n = x.shape

        self.cost_history = []
        theta = np.zeros([n, 1])
        for i in xrange(self.max_iter):
            cost, gradients = cost_function.calculate(x, y, theta)
            self.cost_history.append(cost)
            if cost == 0.0 or self.min_change and len(self.cost_history) > 1 and self.cost_history[-2] - cost <= self.min_change:
                break

            theta = theta - self.alpha * gradients

        return theta


class NormalEquation(ThetaGenerator):
    """
        uses the normal equation to solve for the exact value of theta.  this is only suitable for smaller data sets
    """

    # cost_function isn't used in this class and will be ignored
    def calculate(self, cost_function, x, y):
        x_calc = pad_ones(x)
        x_calc_t = x_calc.T
        return np.dot(np.dot(np.linalg.inv((np.dot(x_calc_t, x_calc))), x_calc_t), y)
