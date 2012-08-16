"""
    This file contains classes that perform classification on data sets with arbitrary features
    Data should already be cleaned and structured before they are passed here
"""

import numpy as np
from abc import abstractmethod

import cost_functions
import normalize
from skrimutils import euclidean_distance, pad_ones, sigmoid_curry


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

    def __init__(self, generator, normalizer=None):
        """
            generator: an instance of a subclass of ThetaGenerator that will calculate theta
            normalize: an instance of a subclass of Normalizer that will be applied (optional)
            regular_coeff: the value of lambda to use for the cost function.  This can be useful
                to avoid overfitting the training set

            subclasses that don't want to pad their input data with extra ones for an intercept term
            should set self.pad_x to False in their constructors
        """
        self.generator = generator
        self.normalizer = normalizer
        self.pad_x = True
        self.reset()

    def train(self, x, y):

        if len(y.shape) == 1:
            y = y.reshape((y.size, 1))
        elif len(y.shape) == 2:
            if y.shape[1] != 1:
                # If y is provided as a row instead of a column,
                # just transpose it into a column
                if y.shape[0] == 1:
                    y = y.T
                else:
                    raise ValueError('all training results must have one column')
        else:
            raise ValueError('y must be a 1- or 2-dimensional matrix')

        if x.shape[0] != y.shape[0]:
            raise ValueError('you must provide the same number of input features as input results')

        if self.normalizer:
            self.normalizer.set_basis(x)

        x_train = self.normalizer.normalize(x) if self.normalizer else x
        if self.pad_x:
            x_train = pad_ones(x_train)

        self.theta = self.generator.calculate(x_train, y)
        self.theta = self.theta.reshape([self.theta.shape[0], 1])

    def predict(self, x):
        if not self.theta.size:
            raise Exception('this classifier has not been trained yet')

        if self.normalizer:
            x = self.normalizer.normalize(x)

        n = x.shape[1]
        expected_n = self.theta.shape[0] - (1 if self.pad_x else 0)
        if n != expected_n:
            raise ValueError('this classifier was trained using inputs with %s features but this input has %s features'
                % (str(expected_n), str(n)))

        x_predict = pad_ones(x) if self.pad_x else x
        return x_predict.dot(self.theta)

    def reset(self):
        self.theta = None
        if self.normalizer:
            self.normalizer.reset()


class LogisticValueClassifier(LinearClassifier):
    """
        This class performs logistic regression and will return P(class=1) as its predictions
    """

    def __init__(self, generator, normalizer=None):
        super(LogisticValueClassifier, self).__init__(generator, normalizer)

    def train(self, x, y):

        if len(x.shape) != 2:
            raise ValueError('training data must be a 2-dimensional array')
        if not np.vectorize(lambda c: c in (0, 1))(y).all():
            raise ValueError('all training classes must be 0 or 1')
        super(LogisticValueClassifier, self).train(x, y)

    def predict(self, x):
        return sigmoid_curry(super(LogisticValueClassifier, self).predict(x))


class LogisticClassifier(LogisticValueClassifier):
    """
        This class is the same as LogisticValueClassifier except that it returns 0 or 1
        depending on whether the predicted probability is above a certain threshold
    """

    def __init__(self, generator, normalizer=None, threshold=0.5):
        super(LogisticClassifier, self).__init__(generator, normalizer)
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
                    lambda: LogisticValueClassifier(GradientDescent(
                        LogisticRegression(), 1, 1000), StandardNormalizer())
        """
        self.classifier_generator = classifier_generator or\
            (lambda: LogisticValueClassifier(GradientDescent(
                cost_functions.LogisticRegression(), 1, 1000), normalize.StandardNormalizer()))
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
            classes.append(y_class)
        return np.vectorize(lambda x: classes[x])(predictions.argmax(1).reshape((x.shape[0], 1)))

    def reset(self):
        self.classifiers.clear()


class SupportVectorMachine(LogisticClassifier):
    """
        This class implements a support vector machine.  It is extremely slow and should not be used
        for any non-trivial amount of data.  If you want to use an SVM for real data, you should look
        into an established SVM library such as LibSVM.
    """

    def __init__(self, kernel, generator, normalizer=normalize.RangeNormalizer()):
        # this is pretty hacky, but we don't want to pass a normalizer to the
        # base class because we don't want to re-normalize after calculating similarities
        super(SupportVectorMachine, self).__init__(generator, None)
        self.svm_normalizer = normalizer
        self.kernel = kernel
        self.pad_x = False

    def train(self, x, y):

        self.reset()
        if self.svm_normalizer:
            self.svm_normalizer.set_basis(x)
            self.landmarks = self.svm_normalizer.normalize(x)
        # we have to set landmarks first because we need it in order to calculate transformed features
        super(SupportVectorMachine, self).train(self.to_distances(self.landmarks), y)

    def predict(self, x):
        if self.svm_normalizer:
            x_transform = self.svm_normalizer.normalize(x)
        return super(SupportVectorMachine, self).predict(self.to_distances(x_transform))

    def to_distances(self, x):
        """
            x: a set of (original) features to convert to similarity features

            return value: the converted feature values of x ready to use for
                cost or prediction functions
        """

        if len(x.shape) != 2:
            raise ValueError('observation data must be a 2-dimensional array')
        m, n = x.shape
        if n != self.landmarks.shape[1]:
            raise ValueError('training data had %s features but you provided data with %s features'
                % (self.landmarks.shape[1], n))

        x_transform = np.zeros((m, self.landmarks.shape[0]))
        for i in xrange(m):
            observation = x[i]
            for j in xrange(self.landmarks.shape[0]):
                x_transform[i, j] = self.kernel(observation, self.landmarks[j])

        return x_transform

    def reset(self):
        super(SupportVectorMachine, self).reset()
        self.landmarks = np.array([[]])


class KNearestNeighbors(Classifier):
    """
        Implements the k-nearest neighbors algorithm to classify items.
        Each test observation is assigned to the class most prevalent among the k
        closest test observations
    """

    def __init__(self, k=1, dist_fun=None, weight=None):
        """
            k: the number of neighbors to consider
            dist_fun: the function to use to compute distance when determining nearest neighbors
            weight: a function to weight the value of each neighbor's class based on their distance
        """
        self.k = k
        self.dist_fun = dist_fun or euclidean_distance
        self.weight = weight or (lambda x: 1 / x ** 2)
        self.reset()

    def train(self, x, y):
        self.x = x
        self.y = y

    def nearest_neighbors(self, x):
        distances = []
        for i in xrange(self.x.shape[0]):
            distances.append((self.dist_fun(x, self.x[i, :]), self.y[i]))
        return sorted(distances, key=lambda val: val[0])[:self.k]

    def predict(self, x):
        predictions = np.zeros((x.shape[0], 1))
        for i_obs in xrange(x.shape[0]):
            nearest = self.nearest_neighbors(x[i_obs, :])
            class_values = {}
            for dist, y_class in nearest:
                if not y_class in class_values:
                    class_values[y_class] = 0
                class_values[y_class] += self.weight(dist)

            best_class, best_value = None, None
            for y_class, y_value in class_values.items():
                if best_value is None or y_value > best_value:
                    best_class, best_value = y_class, y_value
            predictions[i_obs] = best_class

        return predictions

    def reset(self):
        self.x = np.array([[]])
        self.y = np.array([[]])


class ThetaGenerator(object):

    @abstractmethod
    def calculate(self, x, y):
        """
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

    def __init__(self, cost_function, alpha, max_iter, min_change=0):
        """
            alpha: gradient descent step size
            max_iter: maximum number of iterations to perform
            min_change: if this is specified, descent will stop
                whenever the cost decrease is below this value
            cost_function: an instance of a subclass of CostFunction that will
                be used to evaluate the cost of theta after each step.
                It will be called as:
                cost_function.calculate(x, y, theta)
                and should return a tuple whose first value is the cost and
                whose second value is an array of gradients for each feature.
        """

        if alpha <= 0 or max_iter <= 0:
            raise ValueError(
                'you must provide positive values for alpha and max_iter')

        if min_change < 0:
            raise ValueError('negative values are invalid for min_change')

        self.alpha = alpha
        self.max_iter = max_iter
        self.min_change = min_change
        self.cost_function = cost_function

    def calculate(self, x, y):

        m, n = x.shape

        self.cost_history = []
        theta = np.zeros([n, 1])
        for i in xrange(self.max_iter):
            cost, gradients = self.cost_function.calculate(x, y, theta)
            self.cost_history.append(cost)
            if cost == 0.0 or self.min_change and len(self.cost_history) > 1 and self.cost_history[-2] - cost <= self.min_change:
                break

            theta = theta - self.alpha * gradients

        return theta


class NormalEquation(ThetaGenerator):
    """
        uses the normal equation to solve for the exact value of theta.  this is only suitable for smaller data sets
    """

    def calculate(self, x, y):
        x_t = x.T
        # theta = (X' * X)^-1 * X' * y
        return np.dot(np.dot(np.linalg.inv((np.dot(x_t, x))), x_t), y)
