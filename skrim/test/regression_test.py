#!/usr/bin/python

import numpy as np
import sys
import unittest

sys.path.append('..')
import classifier as cl
import cost_functions as cf
import normalize

x1 = np.array([
    [1, 0, -7],
    [-5, 0, -3],
    [8, -2, 1],
    [-2, -3, 4],
    [5, -9, 6],
    [8, 5, 2],
    [4, 6, 9],
    [1, -4, 8],
    [8, -5, -5],
    [-5, 5, 9],
])

x1_normal = np.array([
    [-0.26754, 0.15058, -1.69266],
    [-1.50236, 0.15058, -0.97238],
    [1.17308, -0.27965, -0.25210],
    [-0.88495, -0.49477, 0.28811],
    [0.55567, -1.78546, 0.64825],
    [1.17308, 1.22616, -0.07203],
    [0.34987, 1.44128, 1.18847],
    [-0.26754, -0.70988, 1.00840],
    [1.17308, -0.92500, -1.33252],
    [-1.50236, 1.22616, 1.18847]
])

y1 = np.array([[-3, 4, -5, -1, 0, 7, 3, -5, 4, 4]]).T
y1_logistic = np.array([[1, 0, 0, 0, 1, 0, 1, 1, 0, 1]]).T
y1_multiclass = np.array([[1, 2, 4, 2, 1, 2, 3, 4, 4, 3]]).T

target_theta1 = np.array([[1.17525, 0.05209, 0.45196, -0.07445]]).T
target_theta1_logistic = np.array([[-0.30492, -0.08170, -0.02009, 0.19150]]).T

test_vals1 = np.array([[-7, 3, 1], [0, 9, -4]])
target_predictions1 = np.array([[2.09205], [5.54069]])
target_predictions1_logistic = np.array([[0.59826], [0.22240]])
target_multiclass1 = np.array([[2], [2]])

PRECISION = 0.0001


class NormalizationTest(unittest.TestCase):

    def test_1(self):
        n = normalize.StandardNormalizer()
        n.set_basis(x1)
        self.assertTrue(np.max(np.abs((n.normalize(x1) - x1_normal))) < PRECISION)
        self.assertTrue(np.max(np.abs(n.normalize(np.array([[5, -2, 0]])) - np.array([[0.55567, -0.27965, -0.43217]]))) < PRECISION)


class LinearRegressionTest(unittest.TestCase):

    def test_1(self):
        """
            normal equation, no normalization
        """
        c = cl.LinearClassifier(cl.NormalEquation())
        c.train(x1, y1)

        self.assertTrue(np.max(np.abs(c.theta - target_theta1)) < PRECISION)
        self.assertTrue(np.max(np.abs(c.predict(test_vals1) - target_predictions1)) < PRECISION)

    def test_2(self):
        """
            normal equation, with normalization
        """
        c = cl.LinearClassifier(cl.NormalEquation(), normalizer=normalize.StandardNormalizer())
        c.train(x1, y1)

        self.assertTrue(np.max(np.abs(c.predict(test_vals1) - target_predictions1)) < PRECISION)

    def test_3(self):
        """
            gradient descent, no normalization, no regularization
        """

        c = cl.LinearClassifier(cl.GradientDescent(
            cf.LinearRegression(), alpha=0.05, max_iter=1000, min_change=1e-10))
        c.train(x1, y1)

        self.assertTrue(np.max(np.abs(c.theta - target_theta1)) < PRECISION)
        self.assertTrue(np.max(np.abs(c.predict(test_vals1) - target_predictions1)) < PRECISION)

    def test_4(self):
        """
            gradient descent, with normalization, no regularization
        """

        c = cl.LinearClassifier(cl.GradientDescent(
                cf.LinearRegression(), alpha=0.05, max_iter=1000, min_change=1e-10),
            normalizer=normalize.StandardNormalizer())
        c.train(x1, y1)

        self.assertTrue(np.max(np.abs(c.predict(test_vals1) - target_predictions1)) < PRECISION)


class LogisticRegressionTest(unittest.TestCase):

    def test_1(self):
        """
            gradient descent, no normalization
        """
        c = cl.LogisticValueClassifier(cl.GradientDescent(
            cf.LogisticRegression(), alpha=0.1, max_iter=1000))
        c.train(x1, y1_logistic)

        self.assertTrue(np.max(np.abs(c.theta - target_theta1_logistic)) < PRECISION)
        self.assertTrue(np.max(np.abs(c.predict(test_vals1) - target_predictions1_logistic)) < PRECISION)


class OneVsAllTest(unittest.TestCase):

    def test_1(self):
        c = cl.OneVsAllClassifier(lambda:\
            cl.LogisticValueClassifier(cl.GradientDescent(
                cf.LogisticRegression(), 1, 1000), normalize.RangeNormalizer()))
        c.train(x1, y1_multiclass)

        self.assertTrue((c.predict(test_vals1) == target_multiclass1).all())


if __name__ == '__main__':
    unittest.main()
