#!/usr/bin/python

import numpy as np
import sys
import unittest

sys.path.append('..')
import classifier as cl
import cost_functions as cost
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

target_theta1 = np.array([[1.17525, 0.05209, 0.45196, -0.07445]]).T

test_vals1 = np.array([[-7, 3, 1], [0, 9, -4]])
target_predictions1 = np.array([[2.09205], [5.54069]])

PRECISION = 0.0001


class NormalizationTest(unittest.TestCase):

	def test_1(self):
		n = normalize.StandardNormalizer()
		n.set_basis(x1)
		self.assertTrue(np.max(n.normalize(x1) - x1_normal) < PRECISION)
		self.assertTrue(np.max(n.normalize(np.array([[5, -2, 0]])) - np.array([[0.55567, -0.27965, -0.43217]])) < PRECISION)


class LinearRegressionTest(unittest.TestCase):

	def test_1(self):
		"""
			normal equation, no normalization
		"""
		c = cl.Classifier(cl.NormalEquation())
		c.train(x1, y1)

		self.assertTrue(np.max(c.theta - target_theta1) < PRECISION)
		self.assertTrue(np.max(c.predict(test_vals1) - target_predictions1) < PRECISION)


	def test_2(self):
		"""
			normal equation, with normalization
		"""
		c = cl.Classifier(cl.NormalEquation(), normalizer = normalize.StandardNormalizer())
		c.train(x1, y1)

		self.assertTrue(np.max(c.predict(test_vals1) - target_predictions1) < PRECISION)


	def test_3(self):
		"""
			gradient descent, no normalization, no regularization
		"""

		c = cl.Classifier(cl.GradientDescent(cost.LinearRegression(), alpha=0.05, max_iter=1000))
		c.train(x1, y1)

		self.assertTrue(np.max(c.theta - target_theta1) < PRECISION)
		self.assertTrue(np.max(c.predict(test_vals1) - target_predictions1) < PRECISION)


	def test_4(self):
		"""
			gradient descent, with normalization, no regularization
		"""

		c = cl.Classifier(cl.GradientDescent(cost.LinearRegression(), alpha=0.05, max_iter=1000),
			normalizer = normalize.StandardNormalizer())
		c.train(x1, y1)

		self.assertTrue(np.max(c.predict(test_vals1) - target_predictions1) < PRECISION)


if __name__ == '__main__':
	unittest.main()
