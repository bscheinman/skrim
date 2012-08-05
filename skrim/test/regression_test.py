#!/usr/bin/python

import numpy as np
import sys
import unittest

sys.path.append('..')
import classifier as cl
import cost_functions as cost
import normalize


class LinearRegressionTest(unittest.TestCase):

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

	y1 = np.array([[-3, 4, -5, -1, 0, 7, 3, -5, 4, 4]]).T

	target_theta1 = np.array([[1.17525, 0.05209, 0.45196, -0.07445]]).T

	test_vals1 = np.array([[-7, 3, 1], [0, 9, -4]])
	target_predictions1 = np.array([[2.09205], [5.54069]])

	def test_1(self):
		"""
			normal equation, no normalization
		"""
		c = cl.Classifier(cl.NormalEquation())
		c.train(self.x1, self.y1)

		self.assertTrue(np.max(c.theta - self.target_theta1) < 0.0001)
		self.assertTrue(np.max(c.predict(self.test_vals1) - self.target_predictions1) < 0.0001)


	def test_2(self):
		"""
			gradient descent, no normalization, no regularization
		"""

		c = cl.Classifier(cl.GradientDescent(cost.LinearRegression(), alpha=0.05, max_iter=1000))
		c.train(self.x1, self.y1)

		self.assertTrue(np.max(c.theta - self.target_theta1) < 0.0001)
		self.assertTrue(np.max(c.predict(self.test_vals1) - self.target_predictions1) < 0.0001)


if __name__ == '__main__':
	unittest.main()
