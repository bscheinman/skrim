#!/usr/bin/python

from clustering import KMeansClusterer
import numpy as np
import matplotlib.pyplot as pyplot

c = KMeansClusterer(3, 10)
x1 = np.random.randint(0, 10, (20, 2)) - 10
x2 = np.random.randint(0, 10, (20, 2))
x3 = np.random.randint(0, 10, (20, 2)) + 10
x = np.append(x1, x2, 0)
x = np.append(x, x3, 0)

clusters = c.cluster(x)

pyplot.scatter(x[:, 0], x[:, 1], c=clusters)
pyplot.show()
