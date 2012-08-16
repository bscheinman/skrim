import numpy as np
from abc import abstractmethod
from math import floor

from skrimutils import euclidean_distance


class Clusterer(object):

    @abstractmethod
    def cluster(self, x):
        """
            x: an m x n array containing features for each element to be clustered

            return value: an m-element array containing the cluster number for each element
                NOTE: clusters are 0-indexed
        """


class KMeansClusterer(Clusterer):

    def __init__(self, k, n_iter, dist_fun = None):
        """
            k: the number of clusters to create
                NOTE: it is possible (though unlikely) that the actual number of clusters created will be
                smaller than k because any cluster with 0 elements at an intermediate step will be removed
            n_iter: the number of random initializations to try
        """
        self.k = k
        self.n_iter = n_iter
        self.dist_fun = dist_fun or euclidean_distance

    def cluster(self, x):

        m = x.shape[0]

        best_clusters = np.zeros(m)
        best_cost = None

        for cluster_iter in range(self.n_iter):
            # initialize random starting points for clusters
            centroid_indices = np.vectorize(floor)(m * np.random.sample(self.k))
            centroids = x[centroid_indices, :]
            clusters = np.zeros(m)

            # compute clusters based on this initialization
            while True:
                old_clusters = clusters

                # find the closest centroid for each element
                for i in xrange(m):
                    distances = np.sqrt(np.sum((centroids - x[i, :]) ** 2, 1))
                    closest_centroid = np.argmax(distances, 0)
                    clusters[i] = closest_centroid

                # stop once the clusters are done changing
                if (old_clusters == clusters).all():
                    break

                # re-center the centroids based on their new points
                centroid_totals = np.zeros(centroids.shape)
                centroid_counts = np.zeros((centroids.shape[0], 1))
                for i in xrange(m):
                    centroid = centroids[i]
                    centroid_totals[centroid] += x[i]
                    centroid_counts[centroid] += 1

                # remove any centroids that have no elements
                centroids_to_keep = centroid_counts != 0
                centroid_totals = centroid_totals[centroids_to_keep]
                centroid_counts = centroid_counts[centroids_to_keep]

                centroids = centroid_totals / centroid_counts

            # find total cost of prediction for these initial values
            # and update best estimate if necessary
            predicted_values = np.array(x.shape)
            for i in xrange(m):
                predicted_values[i] = centroids[clusters[i]]
            prediction_errors = predicted_values - x
            cost = sum(np.sqrt(np.sum(prediction_errors ** 2, 1)))
            if best_cost is None or cost < best_cost:
                best_clusters = clusters
                best_cost = cost

        # finally, after all iterations are complete, return the clustering that produced the lowest cost
        return best_clusters

