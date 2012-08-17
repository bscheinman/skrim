import numpy as np
from abc import abstractmethod


class Clusterer(object):

    @abstractmethod
    def cluster(self, x):
        """
            x: an m x n array containing features for each element to be clustered

            return value: an m-element array containing the cluster number for each element
                NOTE: clusters are 0-indexed
        """


class KMeansClusterer(Clusterer):

    def __init__(self, k, n_iter):
        """
            k: the number of clusters to create
                NOTE: it is possible (though unlikely) that the actual number of clusters created will be
                smaller than k because any cluster with 0 elements at an intermediate step will be removed
            n_iter: the number of random initializations to try
        """
        self.k = k
        self.n_iter = n_iter

    def cluster(self, x):

        m, n = x.shape

        best_clusters = np.zeros(m, dtype=np.int)
        best_cost = None

        for cluster_iter in range(self.n_iter):
            # initialize random starting points for clusters
            centroid_indices = np.vectorize(int)(m * np.random.sample(self.k))
            centroids = x[centroid_indices, :]
            clusters = np.zeros(m, dtype=np.int)

            # compute clusters based on this initialization
            while True:
                old_clusters = np.copy(clusters)

                # find the closest centroid for each element
                for i in xrange(m):
                    distances = np.sqrt(np.sum((centroids - x[i, :].reshape((1, n))) ** 2, 1))
                    closest_centroid = np.argmin(distances, 0)
                    clusters[i] = closest_centroid

                # stop once the clusters are done changing
                if (old_clusters == clusters).all():
                    break

                # re-center the centroids based on their new points
                centroid_totals = np.zeros(centroids.shape)
                centroid_counts = np.zeros(centroids.shape[0])
                for i in xrange(m):
                    cluster = clusters[i]
                    centroid_totals[cluster] += x[i]
                    centroid_counts[cluster] += 1

                # remove any centroids that have no elements
                centroids_to_keep = centroid_counts != 0
                centroid_totals = centroid_totals[centroids_to_keep, :]
                centroid_counts = centroid_counts[centroids_to_keep, :]

                centroids = centroid_totals / centroid_counts.reshape((centroid_counts.shape[0], 1))

            # find total cost of prediction for these initial values
            # and update best estimate if necessary
            predicted_values = np.zeros(x.shape)
            for i in xrange(m):
                predicted_values[i] = centroids[clusters[i]]
            prediction_errors = predicted_values - x
            cost = sum(np.sqrt(np.sum(prediction_errors ** 2, 1)))
            if best_cost is None or cost < best_cost:
                best_clusters = clusters
                best_cost = cost

        # finally, after all iterations are complete, return the clustering that produced the lowest cost
        return best_clusters
