import numpy as np
from src.distances import euclidean_distances, manhattan_distances, cosine_distances


def find_mode(arr):
    """
    Return the mode (most common element) of `arr`.
    You may use your `numpy_practice` implementation from HW1.
    """
    unique_values, counts_unique_values = np.unique(arr, return_counts=True)
    return unique_values[np.argmax(counts_unique_values)]


class KNearestNeighbor():
    def __init__(self, n_neighbors, distance_measure='euclidean', aggregator="mode"):
        """
        K-Nearest Neighbor is a straightforward algorithm that can be highly
        effective. 

        You should not have to change this __init__ function, but it's
        important to understand how it works.

        Do not import or use these packages: fairlearn, scipy, sklearn, sys, importlib.

        Arguments:
            n_neighbors {int} -- Number of neighbors to use for prediction.
            distance_measure {str} -- Which distance measure to use. Can be
                'euclidean,' 'manhattan,' or 'cosine'. This is the distance measure
                that will be used to compare features to produce labels.
            aggregator {str} -- How to aggregate neighbors; either mean or mode.
        """
        self.n_neighbors = n_neighbors

        if aggregator == "mean":
            self.aggregator = np.mean
        elif aggregator == "mode":
            self.aggregator = find_mode
        else:
            raise ValueError(f"Unknown aggregator {aggregator}")

        if distance_measure == "euclidean":
            self.distance = euclidean_distances
        elif distance_measure == "manhattan":
            self.distance = manhattan_distances
        elif distance_measure == "cosine":
            self.distance = cosine_distances
        else:
            raise ValueError(f"Unknown distance {distance_measure}")

    def fit(self, features, targets):
        """
        Fit features, a numpy array of size (n_samples, n_features). For a KNN,
        this function should store the features and corresponding targets in
        class variables that can be accessed in the `predict` function. This
        will be a very short function.

        Arguments:
            features {np.ndarray} -- Features of each data point, shape of (n_samples, n_features).
            targets -- Target labels for each data point, shape of (n_samples, 1).
        """
        self.features, self.targets = features, targets 


    def predict(self, features):
        """
        Use the training data to predict labels on the test features.

        For each test example, find the `self.n_neighbors` closest train
        examples, in terms of the `self.distance` measure. Then, predict the
        test label by using `self.aggregator` among those nearest neighbors.

        Arguments:
            features {np.ndarray} -- Features of each data point, shape of
                (n_samples, n_features).

        Returns:
            labels {np.ndarray} -- Labels for each data point, of shape
                (n_samples, 1).
        """
        distances = self.distance(features, self.features)
        n_closest_row_indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]
        labels_of_n_closest = self.targets[n_closest_row_indices]
        labels = np.ones((features.shape[0], 1))
        for i in range(features.shape[0]): 
            labels[i]=self.aggregator(labels_of_n_closest[i, :]).flatten()
        return labels 
