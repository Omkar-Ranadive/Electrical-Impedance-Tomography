import numpy as np


class CosineKMeans():
    def __init__(self, n_clusters, max_iterations=None):
        """
        Args:
            n_clusters (int): Number of clusters to cluster the given data into.
        """
        self.n_clusters = n_clusters
        self.means = None
        self.max_iterations = max_iterations


    def fit(self, features):
        """
        Fit KMeans to the given data using `self.n_clusters` number of clusters. 
        Features can have greater than 2 dimensions.

        Args:
            features (np.ndarray): array containing inputs of size (n_samples, n_features).
        Returns:
            None 
        """
        # Generate the initial mean values
        # np.random.seed(0)
        done = False
        # Set the initial means equal to some random points in the dataset
        indices = np.random.permutation(features.shape[0])
        self.means = np.zeros((self.n_clusters, features.shape[1]))
        for i in range(self.n_clusters):
            self.means[i] = features[indices[i]]

        counter = 0 
        while not done:
            # Update the labels
            labels, points_in_centroid = self.update_labels(features)
            # Update the means
            self.update_centroid(points_in_centroid)

            # Update labels again
            new_labels, points_in_centroid = self.update_labels(features)

            # Check if the labels have changed
            done = np.allclose(labels, new_labels)
            counter += 1 
            if counter % 1000 == 0: 
                print("Iteration number: ", counter)
            if self.max_iterations and counter >= self.max_iterations: 
                done = True  
            

    # def update_labels(self, samples):

    #     points_in_centroid = {}
    #     labels = np.zeros(samples.shape[0])

    #     for i, sample in enumerate(samples):
    #         # Calculate the distances
    #         centroid_distances = [self._custom_cosine_distance(sample, centroid) for centroid in self.means]
    #         # Choose the centroid which is closest to that sample
    #         centroid_index = np.argmin(centroid_distances)
    #         labels[i] = centroid_index
    #         if centroid_index in points_in_centroid:
    #             points_in_centroid[centroid_index].append(sample)
    #         else:
    #             points_in_centroid[centroid_index] = [sample]

    #     return labels, points_in_centroid

    def update_labels(self, samples):
        centroid_distances = 1 - (np.abs(np.dot(samples, self.means.T)) / 
                                (np.linalg.norm(samples, axis=1)[:, np.newaxis] * 
                                np.linalg.norm(self.means, axis=1)[np.newaxis, :]))

        labels = np.argmin(centroid_distances, axis=1)

        points_in_centroid = {label: samples[labels == label] for label in np.unique(labels)}

        return labels, points_in_centroid

    def update_centroid(self, points_in_centroid):

        for k, v in points_in_centroid.items():
            self.means[k] = np.mean(v, axis=0)


    def _custom_cosine_distance(self, a, b):
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        distance = 1 - (np.abs(dot_product) / (norm_a * norm_b))
        return distance


    def predict(self, features):
        """
        Given features, an np.ndarray of size (n_samples, n_features), predict cluster
        membership labels.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            predictions (np.ndarray): predicted cluster membership for each features,
                of size (n_samples,). Each element of the array is the index of the
                cluster the sample belongs to.
        """

        predictions = np.zeros(features.shape[0], dtype=np.int32)
        for i, sample in enumerate(features):
            # Calculate the distances
            centroid_distances = [self._custom_cosine_distance(sample, centroid) for centroid in self.means]
            # Choose the centroid which is closest to that sample
            centroid_index = np.argmin(centroid_distances)
            predictions[i] = centroid_index

        return predictions