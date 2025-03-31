import random

import cv2
import numpy as np
from sklearn.datasets import make_blobs

from synthdocs.components.component import Component


class Letterpress(Component):
    """Produces regions of ink mimicking the effect of ink pressed unevenly onto paper.

    :param n_samples: Pair of ints determining number of points in a cluster.
    :type n_samples: tuple, optional
    :param n_clusters: Pair of ints determining number of clusters.
    :type n_clusters: tuple, optional
    :param std_range: Pair of ints determining the range from which the
           standard deviation of the blob distribution is sampled.
    :type std_range: tuple, optional
    :param value_range: Pair of ints determining the range from which the
           value of a point in the blob is sampled.
    :type value_range: tuple, optional
    :param value_threshold_range: Min value of pixel to enable letterpress effect.
    :type value_threshold_range: tuple, optional
    :param blur: Flag to enable blur in letterpress noise mask.
    :type blur: int, optional
    """

    def __init__(
        self,
        n_samples=(300, 800),
        n_clusters=(300, 800),
        std_range=(1500, 5000),
        value_range=(200, 255),
        value_threshold_range=(128, 128),
        blur=1,
    ):
        """Constructor method"""
        super().__init__()
        self.n_samples = n_samples
        self.n_clusters = n_clusters
        self.std_range = std_range
        self.value_range = value_range
        self.value_threshold_range = value_threshold_range
        self.blur = blur

    def sample(self, meta=None):
        if meta is None:
            meta = {}
        
        # Sample number of iterations for blob generation
        num_iterations = meta.get("num_iterations", random.randint(8, 12))
        
        # Sample value threshold
        value_threshold = meta.get("value_threshold", None)
        if value_threshold is None:
            if self.value_threshold_range[1] >= self.value_threshold_range[0]:
                value_threshold = random.randint(self.value_threshold_range[0], self.value_threshold_range[1])
            else:
                value_threshold = self.value_threshold_range[1]
        
        # Build metadata
        meta = {
            "num_iterations": num_iterations,
            "value_threshold": value_threshold,
            "n_samples": self.n_samples,
            "n_clusters": self.n_clusters,
            "std_range": self.std_range,
            "value_range": self.value_range,
            "blur": self.blur
        }
        
        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        
        num_iterations = meta["num_iterations"]
        value_threshold = meta["value_threshold"]
        n_samples = meta["n_samples"]
        n_clusters = meta["n_clusters"]
        std_range = meta["std_range"]
        value_range = meta["value_range"]
        blur = meta["blur"]
        
        for layer in layers:
            image = layer.image.copy().astype(np.uint8)
            
            ysize, xsize = image.shape[:2]
            max_box_size = max(ysize, xsize)

            generated_points = np.array([[-1, -1]], dtype="float")

            for i in range(num_iterations):
                # Sample number of samples per cluster
                n_samples_per_cluster = [
                    random.randint(n_samples[0], n_samples[1])
                    for _ in range(random.randint(n_clusters[0], n_clusters[1]))
                ]
                
                # Sample standard deviation
                std = random.randint(std_range[0], std_range[1]) / 100

                # Generate clusters of blobs
                generated_points_new, point_group = make_blobs(
                    n_samples=n_samples_per_cluster,
                    center_box=(0, max_box_size),
                    cluster_std=std,
                    n_features=2,
                )

                generated_points = np.concatenate((generated_points, generated_points_new), axis=0)

            # Remove decimals
            generated_points = generated_points.astype("int")

            # Delete location where < 0 and > image size
            ind_delete = np.logical_or.reduce(
                (
                    generated_points[:, 0] < 0,
                    generated_points[:, 1] < 0,
                    generated_points[:, 0] > xsize - 1,
                    generated_points[:, 1] > ysize - 1,
                ),
            )
            generated_points_x = np.delete(generated_points[:, 0], ind_delete.reshape(ind_delete.shape[0]), axis=0)
            generated_points_y = np.delete(generated_points[:, 1], ind_delete.reshape(ind_delete.shape[0]), axis=0)

            # Initialize empty noise mask and noise mask with random values
            noise_mask = np.copy(image)
            noise_mask2 = np.random.randint(
                value_range[0],
                value_range[1],
                size=(image.shape[0], image.shape[1]),
                dtype="uint8",
            )

            # Insert noise value according to generate points
            if len(image.shape) > 2:
                # Skip alpha layer
                for i in range(min(3, image.shape[2])):
                    noise_mask[generated_points_y, generated_points_x, i] = noise_mask2[
                        generated_points_y,
                        generated_points_x,
                    ]
            else:
                noise_mask[generated_points_y, generated_points_x] = noise_mask2[generated_points_y, generated_points_x]

            if blur:
                # Gaussian blur needs uint8 input
                noise_mask = cv2.GaussianBlur(noise_mask, (5, 5), 0)

            # Apply noise to image
            indices = image < value_threshold
            image[indices] = noise_mask[indices]
            
            # Update layer's image
            layer.image = image
            
        return meta