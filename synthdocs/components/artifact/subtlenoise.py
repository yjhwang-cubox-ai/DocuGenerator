import random

import numpy as np

from synthdocs.components.component import Component


class SubtleNoise(Component):
    """Emulates the imperfections in scanning solid colors due to subtle
    lighting differences.

    :param subtle_range: The possible range of noise variation to sample from.
    :type subtle_range: int, optional
    :param p: The probability that this effect will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        subtle_range=(0,10)
    ):
        """Constructor method"""
        super().__init__()
        self.subtle_range = random.randint(subtle_range[0], subtle_range[1])

    def add_subtle_noise(self, image, subtle_range):
        """Generate mask of noise and add it to input image.

        :param image: Image to apply the function.
        :type image: numpy.array
        :param subtle_range: Range of noise variation.
        :type subtle_range: int
        :return: Image with added noise.
        :rtype: numpy.ndarray
        """
        # get image size
        ysize, xsize = image.shape[:2]

        # generate 2d mask of random noise
        image_noise = np.random.randint(-subtle_range, subtle_range, size=(ysize, xsize))

        # add noise to image
        image = image.astype("int") + image_noise

        return image

    def sample(self, meta=None):
        """Sample random parameters for the subtle noise effect.
        
        :param meta: Optional metadata dictionary with parameters to use.
        :type meta: dict, optional
        :return: Dictionary with sampled parameters.
        :rtype: dict
        """
        if meta is None:
            meta = {}
            
        meta["run"] = True
        
        # Build metadata
        meta["subtle_range"] = self.subtle_range
        
        return meta

    def apply(self, layers, meta=None):
        """Apply the subtle noise effect to layers.
        
        :param layers: The layers to apply the effect to.
        :type layers: list of Layer objects
        :param meta: Optional metadata with parameters.
        :type meta: dict, optional
        :return: Updated metadata.
        :rtype: dict
        """
        meta = self.sample(meta)
        
        # Skip processing if run is False
        if not meta.get("run", True):
            return meta
            
        # Get parameters from metadata
        subtle_range = meta["subtle_range"]
        
        for layer in layers:
            image = layer.image.copy()
            
            # multiple channels image
            if len(image.shape) > 2:
                # convert to int to enable negative
                image = image.astype("int")
                # skip alpha layer if it exists
                channels = 3 if image.shape[2] >= 3 else image.shape[2]
                for i in range(channels):
                    image[:, :, i] = self.add_subtle_noise(image[:, :, i], subtle_range)
                
                # Handle alpha channel separately if it exists
                if image.shape[2] > 3:
                    alpha_channel = image[:, :, 3].copy()
            # single channel image
            else:
                image = self.add_subtle_noise(image, subtle_range)
            
            # clip values between 0-255
            image = np.clip(image, 0, 255).astype(np.uint8)
            
            # Add back alpha channel if it existed
            if len(layer.image.shape) > 2 and layer.image.shape[2] > 3:
                if image.shape[2] < 4:  # If alpha was not included in processing
                    image = np.dstack((image, alpha_channel))
                else:  # Ensure alpha is unchanged
                    image[:, :, 3] = layer.image[:, :, 3]
                
            # Update the layer's image
            layer.image = image
            
        return meta