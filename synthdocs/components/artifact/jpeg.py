import random

import cv2
import numpy as np

from synthdocs.components.component import Component


class Jpeg(Component):
    """Uses JPEG encoding to create compression artifacts in the image.

    :param quality_range: Pair of ints determining the range from which to
           sample the compression quality.
    :type quality_range: tuple, optional
    :param p: The probability that this effect will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        quality_range=(25, 95),
        p=1,
    ):
        """Constructor method"""
        super().__init__()
        self.quality_range = quality_range
        self.p = p

    def sample(self, meta=None):
        """Sample random parameters for the JPEG compression effect.
        
        :param meta: Optional metadata dictionary with parameters to use.
        :type meta: dict, optional
        :return: Dictionary with sampled parameters.
        :rtype: dict
        """
        if meta is None:
            meta = {}
            
        # Check if we should run based on probability
        if random.random() > self.p:
            meta["run"] = False
            return meta
            
        meta["run"] = True
        
        # Sample quality value
        quality = random.randint(self.quality_range[0], self.quality_range[1])
        
        # Build metadata
        meta["quality"] = quality
        
        return meta

    def apply(self, layers, meta=None):
        """Apply the JPEG compression effect to layers.
        
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
        quality = meta["quality"]
        
        # Set up encoding parameters
        encode_param = [
            int(cv2.IMWRITE_JPEG_QUALITY),
            quality,
        ]
        
        for layer in layers:
            image = layer.image.copy()
            
            has_alpha = 0
            if len(image.shape) > 2 and image.shape[2] == 4:
                has_alpha = 1
                image, image_alpha = image[:, :, :3], image[:, :, 3]
            
            # Apply JPEG compression
            result, encimg = cv2.imencode(".jpg", image, encode_param)
            compressed_image = cv2.imdecode(encimg, 1)
            
            # Restore alpha channel if needed
            if has_alpha:
                compressed_image = np.dstack((compressed_image, image_alpha))
                
            # Update the layer's image
            layer.image = compressed_image
            
        return meta