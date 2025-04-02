import os
import random

import cv2
import numpy as np

from synthdocs.components.component import Component


class Gamma(Component):
    """Adjusts the gamma of the whole image by a chosen multiplier.

    :param gamma_range: Pair of ints determining the range from which to sample the
           gamma shift.
    :type gamma_range: tuple, optional
    :param p: The probability that this effect will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        gamma_range=(0.5, 1.5),
        p=1,
    ):
        """Constructor method"""
        super().__init__()
        self.gamma_range = gamma_range
        self.p = p

    def sample(self, meta=None):
        """Sample random parameters for the gamma adjustment.
        
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
        
        # Sample gamma value
        gamma_value = random.uniform(self.gamma_range[0], self.gamma_range[1])
        
        # Build metadata
        meta["gamma_value"] = gamma_value
        
        return meta

    def apply(self, layers, meta=None):
        """Apply the gamma adjustment to layers.
        
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
        gamma_value = meta["gamma_value"]
        
        # Create gamma correction lookup table
        invGamma = 1.0 / gamma_value
        table = np.array(
            [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)],
        ).astype("uint8")
        
        for layer in layers:
            image = layer.image.copy()
            image = image.astype(np.uint8)
            
            # Apply gamma correction using lookup table
            adjusted_image = cv2.LUT(image, table)
                
            # Update the layer's image
            layer.image = adjusted_image
            
        return meta