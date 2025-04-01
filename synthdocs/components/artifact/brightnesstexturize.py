import random

import cv2
import numpy as np

from synthdocs.components.component import Component


class BrightnessTexturize(Component):
    """Creates a random noise in the brightness channel to emulate paper
    textures.

    :param texturize_range: Pair of floats determining the range from which to sample values
           for the brightness matrix. Suggested value = <1.
    :type brightness_range: tuple, optional
    :param deviation: Additional variation for the uniform sample.
    :type deviation: float, optional
    :param p: The probability that this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(self, texturize_range=(0.8, 0.99), deviation=0.08, p=1):
        """Constructor method"""
        super().__init__()
        self.low = texturize_range[0]
        self.high = texturize_range[1]
        self.deviation = deviation
        self.texturize_range = texturize_range
        self.p = p

    def sample(self, meta=None):
        """Sample random parameters for the augmentation.
        
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
        
        # Sample the value from texturize_range
        value = meta.get("value", random.uniform(self.low, self.high))
        
        # Calculate low and max values based on deviation
        low_value1 = meta.get("low_value1", value - (value * self.deviation))
        max_value1 = meta.get("max_value1", value + (value * self.deviation))
        
        low_value2 = meta.get("low_value2", value - (value * self.deviation))
        max_value2 = meta.get("max_value2", value + (value * self.deviation))
        
        # Build metadata
        meta.update({
            "value": value,
            "low_value1": low_value1,
            "max_value1": max_value1,
            "low_value2": low_value2,
            "max_value2": max_value2,
            "deviation": self.deviation
        })
        
        return meta

    def apply(self, layers, meta=None):
        """Apply the BrightnessTexturize effect to layers.
        
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
        low_value1 = meta["low_value1"]
        max_value1 = meta["max_value1"]
        low_value2 = meta["low_value2"]
        max_value2 = meta["max_value2"]
        
        for layer in layers:
            image = layer.image.copy()
            # Check for channel information
            has_alpha = 0
            if len(image.shape) > 2:
                is_gray = 0
                if image.shape[2] == 4:
                    has_alpha = 1
                    image, image_alpha = image[:, :, :3], image[:, :, 3]
            else:
                is_gray = 1
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            # Convert to HSV color space
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Convert to float
            hsv = np.array(hsv, dtype=np.float64)

            # First noise application
            brightness_matrix = np.random.uniform(low_value1, max_value1, size=(hsv.shape[0], hsv.shape[1]))
            hsv[:, :, 1] *= brightness_matrix
            hsv[:, :, 2] *= brightness_matrix
            hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
            hsv[:, :, 2][hsv[:, :, 2] > 255] = 255

            # Convert back to uint8, apply bitwise not and convert to hsv again
            hsv = np.array(hsv, dtype=np.uint8)

            # Non hue and saturation channel to prevent color change
            hsv[:, :, 2] = cv2.bitwise_not(hsv[:, :, 2])
            hsv = hsv.astype("float64")

            # Second noise application
            brightness_matrix = np.random.uniform(low_value2, max_value2, size=(hsv.shape[0], hsv.shape[1]))
            hsv[:, :, 1] *= brightness_matrix
            hsv[:, :, 2] *= brightness_matrix
            hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
            hsv[:, :, 2][hsv[:, :, 2] > 255] = 255

            # Convert back to uint8, apply bitwise not
            hsv = np.array(hsv, dtype=np.uint8)
            # Non hue and saturation channel to prevent color change
            hsv[:, :, 2] = cv2.bitwise_not(hsv[:, :, 2])

            # Convert back to original color space
            image_output = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # Return image follows the input image color channel
            if is_gray:
                image_output = cv2.cvtColor(image_output, cv2.COLOR_BGR2GRAY)
            if has_alpha:
                image_output = np.dstack((image_output, image_alpha))
                
            # Update the layer's image
            layer.image = image_output
            
        return meta