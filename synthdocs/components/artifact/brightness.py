import random

import cv2
import numpy as np
from numba import config
from numba import jit

from synthdocs.components.component import Component

class Brightness(Component):
    """Adjusts the brightness of the whole image by a chosen multiplier.

    :param brightness_range: Pair of ints determining the range from which to sample
           the brightness shift.
    :type brightness_range: tuple, optional
    :param min_brightness: Flag to enable min brightness intensity value in
            the augmented image.
    :type min_brightness: int, optional
    :param min_brightness_value: Pair of ints determining the minimum
            brightness intensity of augmented image.
    :type min_brightness_value: tuple, optional
    :param numba_jit: The flag to enable numba jit to speed up the processing in the augmentation.
    :type numba_jit: int, optional
    :param p: The probability that this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        brightness_range=(0.8, 1.4),
        min_brightness=0,
        min_brightness_value=(20, 50),
        numba_jit=1,
    ):
        """Constructor method"""
        super().__init__()
        self.brightness_range = brightness_range
        self.min_brightness = min_brightness
        self.min_brightness_value = min_brightness_value
        self.numba_jit = numba_jit
        config.DISABLE_JIT = bool(1 - numba_jit)

    @staticmethod
    @jit(nopython=True, cache=True)
    def adjust_min_brightness(image, min_brightness_value):
        """Increase image pixel intensity by value of 10 in each iteration until reaching the min brightness value.

        :param image: The image to apply the function.
        :type image: numpy.array (numpy.uint8)
        :param min_brightness_value: The minimum brightness of value of each pixel.
        :type min_brightness_value: int
        """
        ysize, xsize = image.shape[:2]
        image_flat = image.ravel()

        counting_step = 10.0
        counting_value = counting_step
        while counting_value < min_brightness_value:
            indices = image_flat < counting_value
            image_flat[indices] += counting_step
            counting_value += counting_step

        indices = image_flat > 255
        image_flat[indices] = 255
        image = image_flat.reshape(ysize, xsize)

        return image

    def sample(self, meta=None):
        """Sample random parameters for the brightness effect.
        
        :param meta: Optional metadata dictionary with parameters to use.
        :type meta: dict, optional
        :return: Dictionary with sampled parameters.
        :rtype: dict
        """
        if meta is None:
            meta = {}
            
        meta["run"] = True
        
        # Sample brightness value
        value = random.uniform(self.brightness_range[0], self.brightness_range[1])
        
        # Sample min_brightness flag
        min_brightness = meta.get("min_brightness", self.min_brightness)
        
        # Sample min_brightness_value if needed
        min_brightness_value = None
        if min_brightness:
            min_brightness_value = min(
                255,
                random.randint(self.min_brightness_value[0], self.min_brightness_value[1]),
            )
        
        # Build metadata
        meta.update({
            "value": value,
            "min_brightness": min_brightness,
            "min_brightness_value": min_brightness_value
        })
        
        return meta

    def apply(self, layers, meta=None):
        """Apply the brightness effect to layers.
        
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
        value = meta["value"]
        min_brightness = meta["min_brightness"]
        min_brightness_value = meta["min_brightness_value"]

        if isinstance(layers, np.ndarray):
            layers = [layers]
        
        for layer in layers:
            image = layer.image.copy()
            
            # Check for alpha channel
            has_alpha = 0
            if len(image.shape) > 2:
                is_gray = 0
                if image.shape[2] == 4:
                    has_alpha = 1
                    image, image_alpha = image[:, :, :3], image[:, :, 3]
            else:
                is_gray = 1
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            # Apply brightness adjustment
            hsv = cv2.cvtColor(image.astype("uint8"), cv2.COLOR_BGR2HSV)
            hsv = np.array(hsv, dtype=np.float64)
            hsv[:, :, 2] = hsv[:, :, 2] * value
            hsv[:, :, 2][hsv[:, :, 2] > 255] = 255

            # Increase intensity for areas with intensity below min brightness value
            if min_brightness and min_brightness_value is not None:
                v = self.adjust_min_brightness(hsv[:, :, 2], min_brightness_value)
                hsv[:, :, 2] = v

            hsv = np.array(hsv, dtype=np.uint8)
            image_output = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # Return image in original color format
            if is_gray:
                image_output = cv2.cvtColor(image_output, cv2.COLOR_BGR2GRAY)
            if has_alpha:
                image_output = np.dstack((image_output, image_alpha))

            # Update the layer's image
            layer.image = image_output
            
        return meta

    def __call__(self, image):
        image_copy = image.copy()
            
        # Check for alpha channel
        has_alpha = 0
        if len(image.shape) > 2:
            is_gray = 0
            if image.shape[2] == 4:
                has_alpha = 1
                image_copy, image_alpha = image_copy[:, :, :3], image_copy[:, :, 3]
        else:
            is_gray = 1
            image_copy = cv2.cvtColor(image_copy, cv2.COLOR_GRAY2BGR)
        
        value = random.uniform(self.brightness_range[0], self.brightness_range[1])

        # Apply brightness adjustment
        hsv = cv2.cvtColor(image_copy.astype("uint8"), cv2.COLOR_BGR2HSV)

        hsv = np.array(hsv, dtype=np.float64)
        hsv[:, :, 2] = hsv[:, :, 2] * value
        hsv[:, :, 2][hsv[:, :, 2] > 255] = 255

        # Increase intensity for areas with intensity below min brightness value
        # if min_brightness and min_brightness_value is not None:
        #     v = self.adjust_min_brightness(hsv[:, :, 2], min_brightness_value)
        #     hsv[:, :, 2] = v

        hsv = np.array(hsv, dtype=np.uint8)
        image_output = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Return image in original color format
        if is_gray:
            image_output = cv2.cvtColor(image_output, cv2.COLOR_BGR2GRAY)
        if has_alpha:
            image_output = np.dstack((image_output, image_alpha))

        # Update the layer's image
        return image_output