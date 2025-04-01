"""
version: 0.0.1
*********************************

Dependencies:
- PIL
- opencv
- numpy

*********************************

References:

- Numpy Documentation: https://numpy.org/doc/

- OpenCV Documentation: https://docs.opencv.org/4.x/

- Quasicrystals Inspiration: http://mainisusuallyafunction.blogspot.com/2011/10/quasicrystals-as-sums-of-waves-in-plane.html

*********************************
"""
import math
import random
import warnings

import cv2
import numba as nb
import numpy as np
from numba import config
from numba import jit

from synthdocs.components.component import Component
from synthdocs.utils.slidingwindow import PatternMaker

warnings.filterwarnings("ignore")


class PatternGenerator(Component):
    """In this implementation we take a geometric plane and every point in the plane is shaded according
    to its position,(x,y) coordinate. We take the pattern and perform a bitwise not operation so that it can
    be added as an background to an image.This code is a python implementation of a QuasiPattern Distortion augmentation techniques
    using PIL and the OpenCV libraries. This augmentation creates a new pattern image and superimposes it onto an input image.
    To make the pattern more prominent
    a. Increase the 'frequency' parameter: Increasing the frequency of the pattern will the it tightly populated and more prominent.
    b. Decrease the 'n_rotation' parameter: Decreasing the number of rotations will make the pattern less symmetrical.

    :param imgx: width of the pattern image. default is 512
    :type imgx: int, optional
    :param imgy: height of the pattern image, default is 512
    :type imgy: int, optional
    :param n_rotation: is the number of rotations applied to the pattern, default value lies
                       between 10 and 15.
    :type n_rotation: tuple (int) , optional
    :param color: Color of the pattern in BGR format. Use "random" for random color effect.
    :type color: tuple (int), optional
    :param alpha_range: Tuple of floats determining the alpha value of the patterns.
    :type alpha_range: tuple (float), optional
    :param numba_jit: The flag to enable numba jit to speed up the processing in the augmentation.
    :type numba_jit: int, optional
    :param p: The probability this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        imgx=[256, 512],
        imgy=[256, 512],
        n_rotation_range=(10, 15),
        color="random",
        alpha_range=(0.25, 0.5),
        numba_jit=1,
        p=1.0,
    ):
        """Constructor method"""
        super().__init__()
        self.imgx = random.randint(imgx[0], imgx[1])  # width of the image
        self.imgy = random.randint(imgy[0], imgy[1])  # hieght of the image
        self.n_rotation_range = n_rotation_range  # number of rotation to be applied to the pattern
        self.color = color
        self.alpha_range = alpha_range
        self.numba_jit = numba_jit
        self.p = p
        config.DISABLE_JIT = bool(1 - numba_jit)

    @staticmethod
    @jit(nopython=True, cache=True, parallel=True)
    def apply_augmentation(ndim, pattern_image, frequency, phase, n_rotation):
        """
        Apply the augmentation to generate a pattern image.
        
        :param ndim: Dimensions of the pattern (width, height)
        :param pattern_image: Empty image array to fill with pattern
        :param frequency: Frequency of the pattern
        :param phase: Phase shift of the pattern
        :param n_rotation: Number of rotations to apply
        :return: Generated pattern image
        """
        # Applies the Augmentation to input data.
        width, height = ndim
        # apply transformation, each pixel is transformed to cosine function
        for ky in range(height):
            y = np.float32(ky) / (height - 1) * 4 * math.pi - 2 * math.pi  # normalized coordinates of y-coordinate
            for kx in nb.prange(width):
                x = (
                    np.float32(kx) / (width - 1) * 4 * math.pi - 2 * math.pi
                )  # normalized coordinates of the x-coordinate
                z = 0.0  # z value will determine the intensity of the color, initially set to zero
                for i in nb.prange(n_rotation):
                    r = math.hypot(x, y)  # distance between the point to the origin
                    a = (
                        math.atan2(y, x) + i * math.pi * 2.0 / n_rotation
                    )  # angle the point makes to the origin plus rotation angle
                    z += math.cos(r * math.sin(a) * frequency + phase)  # function of cosine added as an offet
                c = int(round(255 * z / n_rotation))  # color
                pattern_image[ky, kx] = (c, c, c)  # RGB value

        return pattern_image

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
        
        # Sample n_rotation
        n_rotation = meta.get("n_rotation", random.randint(
            self.n_rotation_range[0], 
            self.n_rotation_range[1]
        ))
        
        # Sample frequency and phase
        frequency = meta.get("frequency", random.random() * 100 + 18)  # determines the frequency of pattern
        phase = meta.get("phase", random.random() * math.pi)  # phase shift of the pattern
        
        # Sample alpha value
        alpha = meta.get("alpha", random.uniform(
            self.alpha_range[0], 
            self.alpha_range[1]
        ))
        
        # Sample color
        if self.color == "random":
            color = meta.get("color", (
                random.randint(0, 255), 
                random.randint(0, 255), 
                random.randint(0, 255)
            ))
        else:
            color = meta.get("color", self.color)
        
        # Build metadata
        meta.update({
            "n_rotation": n_rotation,
            "frequency": frequency,
            "phase": phase,
            "alpha": alpha,
            "color": color,
            "imgx": self.imgx,
            "imgy": self.imgy
        })
        
        return meta

    def apply(self, layers, meta=None):
        """Apply the PatternGenerator effect to layers.
        
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
        n_rotation = meta["n_rotation"]
        frequency = meta["frequency"]
        phase = meta["phase"]
        alpha = meta["alpha"]
        color = meta["color"]
        imgx = meta["imgx"]
        imgy = meta["imgy"]
        
        for layer in layers:
            image = layer.image.copy()
            
            # Check for alpha channel
            has_alpha = 0
            if len(image.shape) > 2 and image.shape[2] == 4:
                has_alpha = 1
                image, image_alpha = image[:, :, :3], image[:, :, 3]
            
            h, w = image.shape[:2]
            
            # Generate the pattern
            pattern_image = np.zeros((imgy, imgx, 3), dtype=np.uint8)
            ndim = (imgx, imgy)  # dimensions of pattern
            pattern = self.apply_augmentation(ndim, pattern_image, frequency, phase, n_rotation=n_rotation)
            
            # Process the pattern
            invert = cv2.bitwise_not(pattern)  # performing bitwise not operation
            invert = cv2.resize(invert, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # Adjust pattern format to match image
            if len(image.shape) < 3:
                invert = cv2.cvtColor(invert, cv2.COLOR_RGB2GRAY)
            elif len(image.shape) == 3 and image.shape[2] == 1:
                invert = cv2.cvtColor(invert, cv2.COLOR_RGB2GRAY)
            
            # Create pattern maker for superimposition
            sw = PatternMaker(alpha=alpha)
            
            # Apply color to pattern
            if len(invert.shape) > 2:
                color_mask = np.full_like(invert, fill_value=color, dtype="uint8")
            else:
                color_mask = np.full_like(invert, fill_value=np.mean(color), dtype="uint8")
            
            invert = cv2.multiply(invert, color_mask, scale=1 / 255)
            
            # Overlay pattern onto image
            result = sw.superimpose(image, invert)
            
            # Restore alpha channel if needed
            if has_alpha:
                result = np.dstack((result, image_alpha))
                
            # Update the layer's image
            layer.image = result
            
        return meta