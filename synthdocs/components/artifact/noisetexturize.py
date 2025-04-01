import random

import cv2
import numpy as np

from synthdocs.components.component import Component


class NoiseTexturize(Component):
    """Creates a random noise pattern to emulate paper textures.
    Consequently applies noise patterns to the original image from big to small.

    :param sigma_range: Defines bounds of noise fluctuations.
    :type sigma_range: tuple, optional
    :param turbulence_range: Defines how quickly big patterns will be
        replaced with the small ones. The lower value -
        the more iterations will be performed during texture generation.
    :type turbulence_range: tuple, optional
    :param texture_width_range: Tuple of ints determining the width of the texture image.
        If the value is higher, the texture will be more refined.
    :type texture_width_range: tuple, optional
    :param texture_height_range: Tuple of ints determining the height of the texture.
        If the value is higher, the texture will be more refined.
    :type texture_height_range: tuple, optional
    :param p: The probability this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        sigma_range=(3, 10),
        turbulence_range=(2, 5),
        texture_width_range=(100, 500),
        texture_height_range=(100, 500),
        p=1,
    ):
        """Constructor method"""
        super().__init__()
        self.sigma_range = sigma_range
        self.turbulence_range = turbulence_range
        self.texture_width_range = texture_width_range
        self.texture_height_range = texture_height_range
        self.p = p

    def noise(self, width, height, channel, ratio, sigma, texture_width_range, texture_height_range):
        """The function generates an image, filled with gaussian nose. If ratio
        parameter is specified, noise will be generated for a lesser image and
        then it will be upscaled to the original size. In that case noise will
        generate larger square patterns. To avoid multiple lines, the upscale
        uses interpolation.

        :param width: Width of generated image.
        :type width: int
        :param height: Height of generated image.
        :type height: int
        :param channel: Channel number of generated image.
        :type channel: int
        :param ratio: The size of generated noise "pixels".
        :type ratio: int
        :param sigma: Defines bounds of noise fluctuations.
        :type sigma: int
        :param texture_width_range: Range for texture width.
        :type texture_width_range: tuple
        :param texture_height_range: Range for texture height.
        :type texture_height_range: tuple
        :return: Noise texture image
        :rtype: numpy.ndarray
        """
        ysize = random.randint(texture_height_range[0], texture_height_range[1])
        xsize = random.randint(texture_width_range[0], texture_width_range[1])

        result = np.random.normal(0, sigma, size=(ysize, xsize))

        result = cv2.resize(
            result,
            dsize=(width, height),
            interpolation=cv2.INTER_LINEAR,
        )
        if channel:
            result = np.stack([result, result, result], axis=2)

        return result

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
        
        # Sample sigma
        sigma = meta.get("sigma", random.randint(self.sigma_range[0], self.sigma_range[1]))
        
        # Sample turbulence (must be > 1 to prevent endless loop)
        turbulence = meta.get("turbulence", max(
            2,
            random.randint(
                self.turbulence_range[0],
                self.turbulence_range[1],
            ),
        ))
        
        # Build metadata
        meta.update({
            "sigma": sigma,
            "turbulence": turbulence,
            "texture_width_range": self.texture_width_range,
            "texture_height_range": self.texture_height_range
        })
        
        return meta

    def apply(self, layers, meta=None):
        """Apply the NoiseTexturize effect to layers.
        
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
        sigma = meta["sigma"]
        turbulence = meta["turbulence"]
        texture_width_range = meta["texture_width_range"]
        texture_height_range = meta["texture_height_range"]
        
        for layer in layers:
            image = layer.image.copy()
            
            # Check for alpha channel
            has_alpha = 0
            if len(image.shape) > 2 and image.shape[2] == 4:
                has_alpha = 1
                image, image_alpha = image[:, :, :3], image[:, :, 3]
            
            result = image.astype(float)
            rows, cols = image.shape[:2]
            
            # Determine channel
            if len(image.shape) > 2:
                channel = image.shape[2]
            else:
                channel = 0
            
            # Apply noise at different scales
            ratio = cols
            while not ratio == 1:
                result += self.noise(cols, rows, channel, ratio, sigma, texture_width_range, texture_height_range)
                ratio = (ratio // turbulence) or 1
            
            # Clip values and convert to uint8
            cut = np.clip(result, 0, 255)
            cut = cut.astype(np.uint8)
            
            # Restore alpha channel if needed
            if has_alpha:
                cut = np.dstack((cut, image_alpha))
            
            # Update the layer's image
            layer.image = cut
            
        return meta