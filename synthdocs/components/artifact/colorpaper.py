import random
import cv2
import numpy as np

from synthdocs.components.component import Component


class ColorPaper(Component):
    """Change color of input paper based on user input hue and saturation.

    :param hue_range: Pair of ints determining the range from which
           hue value is sampled.
    :type hue_range: tuple, optional
    :param saturation_range: Pair of ints determining the range from which
           saturation value is sampled.
    :param p: The probability that this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        hue_range=(28, 45),
        saturation_range=(10, 40),
        p=1,
    ):
        """Constructor method"""
        super().__init__()
        self.hue_range = hue_range
        self.saturation_range = saturation_range
        self.p = p

    def add_color(self, image):
        """Add color background into input image.

        :param image: The image to apply the function.
        :type image: numpy.array (numpy.uint8)
        """

        has_alpha = 0
        if len(image.shape) > 2:
            is_gray = 0
            if image.shape[2] == 4:
                has_alpha = 1
                image, image_alpha = image[:, :, :3], image[:, :, 3]
        else:
            is_gray = 1
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        ysize, xsize = image.shape[:2]

        # convert to hsv colorspace
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        random_hue = np.random.randint(self.hue_range[0], self.hue_range[1] + 1)
        random_saturation = np.random.randint(self.saturation_range[0], self.saturation_range[1] + 1)

        # assign hue and saturation
        image_h = np.random.randint(max(0, random_hue - 5), min(255, random_hue + 5), size=(ysize, xsize))
        image_s = np.random.randint(max(0, random_saturation - 5), min(255, random_saturation + 5), size=(ysize, xsize))

        # assign hue and saturation channel back to hsv image
        image_hsv[:, :, 0] = image_h
        image_hsv[:, :, 1] = image_s

        # convert back to bgr
        image_color = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

        # return image follows the input image color channel
        if is_gray:
            image_color = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
        if has_alpha:
            image_color = np.dstack((image_color, image_alpha))

        return image_color

    def sample(self, meta=None):
        """Sample random parameters for the augmentation.
        
        :param meta: Optional metadata dictionary with parameters to use.
        :type meta: dict, optional
        :return: Dictionary with sampled parameters.
        :rtype: dict
        """
        if meta is None:
            meta = {}
        
        # Check if we should run this augmentation based on probability
        if random.random() > self.p:
            meta["run"] = False
            return meta
            
        meta["run"] = True
        
        # Sample hue if not provided
        hue = meta.get("hue", None)
        if hue is None:
            hue = np.random.randint(self.hue_range[0], self.hue_range[1] + 1)
        
        # Sample saturation if not provided
        saturation = meta.get("saturation", None)
        if saturation is None:
            saturation = np.random.randint(self.saturation_range[0], self.saturation_range[1] + 1)
        
        # Build metadata
        meta.update({
            "hue": hue,
            "saturation": saturation
        })
        
        return meta

    def apply(self, layers, meta=None):
        """Apply the ColorPaper effect to layers.
        
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
                
            ysize, xsize = image.shape[:2]
            
            # convert to hsv colorspace
            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Get sampled parameters from metadata
            hue = meta["hue"]
            saturation = meta["saturation"]
            
            # assign hue and saturation
            image_h = np.random.randint(max(0, hue - 5), min(255, hue + 5), size=(ysize, xsize))
            image_s = np.random.randint(max(0, saturation - 5), min(255, saturation + 5), size=(ysize, xsize))
            
            # assign hue and saturation channel back to hsv image
            image_hsv[:, :, 0] = image_h
            image_hsv[:, :, 1] = image_s
            
            # convert back to bgr
            image_color = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
            
            # return image follows the input image color channel
            if is_gray:
                image_color = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
            if has_alpha:
                image_color = np.dstack((image_color, image_alpha))
                
            # Update the layer's image
            layer.image = image_color
            
        return meta