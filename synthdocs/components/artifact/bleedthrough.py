import os
import random
from glob import glob

import cv2
import numpy as np

from synthdocs.components.component import Component
from synthdocs.utils import OverlayBuilder
from synthdocs.utils.lib import add_noise, generate_average_intensity

class BleedThrough(Component):
    """Emulates bleed through effect from the combination of ink bleed and
    gaussian blur operations.

    :param intensity_range: Pair of floats determining the range from which
           noise intensity is sampled.
    :type intensity: tuple, optional
    :param color_range: Pair of ints determining the range from which color
           noise is sampled.
    :type color_range: tuple, optional
    :param ksize: Tuple of height/width pairs from which to sample the kernel
           size. Higher value increases the spreadness of bleeding effect.
    :type ksizes: tuple, optional
    :param sigmaX: Standard deviation of the kernel along the x-axis.
    :type sigmaX: float, optional
    :param alpha: Intensity of bleeding effect, recommended value range from
            0.1 to 0.5.
    :type alpha: float, optional
    :param offsets: Tuple of x and y offset pair to shift the bleed through
            effect from original input.
    :type offsets: tuple, optional
    """

    def __init__(
        self,
        intensity_range=(0.1, 0.9),
        color_range=(0, 224),
        ksize=(17, 17),
        sigmaX=1,
        alpha=0.2,
        offsets=(20, 20),
    ):
        super().__init__()
        self.intensity_range = intensity_range
        self.color_range = color_range
        self.ksize = ksize
        self.sigmaX = sigmaX
        self.alpha = alpha
        self.offsets = offsets

    def blend(self, img, img_bleed, alpha):
        """Blend two images based on the alpha value to create bleedthrough effect.

        :param img: The background image to apply the blending function.
        :type img: numpy.array (numpy.uint8)
        :param img_bleed: The foreground image to apply the blending function.
        :type img_bleed: numpy.array (numpy.uint8)
        :param alpha: The alpha value of foreground image for the blending function.
        :type alpha: float
        """

        # convert to single channel to avoid unnecessary noise in colour image
        if len(img_bleed.shape) > 2:
            img_bleed_input = cv2.cvtColor(
                img_bleed.astype("uint8"),
                cv2.COLOR_BGR2GRAY,
            )
        else:
            img_bleed_input = img_bleed.astype("uint8")

        # if the bleedthrough foreground is darker, reduce the blending alpha value
        img_bleed_brightness = generate_average_intensity(img_bleed)
        img_brightness = generate_average_intensity(img)
        if img_bleed_brightness < img_brightness:
            new_alpha = alpha * (img_bleed_brightness / img_brightness) / 2
        else:
            new_alpha = alpha

        ob = OverlayBuilder(
            "normal",
            img_bleed_input,
            img,
            1,
            (1, 1),
            "center",
            0,
            new_alpha,
        )
        return ob.build_overlay()

    def generate_offset(self, img_bleed, offsets):
        """Offset image based on the input offset value so that bleedthrough effect is visible and not stacked with background image.

        :param img_bleed: The input image to apply the offset function.
        :type img_bleed: numpy.array (numpy.uint8)
        :param offsets: The offset value.
        :type offsets: int
        """

        x_offset = offsets[0]
        y_offset = offsets[1]
        result = img_bleed.copy()
        
        if (x_offset == 0) and (y_offset == 0):
            return result
        elif x_offset == 0:
            result[y_offset:, :] = img_bleed[:-y_offset, :]
        elif y_offset == 0:
            result[:, x_offset:] = img_bleed[:, :-x_offset]
        else:
            result[y_offset:, x_offset:] = img_bleed[:-y_offset, :-x_offset]
        return result

    def generate_bleeding_ink(self, img, intensity_range, color_range, ksize, sigmaX):
        """Preprocess and create bleeding ink effect in the input image.

        :param img: The input image to apply the offset function.
        :type img: numpy.array (numpy.uint8)
        :param intensity_range: Pair of floats determining the range from which noise intensity is sampled.
        :type intensity_range: tuple
        :param color_range: Pair of ints determining the range from which color noise is sampled.
        :type color_range: tuple
        :param ksize: Tuple of height/width pairs from which to sample the kernel size. Higher value increases the spreadness of bleeding effect.
        :type ksize: tuple
        :param sigmaX: Standard deviation of the kernel along the x-axis.
        :type sigmaX: float
        """

        img_noise = np.double(
            add_noise(img, intensity_range=intensity_range, color_range=color_range, noise_condition=1),
        )
        img_bleed = cv2.GaussianBlur(img_noise, ksize=ksize, sigmaX=sigmaX)

        return img_bleed

    # create foreground image for bleedthrough effect
    def create_bleedthrough_foreground(self, image: np.ndarray):
        """Create foreground image for bleedthrough effect.

        :param image: The background image of the bleedthrough effect.
        :type image: numpy.array (numpy.uint8)
        """
        # For now, just use the image itself as foreground (could be updated to use cached images)
        image_bleedthrough_foreground = image.copy()
        
        # flip left-right only, flip top-bottom get inverted text, which is not realistic
        image_bleedthrough_foreground = cv2.flip(image_bleedthrough_foreground, 1)

        return image_bleedthrough_foreground

    def sample(self, meta=None):
        if meta is None:
            meta = {}
        
        # Sample intensity or use provided value
        intensity_range = meta.get("intensity_range", self.intensity_range)
        
        # Sample color_range or use provided value
        color_range = meta.get("color_range", self.color_range)
        
        # Sample alpha or use provided value
        alpha = meta.get("alpha", self.alpha)
        
        # Sample offsets or use provided value
        offsets = meta.get("offsets", self.offsets)
        
        # Sample ksize or use provided value
        ksize = meta.get("ksize", self.ksize)
        
        # Sample sigmaX or use provided value
        sigmaX = meta.get("sigmaX", self.sigmaX)
        
        # Build metadata
        meta = {
            "intensity_range": intensity_range,
            "color_range": color_range,
            "alpha": alpha,
            "offsets": offsets,
            "ksize": ksize,
            "sigmaX": sigmaX
        }
        
        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        
        intensity_range = meta["intensity_range"]
        color_range = meta["color_range"]
        alpha = random.uniform(meta["alpha"][0], meta["alpha"][1])
        offsets = meta["offsets"]
        ksize = meta["ksize"]
        sigmaX = meta["sigmaX"]
        
        for layer in layers:
            image = layer.image.copy().astype(np.uint8)
            
            # check for alpha layer
            has_alpha = 0
            if len(image.shape) > 2:
                if image.shape[2] == 4:
                    has_alpha = 1
                    image, image_alpha = image[:, :, :3], image[:, :, 3]

            image_bleedthrough_foreground = self.create_bleedthrough_foreground(image)

            image_bleed = self.generate_bleeding_ink(
                image_bleedthrough_foreground,
                intensity_range,
                color_range,
                ksize,
                sigmaX,
            )
            
            image_bleed_offset = self.generate_offset(image_bleed, offsets)
            image_bleedthrough = self.blend(image, image_bleed_offset, alpha)

            if has_alpha:
                image_bleedthrough = np.dstack((image_bleedthrough, image_alpha))
                
            # Update layer's image
            layer.image = image_bleedthrough
            
        return meta
