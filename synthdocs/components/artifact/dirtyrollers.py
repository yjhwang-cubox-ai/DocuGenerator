import random

import cv2
import numpy as np
from numba import config
from numba import jit

from synthdocs.components.component import Component


class DirtyRollers(Component):
    """Emulates an effect created by certain document scanners.

    :param line_width_range: Pair of ints determining the range from which the
    width of a dirty roller line is sampled.
    :type line_width_range: tuple, optional
    :param scanline_type: Types of scanline, use 0 for white background.
    :type scanline_type: int, optional
    :param numba_jit: The flag to enable numba jit to speed up the processing.
    :type numba_jit: int, optional
    :param p: The probability this effect will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        line_width_range=(8, 12),
        scanline_type=0,
        numba_jit=1,
        p=1,
    ):
        """Constructor method"""
        super().__init__()
        self.line_width_range = line_width_range
        self.scanline_type = scanline_type
        self.numba_jit = numba_jit
        self.p = p
        config.DISABLE_JIT = bool(1 - numba_jit)

    def apply_scanline_mask(self, img, mask, meta_mask, scanline_type):
        """Main function to apply scanline mask to input image.
        
        :param img: The image to apply the function.
        :type img: numpy.array (numpy.uint8)
        :param mask: Mask of scanline effect.
        :type mask: numpy.array (numpy.uint8)
        :param meta_mask: Meta mask of scanline effect.
        :type meta_mask: numpy.array (numpy.uint8)
        :param scanline_type: Type of scanline effect.
        :type scanline_type: int
        :return: Processed image
        :rtype: numpy.ndarray
        """
        # for dark background
        if scanline_type:
            return self.apply_scanline_mask_v2(img, mask, meta_mask)
        # for white background
        else:
            return self.apply_scanline_mask_v1(img, mask, meta_mask)

    def apply_scanline_mask_v2(self, img, mask, meta_mask):
        """Function to apply scanline mask to input image with dark background."""
        mask = self.apply_scanline_metamask_v2(mask, meta_mask)

        for i in range(3):
            img_channel = img[:, :, i].astype("int")
            new_image = np.add(img_channel, np.multiply(img_channel, (1 - (mask / 100))))
            new_image[new_image > 255] = 255
            img[:, :, i] = new_image
        return img

    def apply_scanline_metamask_v2(self, img, mask):
        """Function to apply scanline meta mask to scanline mask of dark background."""
        new_mask = np.subtract(img, np.multiply(img, (1 - (mask / 100))))
        new_mask[new_mask < 0] = 0
        return new_mask

    def apply_scanline_mask_v1(self, img, mask, meta_mask):
        """Function to apply scanline mask to input image with white background."""
        mask = self.apply_scanline_metamask_v1(mask, meta_mask)

        for i in range(3):
            img_channel = img[:, :, i].astype("int")
            new_image = np.subtract(img_channel, np.multiply(img_channel, (1 - (mask / 100))))
            new_image[new_image < 0] = 0
            img[:, :, i] = new_image.astype("uint8")
        return img

    def apply_scanline_metamask_v1(self, img, mask):
        """Function to apply scanline meta mask to scanline mask of white background."""
        new_mask = np.add(img, np.multiply(img, (1 - (mask / 100))))
        new_mask[new_mask > 99] = 99
        return new_mask

    @staticmethod
    @jit(nopython=True, cache=True)
    def create_scanline_mask(width, height, line_width):
        """Function to create scanline mask."""
        # Create Standard Bar
        grad_high_pct = random.randint(86, 99)
        grad_low_pct = random.randint(70, 85)
        grad_grid1 = np.linspace(grad_high_pct, grad_low_pct, line_width)
        grad_grid1_stack = np.hstack((grad_grid1, np.flip(grad_grid1)))
        grad_grid1 = grad_grid1_stack.repeat(height).reshape((-1, height)).T

        # Create Standard Bar with Wide Dark
        grad_center2 = np.full((random.randint(1, 6)), grad_low_pct)
        grad_grid2 = np.linspace(grad_high_pct, grad_low_pct, line_width)
        grad_grid2_stack = np.hstack((grad_grid2, grad_center2, np.flip(grad_grid2)))
        grad_grid2 = grad_grid2_stack.repeat(height).reshape((-1, height)).T

        # Create Standard Bar with Wide Light
        grad_exterior3 = np.full((random.randint(1, 6)), grad_high_pct)
        grad_grid3 = np.linspace(grad_high_pct, grad_low_pct, line_width)
        grad_grid3_stack = np.hstack((grad_grid3, np.flip(grad_grid3), grad_exterior3))
        grad_grid3 = grad_grid3_stack.repeat(height).reshape((-1, height)).T

        # Create Standard Bar with Lower Dark
        grad_high_pct += min(100, random.randint(-3, 3))
        grad_low_pct -= random.randint(5, 8)
        grad_grid4 = np.linspace(grad_high_pct, grad_low_pct, line_width)
        grad_grid4_stack = np.hstack((grad_grid4, np.flip(grad_grid4)))
        grad_grid4 = grad_grid4_stack.repeat(height).reshape((-1, height)).T

        # Create Standard Bar with Low Dark and Wide Dark
        grad_center5 = np.full((random.randint(1, 6)), grad_low_pct)
        grad_grid5 = np.linspace(grad_high_pct, grad_low_pct, line_width)
        grad_grid5_stack = np.hstack((grad_grid5, grad_center5, np.flip(grad_grid5)))
        grad_grid5 = grad_grid5_stack.repeat(height).reshape((-1, height)).T

        # Create Standard Bar with Low Dark Wide Light
        grad_exterior6 = np.full((random.randint(1, 6)), grad_high_pct)
        grad_grid6 = np.linspace(grad_high_pct, grad_low_pct, line_width)
        grad_grid6_stack = np.hstack((grad_grid6, np.flip(grad_grid6), grad_exterior6))
        grad_grid6 = grad_grid6_stack.repeat(height).reshape((-1, height)).T

        # Combine all grad_grid in a list
        grad_list = [grad_grid1, grad_grid2, grad_grid3, grad_grid4, grad_grid5, grad_grid6]

        # Fill mask with random grad_grid
        mask = np.zeros((height, width), dtype="float")
        xcounter = 0
        while True:
            random_index = random.randint(0, 5)
            current_grad = grad_list[random_index]
            ysize, xsize = current_grad.shape[:2]
            if xcounter + xsize < width:
                mask[:, xcounter : xcounter + xsize] = current_grad
                xcounter += xsize
            else:
                mask[:, xcounter:] = current_grad[:, : width - xcounter]
                break

        return mask

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
        
        # Sample parameters
        meta.update({
            "line_width": random.randint(
                self.line_width_range[0],
                self.line_width_range[1],
            ),
            "rotate": random.choice([True, False]),
            "scanline_type": self.scanline_type
        })
        
        return meta

    def apply(self, layers, meta=None):
        """Apply the DirtyRollers effect to layers.
        
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
        line_width = meta["line_width"]
        rotate = meta["rotate"]
        scanline_type = meta["scanline_type"]
        
        for layer in layers:
            image = layer.image.copy()
            
            # Check and convert image into BGR format
            if len(image.shape) > 2:
                is_gray = 0
            else:
                is_gray = 1
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            if rotate:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

            scanline_mask = self.create_scanline_mask(image.shape[1], image.shape[0], line_width)

            meta_mask = self.create_scanline_mask(
                image.shape[1],
                image.shape[0],
                line_width * random.randint(10, 25),
            )

            image_output = self.apply_scanline_mask(
                image, 
                scanline_mask, 
                meta_mask,
                scanline_type
            ).astype("uint8")

            if rotate:
                image_output = cv2.rotate(image_output, cv2.ROTATE_90_COUNTERCLOCKWISE)

            if is_gray:
                image_output = cv2.cvtColor(image_output, cv2.COLOR_BGR2GRAY)
                
            # Update the layer's image
            layer.image = image_output
            
        return meta