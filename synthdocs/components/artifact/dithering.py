import random

import cv2
import numba as nb
import numpy as np
from numba import config
from numba import jit

from synthdocs.components.component import Component


class Dithering(Component):
    """
    Applies Ordered or Floyd Steinberg dithering to the input image.

    :param dither: Types of dithering, random, ordered, Floyd Steinberg dithering.
    :type dither: string, optional
    :param order: Pair of ints determining the range of order number for ordered dithering.
    :type order: tuple, optional
    :param numba_jit: The flag to enable numba jit to speed up the processing in the augmentation.
    :type numba_jit: int, optional
    """

    def __init__(
        self,
        dither="random",
        order=(2, 5),
        numba_jit=1,
    ):
        super().__init__()
        self.dither = dither
        self.order = order
        self.numba_jit = numba_jit
        config.DISABLE_JIT = bool(1 - numba_jit)

    # Floyd Steinberg dithering
    def dither_Floyd_Steinberg(self, image):
        """Apply Floyd Steinberg dithering to the input image.

        :param image: The image to apply the function.
        :type image: numpy.array (numpy.uint8)
        """

        ysize, xsize = image.shape[:2]
        img_dither_fs = image.copy().astype("float")
        if len(image.shape) > 2:  # coloured image
            # skip alpha channel
            for channel_num in range(3):
                self.apply_Floyd_Steinberg(
                    img_dither_fs[:, :, channel_num],
                    ysize,
                    xsize,
                )
        else:  # grayscale or binary
            self.apply_Floyd_Steinberg(img_dither_fs, ysize, xsize)

        return img_dither_fs.astype("uint8")

    @staticmethod
    @jit(nopython=True, cache=True)
    def apply_Floyd_Steinberg(image, ysize, xsize):
        """Run Floyd Steinberg dithering algorithm to the input image.

        :param image: The image to apply the function.
        :type image: numpy.array (numpy.uint8)
        :param ysize: Height of image.
        :type ysize: int
        :param xsize: Width of image.
        :type xsize: int
        """

        for y in range(1, ysize - 1):
            for x in range(1, xsize - 1):
                old_pixel = image[y, x]
                new_pixel = 255 * np.floor(old_pixel / 128)
                image[y, x] = new_pixel
                quant_error = min(old_pixel - new_pixel, 0)  # remove negative
                image[y, x + 1] += quant_error * (7 / 16)
                image[y + 1, x - 1] += quant_error * (3 / 16)
                image[y + 1, x] += quant_error * (5 / 16)
                image[y + 1, x + 1] += quant_error * (1 / 16)

    @staticmethod
    @jit(nopython=True, cache=True, parallel=True)
    def apply_Ordered(image, ysize, xsize, order, ordered_matrix):
        """Run ordered dithering algorithm to the input image.

        :param image: The image to apply the function.
        :type image: numpy.array (numpy.uint8)
        :param ysize: Height of image.
        :type ysize: int
        :param xsize: Width of image.
        :type xsize: int
        :param order: Order number of ordered dithering.
        :type order: int
        :param ordered_matrix: Ordered matrix for ordered dithering algorithm.
        :type ordered_matrix: list
        """

        for y in nb.prange(ysize):
            for x in nb.prange(xsize):
                oy = y % order
                ox = x % order
                if image[y, x] > ordered_matrix[oy, ox]:
                    image[y, x] = 255
                else:
                    image[y, x] = 0

    def dither_Ordered(self, image, order=5):
        """Apply ordered dithering to the input image.

        :param image: The image to apply the function.
        :type image: numpy.array (numpy.uint8)
        :param order: Order number of the ordered dithering.
        :type order: int

        """
        # create bayer matrix based on the order
        ordered_matrix = self.create_bayer(0, 0, 2 ** (order), 0, 1)
        total_number = len(ordered_matrix) * len(ordered_matrix[0]) - 1
        # quantitize order matrix value
        for y, row in enumerate(ordered_matrix):
            for x, value in enumerate(row):
                ordered_matrix[y][x] = np.floor((value / total_number) * 255)
        ordered_matrix = np.array(ordered_matrix, dtype="float64")

        ysize, xsize = image.shape[:2]
        img_dither_ordered = image.copy().astype("float")
        if len(image.shape) > 2:  # coloured image
            # skip alpha channel
            for channel_num in range(3):
                self.apply_Ordered(
                    img_dither_ordered[:, :, channel_num],
                    ysize,
                    xsize,
                    order,
                    ordered_matrix,
                )
        else:  # grayscale or binary
            self.apply_Ordered(
                img_dither_ordered,
                ysize,
                xsize,
                order,
                ordered_matrix,
            )

        return img_dither_ordered.astype("uint8")

    # Adapted from https://github.com/tromero/BayerMatrix
    def create_bayer(self, x, y, size, value, step, matrix=[[]]):
        """Function to create ordered matrix.

        :param x: The x coordinate of current step.
        :type x: int
        :param y: The y coordinate of current step.
        :type y: int
        :param size: Size of ordered matrix.
        :type size: int
        :param value: Value of current step.
        :type value: int
        :param step: Current step value.
        :type step: int
        :param _matrix: The ordered matrix for ordered dithering algorithm.
        :type matrix: list
        """
        if matrix == [[]]:
            matrix = [[0 for i in range(size)] for i in range(size)]
        if size == 1:
            matrix[int(y)][int(x)] = value
            return
        half = size / 2
        # subdivide into quad tree and call recursively
        # pattern is TL, BR, TR, BL
        self.create_bayer(x, y, half, value + (step * 0), step * 4, matrix)
        self.create_bayer(
            x + half,
            y + half,
            half,
            value + (step * 1),
            step * 4,
            matrix,
        )
        self.create_bayer(x + half, y, half, value + (step * 2), step * 4, matrix)
        self.create_bayer(x, y + half, half, value + (step * 3), step * 4, matrix)
        return matrix

    def sample(self, meta=None):
        if meta is None:
            meta = {}
        
        # Determine dither type
        dither_type = meta.get("dither_type", None)
        if dither_type is None:
            if self.dither == "random":
                # dither_type = random.choice(["ordered", "Floyd Steinberg"])
                dither_type = random.choice(["ordered"])
            else:
                dither_type = self.dither
                
        # Get order value for ordered dithering
        order = meta.get("order", None)
        if order is None and dither_type == "ordered":
            order = random.randint(self.order[0], self.order[1])
        
        # Build metadata
        meta = {
            "dither_type": dither_type,
            "order": order,
        }
        
        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        
        dither_type = meta["dither_type"]
        order = meta["order"]
        
        for layer in layers:
            image = layer.image.copy().astype(np.uint8)
            
            # Check and convert image into BGR format
            if len(image.shape) > 2:
                is_gray = 0
            else:
                is_gray = 1
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            # Apply appropriate dithering
            if dither_type == "ordered":
                image_dither = self.dither_Ordered(image, order)
            else:
                image_dither = self.dither_Floyd_Steinberg(image)
            
            # Convert back to grayscale if original was grayscale
            if is_gray:
                image_dither = cv2.cvtColor(image_dither, cv2.COLOR_BGR2GRAY)
            
            # Update layer's image
            layer.image = image_dither
            
        return meta
