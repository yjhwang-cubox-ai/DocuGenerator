import random

import cv2
import numpy as np

from synthdocs.components.component import Component


class ColorShift(Component):
    """Shifts each BGR color channel by certain offsets to create a shifted color effect.

    :param color_shift_offset_x_range: Pair of ints/floats determining the value of x offset in shifting each color channel.
            If the value is within the range of 0.0 to 1.0 and the value is float,
            the x offset will be scaled by image width:
            x offset (int) = image width  * x offset (float and 0.0 - 1.0)
    :type color_shift_offset_x_range: tuple, optional
    :param color_shift_offset_y_range: Pair of ints/floats determining the value of y offset in shifting each color channel.
            If the value is within the range of 0.0 to 1.0 and the value is float,
            the y offset will be scaled by image height:
            y offset (int) = image height  * y offset (float and 0.0 - 1.0)
    :type color_shift_offset_y_range: tuple, optional
    :param color_shift_iterations: Pair of ints determining the number of iterations in applying the color shift operation.
    :type color_shift_iterations: tuple, optional
    :param color_shift_brightness_range: Pair of floats determining the brightness value of the shifted color channel.
            The optimal brightness range is 0.9 to 1.1.
    :type color_shift_brightness_range: tuple, optional
    :param color_shift_gaussian_kernel_range : Pair of ints determining the Gaussian kernel value in blurring the shifted image.
    :type color_shift_gaussian_kernel_range : tuple, optional
    :param p: The probability that this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        color_shift_offset_x_range=(3, 5),
        color_shift_offset_y_range=(3, 5),
        color_shift_iterations=(2, 3),
        color_shift_brightness_range=(0.9, 1.1),
        color_shift_gaussian_kernel_range=(3, 3),
        p=1,
    ):
        """Constructor method"""
        super().__init__()
        self.color_shift_offset_x_range = color_shift_offset_x_range
        self.color_shift_offset_y_range = color_shift_offset_y_range
        self.color_shift_iterations = color_shift_iterations
        self.color_shift_brightness_range = color_shift_brightness_range
        self.color_shift_gaussian_kernel_range = color_shift_gaussian_kernel_range
        self.p = p

    def apply_color_shift(self, image, kernel_value, offset_x_range, offset_y_range, brightness_range):
        """Main function to apply color shift process.

        :param image: The input image.
        :type image: numpy array
        :param kernel_value: The Gaussian kernel value for the blurring effect.
        :type kernel_value: int
        :param offset_x_range: Range for x offset.
        :type offset_x_range: tuple
        :param offset_y_range: Range for y offset.
        :type offset_y_range: tuple
        :param brightness_range: Range for brightness adjustment.
        :type brightness_range: tuple
        :return: Image with color shift applied.
        :rtype: numpy.ndarray
        """
        image_output = image.copy()

        ysize, xsize = image.shape[:2]

        image_b, image_g, image_r = cv2.split(image.copy())
        images = [image_b, image_g, image_r]

        brightness_ratio = random.uniform(brightness_range[0], brightness_range[1])

        # possible combinations in term of x and y direction
        directions = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
        for i, image_single_color in enumerate(images):

            # get random direction
            index = random.randint(0, len(directions) - 1)
            direction_x, direction_y = directions.pop(index)

            # generate random offsets
            if offset_x_range[0] <= 1.0 and isinstance(offset_x_range[0], float):
                offset_x = random.randint(
                    int(offset_x_range[0] * xsize),
                    int(offset_x_range[1] * xsize),
                )
            else:
                offset_x = random.randint(offset_x_range[0], offset_x_range[1])
            if offset_y_range[0] <= 1.0 and isinstance(offset_y_range[0], float):
                offset_y = random.randint(
                    int(offset_y_range[0] * ysize),
                    int(offset_y_range[1] * ysize),
                )
            else:
                offset_y = random.randint(offset_y_range[0], offset_y_range[1])

            # y direction
            translation_matrix_y = np.float32([[1, 0, 0], [0, 1, offset_y * direction_y]])
            # get a copy of translated area
            if direction_y > 0:
                image_patch = image_single_color[-offset_y:, :].copy()
            else:
                image_patch = image_single_color[:offset_y, :].copy()
            # shift image in y direction
            image_single_color = cv2.warpAffine(image_single_color, translation_matrix_y, (xsize, ysize))
            # fill back the empty area after translation
            if direction_y > 0:
                image_single_color[:offset_y, :] = image_patch
            else:
                image_single_color[-offset_y:, :] = image_patch

            # x direction
            translation_matrix_x = np.float32([[1, 0, offset_x * direction_x], [0, 1, 0]])
            # get a copy of translated area
            if direction_x > 0:
                image_patch = image_single_color[:, -offset_x:].copy()
            else:
                image_patch = image_single_color[:, :offset_x].copy()
            # shift image in x direction
            image_single_color = cv2.warpAffine(image_single_color, translation_matrix_x, (xsize, ysize))
            # fill back the empty area after translation
            if direction_x > 0:
                image_single_color[:, :offset_x] = image_patch
            else:
                image_single_color[:, -offset_x:] = image_patch

            # apply random brightness
            image_single_color_ratio = image_single_color.astype("int") * brightness_ratio
            image_single_color_ratio[image_single_color_ratio > 255] = 255
            image_single_color_ratio[image_single_color_ratio < 0] = 0
            image_single_color_ratio = image_single_color_ratio.astype("uint8")

            # perform blur in each image channel
            image_single_color_ratio = cv2.GaussianBlur(image_single_color_ratio, (kernel_value, kernel_value), 0)

            # reassign the shifted color channel back to the image
            image_output[:, :, i] = image_single_color_ratio

            # blend input single channel image with the shifted single channel image
            image_output[:, :, i] = cv2.addWeighted(image[:, :, i], 0.7, image_output[:, :, i], 0.3, 0)

        return image_output

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
        
        # Sample iterations
        iterations = meta.get("iterations", random.randint(
            self.color_shift_iterations[0], 
            self.color_shift_iterations[1]
        ))
        
        # Sample kernel size (must be odd)
        kernel_value = meta.get("kernel_value", random.randint(
            self.color_shift_gaussian_kernel_range[0],
            self.color_shift_gaussian_kernel_range[1],
        ))
        # Kernel must be odd
        if not (kernel_value % 2):
            kernel_value += 1
        
        # Build metadata
        meta.update({
            "iterations": iterations,
            "kernel_value": kernel_value,
            "offset_x_range": self.color_shift_offset_x_range,
            "offset_y_range": self.color_shift_offset_y_range,
            "brightness_range": self.color_shift_brightness_range
        })
        
        return meta

    def apply(self, layers, meta=None):
        """Apply the ColorShift effect to layers.
        
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
        iterations = meta["iterations"]
        kernel_value = meta["kernel_value"]
        offset_x_range = meta["offset_x_range"]
        offset_y_range = meta["offset_y_range"]
        brightness_range = meta["brightness_range"]
        
        for layer in layers:
            image = layer.image.copy()
            
            # Convert and make sure image is color image
            has_alpha = 0
            if len(image.shape) > 2:
                is_gray = 0
                if image.shape[2] == 4:
                    has_alpha = 1
                    image, image_alpha = image[:, :, :3], image[:, :, 3]
            else:
                is_gray = 1
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            # Apply color shift iterations
            image_output = image
            current_kernel = kernel_value
            for i in range(iterations):
                image_output = self.apply_color_shift(
                    image_output, 
                    current_kernel, 
                    offset_x_range, 
                    offset_y_range, 
                    brightness_range
                )
                # Increase kernel value in each iteration to create a better effect
                current_kernel += 2
            
            # Convert back to original format
            if is_gray:
                image_output = cv2.cvtColor(image_output, cv2.COLOR_BGR2GRAY)
            if has_alpha:
                image_output = np.dstack((image_output, image_alpha))
            
            # Update the layer's image
            layer.image = image_output
            
        return meta