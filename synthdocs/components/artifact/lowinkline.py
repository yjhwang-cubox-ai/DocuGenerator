################################################################################
# File: lowinkline.py
#
import random

import cv2
import numpy as np

from synthdocs.components.component import Component


class LowInkLine(Component):
    """Generates streaking behavior common to printers running out of ink.

    :param use_consistent_lines: Whether or not to vary the width and alpha of
           generated low ink lines.
    :type use_consistent_lines: bool, optional
    :param noise_probability: The probability to add noise into the generated lines.
    :type noise_probability: float, optional
    """

    def __init__(
        self,
        use_consistent_lines=True,
        noise_probability=0.1,
    ):
        """Constructor method"""
        super().__init__()
        self.use_consistent_lines = use_consistent_lines
        self.noise_probability = noise_probability

    # Takes an image, a vertical position, and an opacity value,
    # then adds a line at that position in the image with the given
    # opacity.
    def add_transparency_line(self, mask, y, alpha=None):
        """Adds a line with some opacity at a vertical position in the image.

        :param mask: The image to apply the line to.
        :type mask: numpy.array
        :param y: The vertical position to apply the line at.
        :type y: int
        :param alpha: The desired opacity of the line.
        :type alpha: int, optional
        :return: The image with a transparent line added.
        :rtype: numpy.array
        """
        result = mask.copy()
        ysize, xsize = result.shape[:2]

        if alpha is None:
            alpha = random.randint(16, 224)

        if self.use_consistent_lines:
            low_ink_line = np.full(result[y, :].shape, alpha, dtype="uint8")

            # add noise to top and bottom of the line
            if y - 1 >= 0:
                if len(result.shape) > 2:
                    indices = np.random.random((xsize, result.shape[2])) > (1 - self.noise_probability)
                else:
                    indices = np.random.random((xsize)) > (1 - self.noise_probability)
                low_ink_line_top = result[y - 1, :].copy()
                low_ink_line_top[indices] = alpha

                if len(result.shape) > 2:
                    stacked_line = [
                        low_ink_line_top[:, 0],
                        low_ink_line_top[:, 0],
                        low_ink_line_top[:, 0],
                    ]
                    if result.shape[2] == 4:
                        stacked_line += [low_ink_line_top[:, 3]]
                    low_ink_line_top = np.dstack(stacked_line)[0]

            if y + 1 < result.shape[0]:
                if len(result.shape) > 2:
                    indices = np.random.random((xsize, result.shape[2])) > (1 - self.noise_probability)
                else:
                    indices = np.random.random((xsize)) > (1 - self.noise_probability)
                low_ink_line_bottom = result[y + 1, :].copy()
                low_ink_line_bottom[indices] = alpha

                if len(result.shape) > 2:
                    stacked_line = [
                        low_ink_line_bottom[:, 0],
                        low_ink_line_bottom[:, 0],
                        low_ink_line_bottom[:, 0],
                    ]
                    if result.shape[2] == 4:
                        stacked_line += [low_ink_line_bottom[:, 3]]
                    low_ink_line_bottom = np.dstack(stacked_line)[0]

        else:
            low_ink_line = (np.random.random((xsize)) * 255).astype("uint8")
            if len(result.shape) > 2:
                new_low_ink_line = np.zeros((xsize, result.shape[2]), dtype="uint8")
                for i in range(result.shape[2]):
                    new_low_ink_line[:, i] = low_ink_line.copy()
                low_ink_line = new_low_ink_line

            # add noise to top and bottom of the line
            if y - 1 >= 0:
                indices = np.random.random((xsize)) <= (1 - self.noise_probability)
                low_ink_line_top = (np.random.random((xsize)) * 255).astype("uint8")
                if len(result.shape) > 2:
                    new_low_ink_line_top = np.zeros((xsize, result.shape[2]), dtype="uint8")
                    for i in range(result.shape[2]):
                        new_low_ink_line_top[:, i] = low_ink_line_top.copy()
                        new_low_ink_line_top[:, i][indices] = result[y - 1, :, i][indices]
                    low_ink_line_top = new_low_ink_line_top
                else:
                    low_ink_line_top[indices] = result[y - 1, :][indices]

            if y + 1 < result.shape[0]:
                indices = np.random.random((xsize)) <= (1 - self.noise_probability)
                low_ink_line_bottom = (np.random.random((xsize)) * 255).astype("uint8")
                if len(result.shape) > 2:
                    new_low_ink_line_bottom = np.zeros((xsize, result.shape[2]), dtype="uint8")
                    for i in range(result.shape[2]):
                        new_low_ink_line_bottom[:, i] = low_ink_line_bottom.copy()
                        new_low_ink_line_bottom[:, i][indices] = result[y - 1, :, i][indices]
                    low_ink_line_bottom = new_low_ink_line_bottom
                else:
                    low_ink_line_bottom[indices] = result[y - 1, :][indices]

        if len(result.shape) > 2:
            indices = result[y, :, :3] < low_ink_line[:, :3]
            result[y, :, :3][indices] = low_ink_line[:, :3][indices]
        else:
            indices = result[y, :] < low_ink_line
            result[y, :][indices] = low_ink_line[indices]

        if y - 1 >= 0:
            if len(result.shape) > 2:
                indices = result[y - 1, :, :3] < low_ink_line_top[:, :3]
                result[y - 1, :, :3][indices] = low_ink_line_top[:, :3][indices]
            else:
                indices = result[y - 1, :] < low_ink_line_top
                result[y - 1, :][indices] = low_ink_line_top[indices]

        if y + 1 < result.shape[0]:
            if len(result.shape) > 2:
                indices = result[y + 1, :, :3] < low_ink_line_bottom[:, :3]  # 수정된 부분
                result[y + 1, :, :3][indices] = low_ink_line_bottom[:, :3][indices]
            else:
                indices = result[y + 1, :] < low_ink_line_bottom  # 수정된 부분
                result[y + 1, :][indices] = low_ink_line_bottom[indices]

        return result

    def sample(self, meta=None):
        if meta is None:
            meta = {}
        
        # This component doesn't need specific sampling parameters
        # since it's mainly used as a base component
        
        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        
        # This base component doesn't directly apply effects
        # It's mainly used by derived components like LowInkRandomLines
        
        return meta