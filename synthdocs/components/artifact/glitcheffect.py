import random

import cv2
import numpy as np

from synthdocs.components.artifact.colorshift import ColorShift
from synthdocs.components.component import Component


class GlitchEffect(Component):
    """Create glitch effect by applying ColorShift and shifts patches of image horizontally or vertically.

    :param glitch_direction: Direction of the glitch effect, select from "vertical", "horizontal", "all" or "random".
    :type glitch_direction: string, optional
    :param glitch_number_range: Tuple of ints determing the number of shifted image patches.
    :type glitch_number_range: tuple, optional
    :param glitch_size_range: Tuple of ints/floats determing the size of image patches.
            If the value is within the range of 0.0 to 1.0 and the value is float,
            the size will be scaled by image height:
            size (int) = image height  * size (float and 0.0 - 1.0)
    :type glitch_size_range: tuple, optional
    :param glitch_offset_range: Tuple of ints/floats determing the offset value to shift the image patches.
            If the value is within the range of 0.0 to 1.0 and the value is float,
            the size will be scaled by image width:
            offset (int) = image width  * offset (float and 0.0 - 1.0)
    :type glitch_offset_range: tuple, optional
    :param p: The probability that this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        glitch_direction="random",
        glitch_number_range=(8, 16),
        glitch_size_range=(5, 50),
        glitch_offset_range=(10, 50),
    ):
        """Constructor method"""
        super().__init__()
        self.glitch_direction = glitch_direction
        self.glitch_number_range = glitch_number_range
        self.glitch_size_range = glitch_size_range
        self.glitch_offset_range = glitch_offset_range

    def apply_glitch(self, image, glitch_direction, mask=None, keypoints=None, bounding_boxes=None):
        """Apply glitch effect into the image by shifting patches of images.

        :param image: Image to apply the glitch effect.
        :type image: numpy array
        :param glitch_direction: The direction of glitch effect.
        :type glitch_direction: string
        :param mask: The mask of labels for each pixel. Mask value should be in range of 1 to 255.
            Value of 0 will be assigned to the filled area after the transformation.
        :type mask: numpy array (uint8), optional
        :param keypoints: A dictionary of single or multiple labels where each label is a nested list of points coordinate.
        :type keypoints: dictionary, optional
        :param bounding_boxes: A nested list where each nested list contains box location (x1, y1, x2, y2).
        :type bounding_boxes: list, optional
        :return: Tuple of (modified_image, modified_mask)
        :rtype: tuple
        """

        # input image shape
        ysize, xsize = image.shape[:2]
        glitch_number = random.randint(self.glitch_number_range[0], self.glitch_number_range[1])
        for i in range(glitch_number):

            # generate random glitch size
            if self.glitch_size_range[0] <= 1.0 and isinstance(self.glitch_size_range[0], float):
                glitch_size = random.randint(
                    int(self.glitch_size_range[0] * ysize),
                    int(self.glitch_size_range[1] * ysize),
                )
            else:
                glitch_size = random.randint(self.glitch_size_range[0], self.glitch_size_range[1])

            # generate random direction
            direction = random.choice([1, -1])

            # generate random glitch offset
            if self.glitch_offset_range[0] <= 1.0 and isinstance(self.glitch_offset_range[0], float):
                glitch_offset = (
                    random.randint(int(self.glitch_offset_range[0] * xsize), int(self.glitch_offset_range[1] * xsize))
                    * direction
                )
            else:
                glitch_offset = random.randint(self.glitch_offset_range[0], self.glitch_offset_range[1]) * direction

            # vertical glitch effect
            if glitch_direction == "vertical":
                # get a patch of image
                start_x = random.randint(0, xsize - glitch_size)
                image_patch = image[:, start_x : start_x + glitch_size]
                if mask is not None:
                    mask_patch = mask[:, start_x : start_x + glitch_size]
                pysize, pxsize = image_patch.shape[:2]

                # create translation matrix in vertical direction
                translation_matrix = np.float32([[1, 0, 0], [0, 1, glitch_offset]])

                # get a copy of translated area
                image_patch_fill = image_patch[-abs(glitch_offset) :, :].copy()

            # horizontal glitch effect
            else:
                # get a patch of image
                start_y = random.randint(0, ysize - glitch_size)
                image_patch = image[start_y : start_y + glitch_size, :]
                if mask is not None:
                    mask_patch = mask[start_y : start_y + glitch_size, :]
                pysize, pxsize = image_patch.shape[:2]

                # create translation matrix in horizontal direction
                translation_matrix = np.float32([[1, 0, glitch_offset], [0, 1, 0]])

                # get a copy of translated area
                image_patch_fill = image_patch[:, -abs(glitch_offset) :].copy()

            # translate image
            image_patch = cv2.warpAffine(image_patch, translation_matrix, (pxsize, pysize))
            # translate mask
            if mask is not None:
                mask_patch = cv2.warpAffine(mask_patch, translation_matrix, (pxsize, pysize))

            # translate keypoints
            if keypoints is not None:
                for name, points in keypoints.items():
                    for i, (xpoint, ypoint) in enumerate(points):
                        if glitch_direction == "vertical":
                            if (xpoint >= start_x) and (xpoint < (start_x + glitch_size)):
                                points[i] = [xpoint, ypoint + glitch_offset]
                        else:
                            if (ypoint >= start_y) and (ypoint < (start_y + glitch_size)):
                                points[i] = [xpoint + glitch_offset, ypoint]

            # translate bounding boxes
            if bounding_boxes is not None:
                new_boxes = []
                for i, bounding_box in enumerate(bounding_boxes):
                    xspoint, yspoint, xepoint, yepoint = bounding_box

                    if glitch_direction == "vertical":
                        # both start and end point within translated area
                        if (
                            (xspoint >= start_x)
                            and (xspoint < (start_x + glitch_size))
                            and (xepoint >= start_x)
                            and (xepoint < (start_x + glitch_size))
                        ):
                            bounding_boxes[i] = [
                                xspoint,
                                max(0, yspoint + glitch_offset),
                                xepoint,
                                min(yepoint + glitch_offset, ysize - 1),
                            ]

                        # left portion of box is in translation area, but right portion is not
                        elif (
                            (xspoint >= start_x)
                            and (xspoint < (start_x + glitch_size))
                            and ((xepoint < start_x) or (xepoint >= (start_x + glitch_size)))
                        ):
                            # shift left box
                            bounding_boxes[i] = [
                                xspoint,
                                max(0, yspoint + glitch_offset),
                                start_x + glitch_size,
                                min(yepoint + glitch_offset, ysize - 1),
                            ]
                            # remain right box
                            new_boxes.append(
                                [
                                    start_x + glitch_size,
                                    yspoint,
                                    xepoint,
                                    yepoint,
                                ],
                            )

                        # right portion of box is in translation area, but left portion is not
                        elif (
                            ((xspoint < start_x) or (xspoint >= (start_x + glitch_size)))
                            and (xepoint >= start_x)
                            and (xepoint < (start_x + glitch_size))
                        ):
                            # shift right box
                            bounding_boxes[i] = [
                                start_x,
                                max(0, yspoint + glitch_offset),
                                xepoint,
                                min(yepoint + glitch_offset, ysize - 1),
                            ]
                            # remain left box
                            new_boxes.append(
                                [
                                    xspoint,
                                    yspoint,
                                    start_x,
                                    yepoint,
                                ],
                            )

                    else:
                        # both start and end point within translated area
                        if (
                            (yspoint >= start_y)
                            and (yspoint < (start_y + glitch_size))
                            and (yepoint >= start_y)
                            and (yepoint < (start_y + glitch_size))
                        ):
                            bounding_boxes[i] = [
                                max(0, xspoint + glitch_offset),
                                yspoint,
                                min(xepoint + glitch_offset, xsize - 1),
                                yepoint,
                            ]

                        # top portion of box is in translation area, but bottom portion is not
                        elif (
                            (yspoint >= start_y)
                            and (yspoint < (start_y + glitch_size))
                            and ((yepoint < start_y) or (yepoint >= (start_y + glitch_size)))
                        ):
                            # shift top box
                            bounding_boxes[i] = [
                                max(0, xspoint + glitch_offset),
                                yspoint,
                                min(xepoint + glitch_offset, xsize - 1),
                                start_y + glitch_size,
                            ]
                            # remain bottom box
                            new_boxes.append(
                                [
                                    xspoint,
                                    start_y + glitch_size,
                                    xepoint,
                                    yepoint,
                                ],
                            )

                        # bottom portion of box is in translation area, but top portion is not
                        elif (
                            ((yspoint < start_y) or (yspoint >= (start_y + glitch_size)))
                            and (yepoint >= start_y)
                            and (yepoint < (start_y + glitch_size))
                        ):
                            # shift bottom box
                            bounding_boxes[i] = [
                                max(0, xspoint + glitch_offset),
                                start_y,
                                min(xepoint + glitch_offset, xsize - 1),
                                yepoint,
                            ]

                            # remain top box
                            new_boxes.append(
                                [
                                    xspoint,
                                    yspoint,
                                    xepoint,
                                    start_y,
                                ],
                            )

                # merge boxes
                bounding_boxes += new_boxes

            # fill back the empty area after translation
            if glitch_direction == "vertical":
                if direction > 0:
                    image_patch[:glitch_offset, :] = image_patch_fill
                    # mask's empty area is filled with 0
                    if mask is not None:
                        mask_patch[:glitch_offset, :] = 0
                else:
                    image_patch[glitch_offset:, :] = image_patch_fill
                    # mask's empty area is filled with 0
                    if mask is not None:
                        mask_patch[glitch_offset:, :] = 0
            else:
                if direction > 0:
                    image_patch[:, :glitch_offset] = image_patch_fill
                    # mask's empty area is filled with 0
                    if mask is not None:
                        mask_patch[:, :glitch_offset] = 0

                else:
                    image_patch[:, glitch_offset:] = image_patch_fill
                    # mask's empty area is filled with 0
                    if mask is not None:
                        mask_patch[:, glitch_offset:] = 0

            # randomly scale single channel to create a single color contrast effect
            random_ratio = random.uniform(0.8, 1.2)
            channel = random.randint(0, 2)
            image_patch_ratio = image_patch[:, :, channel].astype("int") * random_ratio
            image_patch_ratio[image_patch_ratio > 255] = 255
            image_patch_ratio[image_patch_ratio < 0] = 0
            image_patch[:, :, channel] = image_patch_ratio.astype("uint8")

            if glitch_direction == "vertical":
                image[:, start_x : start_x + glitch_size] = image_patch
                if mask is not None:
                    mask[:, start_x : start_x + glitch_size] = mask_patch
            else:
                image[start_y : start_y + glitch_size, :] = image_patch
                if mask is not None:
                    mask[start_y : start_y + glitch_size, :] = mask_patch

        return image, mask

    def sample(self, meta=None):
        """Sample random parameters for the augmentation.
        
        :param meta: Optional metadata dictionary with parameters to use.
        :type meta: dict, optional
        :return: Dictionary with sampled parameters.
        :rtype: dict
        """
        if meta is None:
            meta = {}
            
        meta["run"] = True
        
        # Sample glitch direction
        glitch_direction = meta.get("glitch_direction", None)
        if glitch_direction is None:
            if self.glitch_direction == "random":
                glitch_direction = random.choice(["vertical", "horizontal"])
            else:
                glitch_direction = self.glitch_direction
                
        # For "all" direction, determine the order
        horizontal_first = meta.get("horizontal_first", None)
        if glitch_direction == "all" and horizontal_first is None:
            horizontal_first = random.random() > 0.5
            
        # Build metadata
        meta.update({
            "glitch_direction": glitch_direction,
            "horizontal_first": horizontal_first
        })
        
        return meta

    def apply(self, layers, meta=None):
        """Apply the GlitchEffect to layers.
        
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
        glitch_direction = meta["glitch_direction"]
        horizontal_first = meta.get("horizontal_first", False)
        
        for layer in layers:
            image = layer.image.copy()
            
            # Check if grayscale
            if len(image.shape) > 2:
                is_gray = 0
            else:
                is_gray = 1
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
                
            # Apply color shift before the glitch effect
            color_shift = ColorShift(
                color_shift_offset_x_range=(3, 5),
                color_shift_offset_y_range=(3, 5),
                color_shift_iterations=(1, 2),
                color_shift_brightness_range=(0.9, 1.1),
                color_shift_gaussian_kernel_range=(1, 3)
            )
            layer.image = image
            # Note: Assuming ColorShift is also modified to Component structure
            # We'll use its __call__ method for now to maintain the current logic
            # image_output = color_shift.apply(image)
            meta = color_shift.apply([layer])
            image_output = layer.image.copy()
            # Dummy mask for now, in a real implementation it would be passed in
            mask = None
            
            # Apply glitches based on direction
            if glitch_direction == "all":
                # Apply horizontal glitch before vertical glitch
                if horizontal_first:
                    image_output, mask = self.apply_glitch(image_output, "horizontal", mask)
                
                # Apply vertical glitch
                image_output, mask = self.apply_glitch(image_output, "vertical", mask)
                
                # Apply horizontal glitch after vertical glitch
                if not horizontal_first:
                    image_output, mask = self.apply_glitch(image_output, "horizontal", mask)
            else:
                image_output, mask = self.apply_glitch(image_output, glitch_direction, mask)
            
            # Convert back to grayscale if needed
            if is_gray:
                image_output = cv2.cvtColor(image_output, cv2.COLOR_BGRA2GRAY)
                
            # Update the layer's image
            layer.image = image_output
            
        return meta