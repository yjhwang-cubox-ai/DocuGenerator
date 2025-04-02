import math
import os
import random
from pathlib import Path

import cv2
import numpy as np

from augraphy.augmentations.brightness import Brightness
from synthdocs.utils.lib import add_noise as lib_add_noise
from synthdocs.utils.lib import generate_average_intensity
from synthdocs.utils.lib import smooth
from synthdocs.utils.lib import sobel
from synthdocs.components.component import Component
from synthdocs.utils.inkgenerator import InkGenerator


class Markup(Component):
    """Uses contours detection to detect text lines and add a smooth text strikethrough, highlight or underline effect.

    :param num_lines_range: Pair of ints determining the number of added markup effect.
    :type num_lines_range: int tuple, optional
    :param markup_length_range: Pair of floats between 0 to 1, to determine the length of added markup effect.
    :type markup_length_range: float tuple, optional
    :param markup_thickness_range: Pair of ints, to determine the thickness of added markup effect.
    :type markup_thickness_range: int tuple, optional
    :param markup_type: Choice of markup "strikethrough", "highlight", "underline" or "crossed".
    :type markup_type: string
    :param markup_ink: Types of markup ink, choose from "random", "pencil", "pen", "marker" or "highlighter".
    :type markup_ink: string, optional
    :param markup_color: BGR color tuple.
    :type markup_color: tuple or string
    :param repetitions: Determine how many time a single markup effect should be drawn.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
    :type repetitions: int
    :param large_word_mode: Set true to draw markup on large words, else large word will be ignored.
    :type large_word_mode: boolean
    :param single_word_mode: Set true to draw markup on a single word only.
    :type single_word_mode: boolean
    :param p: The probability that this effect will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        num_lines_range=(2, 7),
        markup_length_range=(0.5, 1),
        markup_thickness_range=(1, 3),
        markup_type="random",
        markup_ink="random",
        markup_color="random",
        large_word_mode="random",
        single_word_mode=False,
        repetitions=1,
        p=1,
    ):
        """Constructor method"""
        super().__init__()
        self.num_lines_range = num_lines_range
        self.markup_length_range = markup_length_range
        self.markup_thickness_range = markup_thickness_range
        self.markup_type = markup_type
        self.markup_ink = markup_ink
        self.markup_color = markup_color
        self.repetitions = repetitions
        self.large_word_mode = large_word_mode
        self.single_word_mode = single_word_mode
        self.p = p

    def distribute_line(self, starting_point, ending_point, offset):
        """Create smoothed line from the provided starting and ending point.

        :param starting_point: Starting point (x, y) of the line.
        :type starting_point: tuple
        :param ending_point: Ending point (x, y) of the line.
        :type ending_point: tuple
        :param offset: Offset value to randomize point position.
        :type offset: int
        :return: Smoothed points
        :rtype: numpy.ndarray
        """
        points_count = random.randint(3, 6)  # dividing the line into points
        points_x = np.linspace(starting_point[0], ending_point[0], points_count)
        points_y = [starting_point[1] + random.uniform(-offset, offset) for _ in points_x]
        points = smooth(
            np.column_stack((points_x, points_y)).astype("float"),
            6,
        )  # adding a smoothing effect in points using chaikin's algorithm
        return points

    def _preprocess(self, image, single_word_mode):
        """Preprocess image with binarization, dilation and erosion.
        
        :param image: Input image to process
        :type image: numpy.ndarray
        :param single_word_mode: Whether to process in single word mode
        :type single_word_mode: bool
        :return: Preprocessed binary image
        :rtype: numpy.ndarray
        """
        blurred = cv2.blur(image, (5, 5))
        blurred = blurred.astype("uint8")
        if len(blurred.shape) > 2 and blurred.shape[2] == 3:
            blurred = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        elif len(blurred.shape) > 2 and blurred.shape[2] == 4:
            blurred = cv2.cvtColor(blurred, cv2.COLOR_BGRA2GRAY)

        _, binarized = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV,
        )

        # get kernel for dilation
        if single_word_mode is False:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))

        # dilating the threshold image to combine horizontal lines
        dilation = cv2.dilate(
            binarized,
            kernel,
            iterations=2,
        )
        dilation = cv2.erode(
            dilation,
            None,
            iterations=1,
        )

        return dilation

    def draw_line(self, p1, p2, markup_mask, markup_thickness, markup_color, reverse):
        """Draw line across two provided points.

        :param p1: Starting point (x, y) of the line.
        :type p1: tuple
        :param p2: Ending point (x, y) of the line.
        :type p2: tuple
        :param markup_mask: Mask of markup effect.
        :type markup_mask: numpy.array (numpy.uint8)
        :param markup_thickness: Thickness of the line.
        :type markup_thickness: int
        :param markup_color: Color of the line in BGR format.
        :type markup_color: tuple
        :param reverse: Reverse the order of line points distribution.
        :type reverse: int
        :return: Updated markup mask
        :rtype: numpy.ndarray
        """
        # get min and max of points
        min_x = min(p2[0], p1[0])
        max_x = max(p2[0], p1[0])
        min_y = min(p2[1], p1[1])
        max_y = max(p2[1], p1[1])

        # set point x in ascending or descending order based on direction
        if reverse:
            points_x = [min_x, random.randint(min_x, max_x), max_x]
        else:
            points_x = [max_x, random.randint(min_x, max_x), min_x]
        points_y = [min_y, random.randint(min_y, max_y), max_y]

        # smooth points
        points = smooth(np.column_stack((points_x, points_y)).astype("float"), 6)

        # draw curvy lines
        for (point1_x, point1_y), (point2_x, point2_y) in zip(points[:-1], points[1:]):
            point1 = (int(point1_x), int(point1_y))
            point2 = (int(point2_x), int(point2_y))

            markup_mask = cv2.line(
                markup_mask,
                point1,
                point2,
                markup_color,
                markup_thickness,
                lineType=cv2.LINE_AA,
            )
            
        return markup_mask

    def sample(self, meta=None):
        """Sample random parameters for the markup effect.
        
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
        
        # Sample markup type
        if self.markup_type == "random":
            markup_type = random.choice(["strikethrough", "crossed", "underline", "highlight"])
        else:
            markup_type = self.markup_type
            
        # Sample markup color
        if self.markup_color == "random":
            markup_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        else:
            markup_color = self.markup_color
            
        # Sample large word mode
        if self.large_word_mode == "random":
            large_word_mode = random.choice([True, False])
        else:
            large_word_mode = self.large_word_mode
            
        # Sample number of lines
        num_lines = random.randint(self.num_lines_range[0], self.num_lines_range[1])
        
        # Sample markup ink
        if self.markup_ink == "random":
            markup_ink = random.choice(["pencil", "pen", "marker", "highlighter"])
        else:
            markup_ink = self.markup_ink
            
        # Build metadata
        meta.update({
            "markup_type": markup_type,
            "markup_color": markup_color,
            "large_word_mode": large_word_mode,
            "num_lines": num_lines,
            "markup_ink": markup_ink,
            "single_word_mode": self.single_word_mode,
            "markup_length_range": self.markup_length_range.copy() if self.single_word_mode else self.markup_length_range,
            "markup_thickness_range": self.markup_thickness_range,
            "repetitions": self.repetitions
        })
        
        return meta

    def apply(self, layers, meta=None):
        """Apply the markup effect to layers.
        
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
        markup_type = meta["markup_type"]
        markup_color = meta["markup_color"]
        large_word_mode = meta["large_word_mode"]
        num_lines = meta["num_lines"]
        markup_ink = meta["markup_ink"]
        single_word_mode = meta["single_word_mode"]
        markup_length_range = meta["markup_length_range"]
        markup_thickness_range = meta["markup_thickness_range"]
        repetitions = meta["repetitions"]
        
        # Adjust markup length range for single word mode
        if single_word_mode:
            markup_length_range = (1, 1)
        
        for layer in layers:
            image = layer.image.copy()
            
            # Change to 3 channels BGR format
            has_alpha = 0
            if len(image.shape) < 3:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                has_alpha = 1
                image, image_alpha = image[:, :, :3], image[:, :, 3]
                
            # Process contrast color if needed
            if markup_color == "contrast":
                single_color = cv2.resize(image, (1, 1), interpolation=cv2.INTER_AREA)
                markup_color = 255 - single_color[0][0]
                markup_color = markup_color.tolist()
            
            # Preprocess image
            binary_image = self._preprocess(image, single_word_mode)
            
            # Applying dilate operation to connect text lines horizontally
            contours, hierarchy = cv2.findContours(
                binary_image,
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_NONE,
            )  # Each line is detected as a contour
            
            # Calculate average character height
            heights = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                heights.append(h)
                
            # Get average of character height
            bins = np.unique(heights)
            hist, bin_edges = np.histogram(heights, bins=bins, density=False)
            if len(bin_edges) > 1 and np.max(hist) > 20:
                character_height_min = bin_edges[np.argmax(hist)]
                character_height_max = bin_edges[np.argmax(hist) + 1]
                character_height_average = int((character_height_max + character_height_min) / 2)
                height_range = ((character_height_max - character_height_min) / 2) + 1
            else:
                character_height_average = -1
                height_range = -1
                
            # Initialize coordinates of lines
            lines_coordinates = []
            
            # Shuffle contours to randomize location to apply augmentation
            if len(contours) > 0:
                contours = list(contours)
                random.shuffle(contours)
                
            for cnt in contours:
                # Adding randomization
                choice = random.choice([False, True])
                x, y, w, h = cv2.boundingRect(cnt)
                
                if character_height_average == -1:
                    check_height = h > 10
                else:
                    check_height = (h > character_height_average - height_range) and (
                        h < character_height_average + height_range
                    )
                    
                if large_word_mode:
                    conditions = check_height
                else:
                    conditions = (
                        choice
                        and (w > h * 2)
                        and (w * h < (image.shape[0] * image.shape[1]) / 10)
                        and w < int(image.shape[1] / 5)
                        and check_height
                    )
                    
                if conditions:
                    if num_lines == 0:
                        break
                    num_lines = num_lines - 1
                    markup_length = random.uniform(
                        markup_length_range[0],
                        markup_length_range[1],
                    )
                    # Adjusting width according to markup length
                    w = int(w * markup_length)
                    # Adjusting starting-point according to markup length
                    x = int(x + (1 - markup_length) * w)
                    # Offset to interpolate markup effect up/down
                    offset = 6
                    
                    # For strikethrough and highlight, we need center points
                    if markup_type == "strikethrough" or markup_type == "highlight":
                        starting_point = [x, int(y + (h / 2))]
                        ending_point = [x + w, int(y + (h / 2))]
                    # For crossed-off we need points representing primary diagonal
                    elif markup_type == "crossed":
                        starting_point = [x, y]
                        ending_point = [x + w, y + h]
                    else:
                        # For underline, we need points corresponding to bottom part of text
                        starting_point = [x, y + h]
                        ending_point = [x + w, y + h]
                        
                    for i in range(repetitions):
                        if markup_type == "crossed":
                            ysize, xsize = image.shape[:2]
                            
                            # Primary diagonal
                            p1_x = np.clip(
                                starting_point[0] + random.randint(-offset * 5, offset * 5),
                                0,
                                xsize,
                            )
                            p1_y = np.clip(
                                starting_point[1] + random.randint(-offset * 1, offset * 1),
                                0,
                                ysize,
                            )
                            p2_x = np.clip(
                                ending_point[0] + random.randint(-offset * 5, offset * 5),
                                0,
                                xsize,
                            )
                            p2_y = np.clip(
                                ending_point[1] + random.randint(-offset * 1, offset * 1),
                                0,
                                ysize,
                            )
                            p1 = (p1_x, p1_y)
                            p2 = (p2_x, p2_y)
                            lines_coordinates.append(np.array([p1, p2]))
                            
                            # Secondary diagonal
                            p1_x = np.clip(
                                ending_point[0] + random.randint(-offset * 5, offset * 5),
                                0,
                                xsize,
                            )
                            p1_y = np.clip(
                                starting_point[1] + random.randint(-offset * 1, offset * 1),
                                0,
                                ysize,
                            )
                            p2_x = np.clip(
                                starting_point[0] + random.randint(-offset * 5, offset * 5),
                                0,
                                xsize,
                            )
                            p2_y = np.clip(
                                ending_point[1] + random.randint(-offset * 1, offset * 1),
                                0,
                                ysize,
                            )
                            p1 = (p1_x, p1_y)
                            p2 = (p2_x, p2_y)
                            lines_coordinates.append(np.array([p1, p2]))
                            
                        else:
                            # Dividing the line into points to mimic a smoothing effect
                            points_list = self.distribute_line(
                                starting_point,
                                ending_point,
                                offset,
                            ).astype("int")
                            lines_coordinates.append(points_list)
                            
            # Process lines if coordinates are available
            if lines_coordinates:
                # For highlight, the ink should be thicker
                if markup_type == "highlight":
                    thickness_range = (markup_thickness_range[0] + 5, markup_thickness_range[1] + 5)
                else:
                    thickness_range = markup_thickness_range
                
                image = image.astype(np.uint8) 

                # Create ink generator
                ink_generator = InkGenerator(
                    ink_type=markup_ink,
                    ink_draw_method="lines",
                    ink_draw_iterations=(1, 1),
                    ink_location="random",
                    ink_background=image,
                    ink_background_size=None,
                    ink_background_color=None,
                    ink_color=markup_color,
                    ink_min_brightness=1,
                    ink_min_brightness_value_range=(150, 200),
                    ink_draw_size_range=None,
                    ink_thickness_range=thickness_range,
                    ink_brightness_change=[0],
                    ink_skeletonize=0,
                    ink_skeletonize_iterations_range=(1, 1),
                    ink_text=None,
                    ink_text_font=None,
                    ink_text_rotate_range=None,
                    ink_lines_coordinates=lines_coordinates,
                    ink_lines_stroke_count_range=(1, 1),
                )
                
                image = ink_generator.generate_ink()
                
            # Restore alpha channel if needed
            if has_alpha:
                image = np.dstack((image, image_alpha))
                
            # Update the layer's image
            layer.image = image
            
        return meta