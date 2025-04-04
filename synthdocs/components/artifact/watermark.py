import random

import cv2
import numpy as np

from synthdocs.utils.lib import rotate_image
from synthdocs.components.component import Component
from synthdocs.utils import *


class WaterMark(Component):
    """
    Add watermark effect into input image.

    :param watermark_word: Word for watermark effect.
    :type watermark_word: string, optional
    :param watermark_font_size: Pair of ints to determine font size of watermark effect.
    :type watermark_font_size: tuple, optional
    :param watermark_font_thickness: Pair of ints to determine thickness of watermark effect.
    :type watermark_font_thickness: tuple, optional
    :param watermark_font_type: Font type of watermark effect.
    :type watermark_font_type: cv2 font types, optional
    :param watermark_rotation: Pair of ints to determine angle of rotation in watermark effect.
    :type watermark_rotation: tuple, optional
    :param watermark_location: Location of watermark effect, select from top, bottom, left, right, center and random.
    :type watermark_location: string, optional
    :param watermark_color: Triplets of ints to determine RGB color of watermark effect.
    :type watermark_color: tuple, optional
    :param watermark_method: Method to overlay watermark foreground into input image.
    :type watermark_method: string, optional
    :param p: The probability this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        watermark_word="random",
        watermark_font_size=(10, 15),
        watermark_font_thickness=(20, 25),
        watermark_font_type=cv2.FONT_HERSHEY_SIMPLEX,
        watermark_rotation=(0, 360),
        watermark_location="random",
        watermark_color="random",
        watermark_method="darken",
    ):
        """Constructor method"""
        super().__init__()
        self.watermark_word = watermark_word
        self.watermark_font_size = watermark_font_size
        self.watermark_font_thickness = watermark_font_thickness
        self.watermark_font_type = watermark_font_type
        self.watermark_rotation = watermark_rotation
        self.watermark_location = watermark_location
        self.watermark_color = watermark_color
        self.watermark_method = watermark_method

    # Create watermark
    def create_watermark(self, word, font_size, font_thickness, font_type, rotation, color):
        """
        Create watermark image.
        
        :param word: Word for watermark effect
        :type word: string
        :param font_size: Font size of watermark
        :type font_size: int
        :param font_thickness: Thickness of watermark font
        :type font_thickness: int
        :param font_type: Font type of watermark
        :type font_type: cv2 font type
        :param rotation: Angle of rotation
        :type rotation: int
        :param color: Color of watermark (BGR)
        :type color: tuple
        :return: Watermark foreground image
        :rtype: numpy.ndarray
        """
        (image_width, image_height), _ = cv2.getTextSize(
            word,
            font_type,
            font_size,
            font_thickness,
        )

        offset = 20 + font_thickness

        # initialize watermark foreground
        watermark_foreground = np.full((image_height + offset, image_width + offset, 3), fill_value=255, dtype="uint8")

        # draw watermark text
        cv2.putText(
            watermark_foreground,
            word,
            (int(offset / 2), int(offset / 2) + image_height),
            font_type,
            font_size,
            color,
            font_thickness,
        )

        # rotate image
        watermark_foreground = rotate_image(watermark_foreground, rotation)

        return watermark_foreground

    # Apply watermark into input image
    def apply_watermark(self, watermark_foreground, image, location, method):
        """
        Apply watermark foreground image to the background image.

        :param watermark_foreground: Foreground image contains the watermark effect.
        :type watermark_foreground: numpy.array (numpy.uint8)
        :param image: The background image.
        :type image: numpy.array (numpy.uint8)
        :param location: Location of watermark (top, bottom, left, right, center)
        :type location: string
        :param method: Method to overlay watermark (overlay, obfuscate)
        :type method: string
        :return: Image with watermark applied
        :rtype: numpy.ndarray
        """

        # resize watermark foreground if the size is larger than input image
        ysize, xsize = image.shape[:2]
        ysize_watermark, xsize_watermark = watermark_foreground.shape[:2]
        if ysize_watermark > ysize or xsize_watermark > xsize:
            watermark_foreground = cv2.resize(watermark_foreground, (xsize, ysize), interpolation=cv2.INTER_AREA)

        if method == "obfuscate":
            # blur image
            image_blurred = cv2.blur(image, (5, 5)).astype("uint8")
            if len(image_blurred.shape) > 2:
                image_blurred = cv2.cvtColor(image_blurred, cv2.COLOR_BGR2GRAY)

            # convert to binary
            _, image_binarized = cv2.threshold(
                image_blurred,
                0,
                255,
                cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV,
            )

            # get kernel for dilation
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))

            # dilate and erode the binary image
            image_dilated = cv2.dilate(
                image_binarized,
                kernel,
                iterations=2,
            )
            image_eroded = cv2.erode(
                image_dilated,
                None,
                iterations=1,
            )

            # get contours
            contours, hierarchy = cv2.findContours(
                image_eroded,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE,
            )

            # complement image
            image_eroded = 255 - image_eroded
            # remove contours in fixed interval
            for i, contour in enumerate(contours):
                if i % 2:
                    x, y, w, h = cv2.boundingRect(contour)
                    image_eroded[y : y + h, x : x + w] = 255

            # create blank image and overlay the foreground only
            image_blank = np.full_like(image, fill_value=255, dtype="uint8")
            ob = OverlayBuilder(
                "darken",
                watermark_foreground,
                image_blank,
                ntimes=1,
                nscales=(1, 1),
                edge=location,
                edge_offset=10,
            )
            new_watermark_foreground = ob.build_overlay()

            # set removed contours to white
            if len(new_watermark_foreground.shape) > 2:
                image_eroded = cv2.cvtColor(image_eroded, cv2.COLOR_GRAY2BGR)

            new_watermark_foreground[image_eroded == 0] = 255
            image[new_watermark_foreground < 255] = 255

            # overlay watermark foreground and input image
            ob = OverlayBuilder(
                "darken",
                new_watermark_foreground,
                image,
                ntimes=1,
                nscales=(1, 1),
                edge="center",
                edge_offset=0,
                alpha=0.5,
            )

        else:
            # overlay watermark foreground and input image
            ob = OverlayBuilder(
                "darken",
                watermark_foreground,
                image,
                ntimes=1,
                nscales=(1, 1),
                edge=location,
                edge_offset=10,
                alpha=0.5,
            )

        return ob.build_overlay()

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
        
        # Sample watermark word
        word = meta.get("watermark_word", None)
        if word is None:
            if self.watermark_word == "random":
                word = random.choice(
                    ["COPY", "VOID", "DRAFT", "CONFIDENTIAL", "UNOFFICIAL", "DO NOT COPY", "SAMPLE", "ORIGINAL"],
                )
            else:
                word = self.watermark_word
        
        # Sample font size
        font_size = meta.get("font_size", random.randint(
            self.watermark_font_size[0], 
            self.watermark_font_size[1]
        ))
        
        # Sample font thickness
        font_thickness = meta.get("font_thickness", random.randint(
            self.watermark_font_thickness[0], 
            self.watermark_font_thickness[1]
        ))
        
        # Sample rotation
        rotation = meta.get("rotation", random.randint(
            self.watermark_rotation[0], 
            self.watermark_rotation[1]
        ))
        
        # Sample color
        color = meta.get("color", None)
        if color is None:
            if self.watermark_color == "random":
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            else:
                color = self.watermark_color
        
        # Sample location
        location = meta.get("location", None)
        if location is None:
            if self.watermark_location == "random":
                location = random.choice(["left", "right", "top", "bottom", "center"])
            else:
                location = self.watermark_location
        
        # Sample method
        method = meta.get("method", None)
        if method is None:
            if self.watermark_method == "random" or self.watermark_method not in ["overlay", "obfuscate"]:
                method = random.choice(["overlay", "obfuscate"])
            else:
                method = self.watermark_method
        
        # Build metadata
        meta.update({
            "word": word,
            "font_size": font_size,
            "font_thickness": font_thickness,
            "font_type": self.watermark_font_type,
            "rotation": rotation,
            "color": color,
            "location": location,
            "method": method
        })
        
        return meta

    def apply(self, layers, meta=None):
        """Apply the WaterMark effect to layers.
        
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
        word = meta["word"]
        font_size = meta["font_size"]
        font_thickness = meta["font_thickness"]
        font_type = meta["font_type"]
        rotation = meta["rotation"]
        color = meta["color"]
        location = meta["location"]
        method = meta["method"]
        
        # Create watermark foreground image
        watermark_foreground = self.create_watermark(
            word, font_size, font_thickness, font_type, rotation, color
        )
        
        for layer in layers:
            image = layer.image.copy()
            
            # Check for alpha channel
            has_alpha = 0
            if len(image.shape) > 2 and image.shape[2] == 4:
                has_alpha = 1
                image, image_alpha = image[:, :, :3], image[:, :, 3]
            
            # Apply watermark to image
            watermark_image = self.apply_watermark(watermark_foreground, image, location, method)
            
            # Restore alpha channel if needed
            if has_alpha:
                watermark_image = np.dstack((watermark_image, image_alpha))
            
            # Update the layer's image
            layer.image = watermark_image
            
        return meta