import os
import random
import shutil
from glob import glob

import matplotlib
import numpy as np
import requests

from synthdocs.components.component import Component
from augraphy.utilities.inkgenerator import InkGenerator


class Scribbles(Component):
    """Applies scribbles to image.

    :param scribbles_type: Types of scribbles, choose from "random", "lines" or "text".
    :type scribbles_type: string, optional
    :param scribbles_ink: Types of scribbles ink, choose from "random", "pencil", "pen" or "marker".
    :type scribbles_ink: string, optional
    :param scribbles_location: Tuple of ints or floats (x,y) determining location of scribbles effect
           or use "random" for random location.
           The value will be in percentage of the image size if the value is float and in between 0 - 1:
           x (int) = image width  * x (float and 0 - 1);
           y (int) = image height * y (float and 0 - 1)
    :type scribbles_location: tuple or string, optional
    :param scribbles_size_range: Pair of floats determining the range for
           the size of the scribble to be created.
    :type scribbles_size_range: tuple, optional
    :param scribbles_count_range: Pair of floats determining the range for
           the number of scribbles to create.
    :type scribbles_count_range: tuple, optional
    :param scribbles_thickness_range: Pair of floats determining the range for
           the size of the scribbles to create.
    :type scribbles_thickness_range: tuple, optional
    :param scribbles_brightness_change: A list of value change for the brightness of
           the strokes. Default 128 creates a graphite-like appearance.
           32 creates a charcoal-like appearance.
           If more than one value is provided, the final value will be randomly selected.
    :type scribbles_brightness_change: list, optional
    :param scribbles_skeletonize: Flag to enable skeletonization effect.
    :type scribbles_skeletonize: int, optional
    :param scribbles_skeletonize_iterations: Tuple of ints determing number of skeletonization iterations.
    :type scribbles_skeletonize_iterations: tuple, optional
    :param scribbles_color: Tuple of ints (BGR) determining the color of scribbles, or use "random" for random color.
    :type scribbles_color: tuple, optional
    :param scribbles_text: Text value for "text" based scribbles.
    :type scribbles_text: string, optional
    :param scribbles_text_font: Font types for "text" based scribbles.
            It can be the path to the ttf file, a path to the folder contains ttf files,
            an url to the ttf file, or simply "random" to use default randomized font types.
    :type scribbles_text_font: string, optional
    :param scribbles_text_rotate_range: Tuple of ints to determine rotation angle of "text" based scribbles.
    :type scribbles_text_rotate_range: tuple, optional
    :param scribbles_lines_stroke_count_range: Pair of floats determining the range for
           the number of strokes to create in each scribble.
    :type scribbles_lines_stroke_count_range: tuple, optional
    :param p: Probability of this effect being applied.
    :type p: float, optional
    """

    def __init__(
        self,
        scribbles_type="random",
        scribbles_ink="random",
        scribbles_location="random",
        scribbles_size_range=(400, 600),
        scribbles_count_range=(1, 6),
        scribbles_thickness_range=(1, 3),
        scribbles_brightness_change=[32, 64, 128],
        scribbles_skeletonize=0,
        scribbles_skeletonize_iterations=(2, 3),
        scribbles_color="random",
        scribbles_text="random",
        scribbles_text_font="random",
        scribbles_text_rotate_range=(0, 360),
        scribbles_lines_stroke_count_range=(1, 6),
        p=1,
    ):
        """Constructor method"""
        super().__init__()
        self.scribbles_type = scribbles_type
        self.scribbles_ink = scribbles_ink
        self.scribbles_location = scribbles_location
        self.scribbles_size_range = scribbles_size_range
        self.scribbles_count_range = scribbles_count_range
        self.scribbles_thickness_range = scribbles_thickness_range
        self.scribbles_brightness_change = scribbles_brightness_change
        self.scribbles_skeletonize = scribbles_skeletonize
        self.scribbles_skeletonize_iterations = scribbles_skeletonize_iterations
        self.scribbles_color = scribbles_color
        self.scribbles_text = scribbles_text
        self.scribbles_text_font = scribbles_text_font
        self.scribbles_text_rotate_range = scribbles_text_rotate_range
        self.scribbles_lines_stroke_count_range = scribbles_lines_stroke_count_range
        self.p = p
        self.fonts_directory = "fonts/"

    def setup_fonts(self):
        """Set up font files for text-based scribbles.
        
        :return: List of font files
        :rtype: list
        """
        # Create fonts directory
        os.makedirs(self.fonts_directory, exist_ok=True)

        if self.scribbles_text_font != "random":
            # Check if it is a path to ttf file
            if os.path.isfile(self.scribbles_text_font):
                if self.scribbles_text_font.endswith("ttf"):
                    # Remove all existing file
                    shutil.rmtree(self.fonts_directory)
                    os.makedirs(self.fonts_directory, exist_ok=True)

                    # Move the ttf file into fonts directory
                    shutil.copy(self.scribbles_text_font, self.fonts_directory)
                # If the path is not valid, set to default random fonts
                else:
                    print("Invalid font.ttf file!")
                    self.scribbles_text_font = "random"

            # Check if it is a folder
            elif os.path.isdir(self.scribbles_text_font):
                file_list = glob(self.scribbles_text_font + "/*.ttf")
                if len(file_list) > 0:
                    self.fonts_directory = self.scribbles_text_font
                else:
                    print("No font.ttf file in the directory!")
                    self.scribbles_text_font = "random"

            # Check if it is a valid url
            else:
                try:
                    # Remove all existing file
                    shutil.rmtree(self.fonts_directory)
                    os.makedirs(self.fonts_directory, exist_ok=True)

                    # Download new ttf file
                    response = requests.get(self.scribbles_text_font)
                    open("fonts/font_type.zip", "wb").write(response.content)
                    shutil.unpack_archive("fonts/font_type.zip", self.fonts_directory)
                except Exception:
                    print("Font url is not valid")
                    self.scribbles_text_font = "random"

        # Download random fonts or get it from system fonts
        if self.scribbles_text_font == "random":
            file_list = glob("fonts/*.ttf")
            if len(file_list) < 1:
                # Source: https://www.fontsquirrel.com/fonts/list/tag/handwritten
                urls = [
                    "https://www.fontsquirrel.com/fonts/download/Jinky",
                    "https://www.fontsquirrel.com/fonts/download/Journal",
                    "https://www.fontsquirrel.com/fonts/download/indie-flower",
                ]

                # Choose random font
                url = random.choice(urls)

                # Try to download from url first
                try:
                    # Download from url and unzip them into font directory
                    response = requests.get(url)
                    open("fonts/font_type.zip", "wb").write(response.content)
                    shutil.unpack_archive("fonts/font_type.zip", self.fonts_directory)
                # Get system font if download failed
                except Exception:
                    # From here, looks like this is the only solution to get system fonts
                    # https://stackoverflow.com/questions/65141291/get-a-list-of-all-available-fonts-in-pil
                    system_fonts = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext="ttf")

                    # Move the ttf file into fonts directory
                    shutil.copy(np.random.choice(system_fonts), self.fonts_directory)

        return glob(self.fonts_directory + "/*.ttf")

    def sample(self, meta=None):
        """Sample random parameters for the scribbles effect.
        
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
        if self.scribbles_type == "random":
            scribbles_type = random.choice(["lines", "texts"])
        else:
            scribbles_type = self.scribbles_type

        if self.scribbles_ink == "random":
            scribbles_ink = random.choice(["pencil", "pen", "marker", "highlighter"])
        else:
            scribbles_ink = self.scribbles_ink

        if self.scribbles_skeletonize == "random":
            scribbles_skeletonize = random.choice([0, 1])
        else:
            scribbles_skeletonize = self.scribbles_skeletonize

        if self.scribbles_color == "random":
            scribbles_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        else:
            scribbles_color = self.scribbles_color
            
        # Location will be handled during apply since it depends on the image dimensions
        
        # Build metadata
        meta.update({
            "scribbles_type": scribbles_type,
            "scribbles_ink": scribbles_ink,
            "scribbles_skeletonize": scribbles_skeletonize,
            "scribbles_color": scribbles_color,
            "scribbles_location": self.scribbles_location,
            "scribbles_size_range": self.scribbles_size_range,
            "scribbles_count_range": self.scribbles_count_range,
            "scribbles_thickness_range": self.scribbles_thickness_range,
            "scribbles_brightness_change": self.scribbles_brightness_change,
            "scribbles_skeletonize_iterations": self.scribbles_skeletonize_iterations,
            "scribbles_text": self.scribbles_text,
            "scribbles_text_rotate_range": self.scribbles_text_rotate_range,
            "scribbles_lines_stroke_count_range": self.scribbles_lines_stroke_count_range
        })
        
        return meta

    def apply(self, layers, meta=None):
        """Apply the scribbles effect to layers.
        
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
        scribbles_type = meta["scribbles_type"]
        scribbles_ink = meta["scribbles_ink"]
        scribbles_skeletonize = meta["scribbles_skeletonize"]
        scribbles_color = meta["scribbles_color"]
        scribbles_location = meta["scribbles_location"]
        scribbles_size_range = meta["scribbles_size_range"]
        scribbles_count_range = meta["scribbles_count_range"]
        scribbles_thickness_range = meta["scribbles_thickness_range"]
        scribbles_brightness_change = meta["scribbles_brightness_change"]
        scribbles_skeletonize_iterations = meta["scribbles_skeletonize_iterations"]
        scribbles_text = meta["scribbles_text"]
        scribbles_text_rotate_range = meta["scribbles_text_rotate_range"]
        scribbles_lines_stroke_count_range = meta["scribbles_lines_stroke_count_range"]
        
        # Set up fonts if needed for text-based scribbles
        if scribbles_type == "texts" or scribbles_type == "text":
            fonts_list = self.setup_fonts()
        else:
            fonts_list = []
        
        for layer in layers:
            image = layer.image.copy()

            image = image.astype(np.uint8)
            
            # Handle alpha channel
            has_alpha = 0
            if len(image.shape) > 2 and image.shape[2] == 4:
                has_alpha = 1
                image, image_alpha = image[:, :, :3], image[:, :, 3]
                
            # Handle location
            if scribbles_location != "random":
                ysize, xsize = image.shape[:2]
                target_x, target_y = scribbles_location
                # Check if provided location is float and scale them with target size
                if target_x >= 0 and target_x <= 1 and isinstance(target_x, float):
                    target_x = int(target_x * xsize)
                if target_y >= 0 and target_y <= 1 and isinstance(target_y, float):
                    target_y = int(target_y * ysize)
                location = (target_x, target_y)
            else:
                location = scribbles_location
                
            try:
                # Create an ink generator and generate scribbles
                ink_generator = InkGenerator(
                    ink_type=scribbles_ink,
                    ink_draw_method=scribbles_type,
                    ink_draw_iterations=scribbles_count_range,
                    ink_location=location,
                    ink_background=image,
                    ink_background_size=None,
                    ink_background_color=None,
                    ink_color=scribbles_color,
                    ink_min_brightness=0,
                    ink_min_brightness_value_range=(0, 0),
                    ink_draw_size_range=scribbles_size_range,
                    ink_thickness_range=scribbles_thickness_range,
                    ink_brightness_change=scribbles_brightness_change,
                    ink_skeletonize=scribbles_skeletonize,
                    ink_skeletonize_iterations_range=scribbles_skeletonize_iterations,
                    ink_text=scribbles_text,
                    ink_text_font=fonts_list,
                    ink_text_rotate_range=scribbles_text_rotate_range,
                    ink_lines_coordinates="random",
                    ink_lines_stroke_count_range=scribbles_lines_stroke_count_range,
                )

                image_output = ink_generator.generate_ink()
                
            except Exception as e:
                print(f"Error generating scribbles: {e}")
                image_output = image  # Fallback to original image
                
            # Restore alpha channel if needed
            if has_alpha:
                image_output = np.dstack((image_output, image_alpha))
                
            # Update the layer's image
            layer.image = image_output
            
        return meta