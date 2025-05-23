import random

import cv2
import numpy as np

from synthdocs.utils.lib import smooth
from synthdocs.components.component import Component


class ShadowCast(Component):
    """Emulates shadow effect on the surface of the paper.

    :param shadow_side: Side of the image to apply the shadow effect.
            Choose from "random", "left", "right", "top" or "bottom".
    :type shadow_side: string, optional
    :param shadow_vertices_range: Pair of ints determining the vertices of shadow shape.
            The minumum value is 1, excluding the start point and end point to form a triangle.
    :type shadow_vertices_range: Tuple, optional
    :param shadow_width_range: Pair of values determining the width of shadow effect.
            The width value will be in percentage of the image size if the value is float and in between 0.0 - 1.0:
            shadow_width (int) = image width * shadow_width (float and 0.0 - 1.0)
    :type shadow_width_range: tuple, optional
    :param shadow_height_range: Pair of values determining the maximum height of shadow effect.
            The height value will be in percentage of the image size if the value is float and in between 0.0 - 1.0:
            shadow_height (int) = image height * shadow_height (float and 0.0 - 1.0)
    :type shadow_height_range: tuple, optional
    :param shadow_color: Color of shadow in BGR or use "random" for random color.
    :type shadow_color: tuple, optional
    :param shadow_opacity_range: Pair of floats determining the opacity of the shadow.
            The minimum of opacity is 0.0 and the maximum value of opacity is 1.0.
    :type shadow_opacity_range: tuple, optional
    :param shadow_iterations_range: Pair of ints determining the iteration numbers to apply the shadow effect.
    :type shadow_iterations_range: tuple, optional
    :param shadow_blur_kernel_range: Pair of ints determining the value of the kernel to blur the shadow effect.
    :type shadow_blur_kernel_range: tuple, optional
    :param p: The probability this effect will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        shadow_side="random",
        shadow_vertices_range=(1, 20),
        shadow_width_range=(0.3, 0.8),
        shadow_height_range=(0.3, 0.8),
        shadow_color=(0, 0, 0),
        shadow_opacity_range=(0.2, 0.9),
        shadow_iterations_range=(1, 2),
        shadow_blur_kernel_range=(101, 301),
    ):
        """Constructor method"""
        super().__init__()
        self.shadow_side = shadow_side
        self.shadow_vertices_range = shadow_vertices_range
        self.shadow_width_range = shadow_width_range
        self.shadow_height_range = shadow_height_range
        self.shadow_color = shadow_color
        self.shadow_opacity_range = shadow_opacity_range
        self.shadow_iterations_range = shadow_iterations_range
        self.shadow_blur_kernel_range = shadow_blur_kernel_range

    def generate_shadow_mask(self, image, shadow_color, shadow_width, shadow_height):
        """Generate a shadow mask for the image.
        
        :param image: Input image to generate shadow mask for
        :type image: numpy.ndarray
        :param shadow_color: Color of the shadow (BGR)
        :type shadow_color: tuple
        :param shadow_width: Width of the shadow
        :type shadow_width: int
        :param shadow_height: Height of the shadow
        :type shadow_height: int
        :return: Shadow mask
        :rtype: numpy.ndarray
        """
        ysize, xsize = image.shape[:2]
        shadow_mask = np.full_like(image, fill_value=255, dtype="uint8")

        # Generate shape for the shadow
        xstart = random.randint(0, xsize - shadow_width)
        xend = xstart + shadow_width - 1
        ystart = 0
        yend = 0
        ymax = shadow_height - 1

        # Generate initial points of shape
        number_points = 2 + random.randint(
            max(1, self.shadow_vertices_range[0]),
            max(1, self.shadow_vertices_range[1]),
        )
        points_x = np.linspace(xstart, xend, number_points - 2)
        points_y = [random.randint(ystart, ymax) for _ in points_x]
        points = np.zeros((number_points, 2), dtype="float")

        # Initialize first and last point
        points[0] = [xstart, ystart]
        points[-1] = [xend, yend]
        # Update the rest of the points
        for i in range(number_points - 2):
            points[i + 1] = [points_x[i], points_y[i]]

        # Smoothen shape
        smooth_points = smooth(points, 12).astype("int")
        smooth_points = np.unique(smooth_points, axis=0)
        
        # Update value of the shape
        for point in smooth_points:
            shadow_mask[0 : point[1], point[0]] = shadow_color

        # Blur shadow mask to create a more realistic effect
        shadow_kernel_value = random.randint(self.shadow_blur_kernel_range[0], self.shadow_blur_kernel_range[1])
        # Kernel must be odd
        if not shadow_kernel_value % 2:
            shadow_kernel_value += 1
            
        blur_intensity = random.randint(2, 3)
        ysize_small = int(ysize / blur_intensity)
        xsize_small = int(xsize / blur_intensity)
        shadow_mask = cv2.resize(shadow_mask, (xsize_small, ysize_small), interpolation=cv2.INTER_AREA)
        shadow_mask = cv2.GaussianBlur(shadow_mask, (shadow_kernel_value, shadow_kernel_value), 0)
        shadow_mask = cv2.resize(shadow_mask, (xsize, ysize), interpolation=cv2.INTER_AREA)
        
        return shadow_mask

    def sample(self, meta=None):
        """Sample random parameters for the shadow effect.
        
        :param meta: Optional metadata dictionary with parameters to use.
        :type meta: dict, optional
        :return: Dictionary with sampled parameters.
        :rtype: dict
        """
        if meta is None:
            meta = {}
            
        meta["run"] = True
        
        # Sample shadow side
        if self.shadow_side == "random":
            shadow_side = random.choice(["left", "right", "top", "bottom"])
        else:
            shadow_side = self.shadow_side
            
        # Sample shadow color
        if self.shadow_color == "random":
            shadow_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        else:
            shadow_color = self.shadow_color
            
        # Sample shadow iterations
        shadow_iterations = random.randint(self.shadow_iterations_range[0], self.shadow_iterations_range[1])
        
        # Sample shadow opacity
        shadow_opacity = random.uniform(self.shadow_opacity_range[0], self.shadow_opacity_range[1])
        
        # Build metadata
        meta.update({
            "shadow_side": shadow_side,
            "shadow_color": shadow_color,
            "shadow_iterations": shadow_iterations,
            "shadow_opacity": shadow_opacity,
            "shadow_width_range": self.shadow_width_range,
            "shadow_height_range": self.shadow_height_range,
        })
        
        return meta

    def apply(self, layers, meta=None):
        """Apply the shadow effect to layers.
        
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
        shadow_side = meta["shadow_side"]
        shadow_color = meta["shadow_color"]
        shadow_iterations = meta["shadow_iterations"]
        shadow_opacity = meta["shadow_opacity"]
        shadow_width_range = meta["shadow_width_range"]
        shadow_height_range = meta["shadow_height_range"]
        
        for layer in layers:
            image = layer.image.copy()
            
            image = image.astype(np.uint8)
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
                
            ysize, xsize = image.shape[:2]
            shadow_mask_all = np.full_like(image, fill_value=255, dtype="uint8")
            
            for i in range(shadow_iterations):
                # Check if input width is float, if yes scale the value with image size
                if shadow_width_range[1] <= 1 and isinstance(shadow_width_range[1], float):
                    width_range = [shadow_width_range[0] * xsize, shadow_width_range[1] * xsize]
                else:
                    width_range = shadow_width_range
                shadow_width = random.randint(int(width_range[0]), int(width_range[1]))
                if shadow_width > xsize:
                    shadow_width = xsize

                # Check if input height is float, if yes scale the value with image size
                if shadow_height_range[1] <= 1 and isinstance(shadow_height_range[1], float):
                    height_range = [shadow_height_range[0] * ysize, shadow_height_range[1] * ysize]
                else:
                    height_range = shadow_height_range
                shadow_height = random.randint(int(height_range[0]), int(height_range[1]))
                if shadow_height > ysize:
                    shadow_height = ysize
                    
                # Generate shadow mask
                shadow_mask = self.generate_shadow_mask(image, shadow_color, shadow_width, shadow_height)
                
                # Merge shadow with the accumulator mask
                shadow_mask_all = cv2.multiply(shadow_mask, shadow_mask_all, scale=1 / 255)
                
            # Rotate mask based on the provided image side
            if shadow_side == "right":
                shadow_mask_all = np.rot90(shadow_mask_all, 3)
                shadow_mask_all = cv2.resize(
                    shadow_mask_all,
                    (xsize, ysize),
                    interpolation=cv2.INTER_AREA,
                )
            elif shadow_side == "left":
                shadow_mask_all = np.rot90(shadow_mask_all, 1)
                shadow_mask_all = cv2.resize(
                    shadow_mask_all,
                    (xsize, ysize),
                    interpolation=cv2.INTER_AREA,
                )
            elif shadow_side == "bottom":
                shadow_mask_all = np.rot90(shadow_mask_all, 2)
                shadow_mask_all = cv2.resize(
                    shadow_mask_all,
                    (xsize, ysize),
                    interpolation=cv2.INTER_AREA,
                )
                
            # Update opacity of the shadow
            shadow_mask_all = cv2.addWeighted(
                shadow_mask_all,
                shadow_opacity,
                np.full_like(image, fill_value=255, dtype="uint8"),
                1 - shadow_opacity,
                0,
            )
            
            # Merge image and mask
            image_output = cv2.multiply(image, shadow_mask_all, scale=1 / 255)
            
            # Return image follows the input image color channel
            if is_gray:
                image_output = cv2.cvtColor(image_output, cv2.COLOR_BGR2GRAY)
            if has_alpha:
                image_output = np.dstack((image_output, image_alpha))
                
            # Update the layer's image
            layer.image = image_output
            
        return meta