import random

import cv2
import numpy as np

from synthdocs.components.component import Component


class ReflectedLight(Component):
    """Creates reflected light effect by drawing ellipses of different brightness."""

    def __init__(
        self,
        reflected_light_smoothness=0.8,
        reflected_light_internal_radius_range=(0.0, 0.2),
        reflected_light_external_radius_range=(0.1, 0.8),
        reflected_light_minor_major_ratio_range=(0.9, 1.0),
        reflected_light_color=(255, 255, 255),
        reflected_light_internal_max_brightness_range=(0.9, 1.0),
        reflected_light_external_max_brightness_range=(0.75, 0.9),
        reflected_light_location="random",
        reflected_light_ellipse_angle_range=(0, 360),
        reflected_light_gaussian_kernel_size_range=(5, 310),
        p=1,
    ):
        """Constructor method"""
        super().__init__()
        self.reflected_light_smoothness = reflected_light_smoothness
        self.reflected_light_internal_radius_range = reflected_light_internal_radius_range
        self.reflected_light_external_radius_range = reflected_light_external_radius_range
        self.reflected_light_minor_major_ratio_range = reflected_light_minor_major_ratio_range
        self.reflected_light_color = reflected_light_color
        self.reflected_light_internal_max_brightness_range = reflected_light_internal_max_brightness_range
        self.reflected_light_external_max_brightness_range = reflected_light_external_max_brightness_range
        self.reflected_light_location = reflected_light_location
        self.reflected_light_ellipse_angle_range = reflected_light_ellipse_angle_range
        self.reflected_light_gaussian_kernel_size_range = reflected_light_gaussian_kernel_size_range
        self.p = p

    def sample(self, meta=None):
        """Sample random parameters for this effect.
        
        :param meta: Optional metadata dictionary with parameters to use
        :type meta: dict, optional
        :return: Metadata dictionary with sampled parameters
        :rtype: dict
        """
        if meta is None:
            meta = {}
            
        # Check if we should run based on probability
        if random.random() > self.p:
            meta["run"] = False
            return meta
            
        meta["run"] = True
        
        # Sample reflected light color
        if self.reflected_light_color == "random":
            meta["reflected_light_color"] = (random.randint(1, 255), random.randint(1, 255), random.randint(1, 255))
        else:
            # Ensure minimum color value is 1
            meta["reflected_light_color"] = (
                max(1, self.reflected_light_color[0]),
                max(1, self.reflected_light_color[1]),
                max(1, self.reflected_light_color[2]),
            )
        
        # Sample brightness parameters
        meta["reflected_light_internal_max_brightness"] = random.uniform(
            self.reflected_light_internal_max_brightness_range[0],
            self.reflected_light_internal_max_brightness_range[1]
        )
        meta["reflected_light_external_max_brightness"] = random.uniform(
            self.reflected_light_external_max_brightness_range[0],
            self.reflected_light_external_max_brightness_range[1]
        )
        
        # Sample ellipse parameters
        meta["reflected_light_minor_major_ratio"] = random.uniform(
            self.reflected_light_minor_major_ratio_range[0],
            self.reflected_light_minor_major_ratio_range[1]
        )
        
        meta["reflected_light_ellipse_angle"] = random.randint(
            self.reflected_light_ellipse_angle_range[0],
            self.reflected_light_ellipse_angle_range[1]
        )
        
        # Sample Gaussian kernel size
        meta["reflected_light_gaussian_kernel_value"] = random.randint(
            self.reflected_light_gaussian_kernel_size_range[0],
            self.reflected_light_gaussian_kernel_size_range[1]
        )
        # Ensure kernel size is odd
        if meta["reflected_light_gaussian_kernel_value"] % 2 == 0:
            meta["reflected_light_gaussian_kernel_value"] += 1
            
        # Store other parameters for later use
        meta["reflected_light_smoothness"] = self.reflected_light_smoothness
        meta["reflected_light_internal_radius_range"] = self.reflected_light_internal_radius_range
        meta["reflected_light_external_radius_range"] = self.reflected_light_external_radius_range
        meta["reflected_light_location"] = self.reflected_light_location
            
        return meta

    def apply(self, layers, meta=None):
        """Apply the reflected light effect to the layers.
        
        :param layers: The layers to apply the effect to
        :type layers: list
        :param meta: Optional metadata with parameters
        :type meta: dict, optional
        :return: Updated metadata
        :rtype: dict
        """
        meta = self.sample(meta)
        
        # Skip processing if run is False
        if not meta.get("run", True):
            return meta
        
        # Extract parameters from metadata
        reflected_light_color = meta["reflected_light_color"]
        reflected_light_internal_max_brightness = meta["reflected_light_internal_max_brightness"]
        reflected_light_external_max_brightness = meta["reflected_light_external_max_brightness"]
        reflected_light_minor_major_ratio = meta["reflected_light_minor_major_ratio"]
        reflected_light_ellipse_angle = meta["reflected_light_ellipse_angle"]
        reflected_light_gaussian_kernel_value = meta["reflected_light_gaussian_kernel_value"]
        reflected_light_smoothness = meta["reflected_light_smoothness"]
        reflected_light_internal_radius_range = meta["reflected_light_internal_radius_range"]
        reflected_light_external_radius_range = meta["reflected_light_external_radius_range"]
        reflected_light_location = meta["reflected_light_location"]
        
        for layer in layers:
            try:
                # Get a copy of the image to work with
                image = layer.image.copy()

                image = image.astype(np.uint8)
                
                # Check for image format and alpha channel
                has_alpha = False
                is_gray = False
                image_alpha = None
                
                if len(image.shape) > 2:
                    if image.shape[2] == 4:
                        has_alpha = True
                        image, image_alpha = image[:, :, :3], image[:, :, 3]
                else:
                    is_gray = True
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                
                # Get image dimensions
                ysize, xsize = image.shape[:2]
                min_size = min(ysize, xsize)
                
                # Ensure color values are appropriate
                single_color_pixel = np.full((1, 1, 3), fill_value=reflected_light_color, dtype="uint8")
                reflected_light_color_hsv = cv2.cvtColor(single_color_pixel, cv2.COLOR_BGR2HSV)
                reflected_light_color_hsv[:, :, 2][0, 0] = max(64, reflected_light_color_hsv[:, :, 2][0, 0])
                single_color_pixel = cv2.cvtColor(reflected_light_color_hsv, cv2.COLOR_HSV2BGR)
                reflected_light_color = [int(color) for color in single_color_pixel[0, 0]]
                
                # Calculate radius values based on image size
                if reflected_light_internal_radius_range[1] <= 1 and isinstance(
                    reflected_light_internal_radius_range[1],
                    float,
                ):
                    internal_radius_range = [
                        reflected_light_internal_radius_range[0] * min_size,
                        reflected_light_internal_radius_range[1] * min_size,
                    ]
                else:
                    internal_radius_range = reflected_light_internal_radius_range
                
                if reflected_light_external_radius_range[1] <= 1 and isinstance(
                    reflected_light_external_radius_range[1],
                    float,
                ):
                    external_radius_range = [
                        reflected_light_external_radius_range[0] * min_size,
                        reflected_light_external_radius_range[1] * min_size,
                    ]
                else:
                    external_radius_range = reflected_light_external_radius_range
                
                reflected_light_internal_radius = random.randint(
                    int(internal_radius_range[0]),
                    int(internal_radius_range[1]),
                )
                
                reflected_light_external_radius = random.randint(
                    int(external_radius_range[0]),
                    int(external_radius_range[1]),
                )
                
                # Determine light center position
                if reflected_light_location == "random":
                    reflected_light_center_x = random.randint(0, xsize - 1)
                    reflected_light_center_y = random.randint(0, ysize - 1)
                else:
                    # Generate light location based on image size
                    if reflected_light_location[0] <= 1 and isinstance(reflected_light_location[0], float):
                        reflected_light_center_x = int(reflected_light_location[0] * xsize)
                    else:
                        reflected_light_center_x = int(reflected_light_location[0])
                        
                    if reflected_light_location[1] <= 1 and isinstance(reflected_light_location[1], float):
                        reflected_light_center_y = int(reflected_light_location[1] * ysize)
                    else:
                        reflected_light_center_y = int(reflected_light_location[1])
                
                # Initial ellipse axes (major and minor radius)
                reflected_light_axes = [
                    reflected_light_external_radius + reflected_light_internal_radius,
                    int(
                        (reflected_light_external_radius + reflected_light_internal_radius)
                        * reflected_light_minor_major_ratio
                    ),
                ]
                
                # Create background for the light effect
                image_background = np.zeros_like(image, dtype="uint8")
                
                # Draw concentric ellipses with varying brightness to create light effect
                reflected_light_alpha = reflected_light_external_max_brightness - 1
                # reflected_light_alpha = 0.2

                # Compute parameters for drawing steps
                total_diameter = int(reflected_light_external_radius * reflected_light_minor_major_ratio)
                smooth_threshold = 50
                smoothness = max(1, smooth_threshold - (smooth_threshold * reflected_light_smoothness))
                total_length = int(total_diameter / smoothness)
                step_length = max(1, int(total_diameter / total_length))
                step_alpha = 1 / max(1, total_length)
                
                # Create random pattern for realistic light effect
                image_random = np.random.uniform(0, 1, size=image.shape[:2])
                
                # Draw external ellipse area with decreasing diameter and increasing brightness
                axes_copy = reflected_light_axes.copy()
                current_diameter = total_diameter
                
                while True:
                    current_reflected_light_alpha = reflected_light_alpha
                    # Clip alpha between 0 and 1
                    current_reflected_light_alpha = np.clip(current_reflected_light_alpha, 0.0, 1.0)
                    
                    # Create temporary image for current ellipse
                    image_background_new = np.zeros_like(image, dtype="uint8")
                    
                    # Draw ellipse
                    cv2.ellipse(
                        image_background_new,
                        (reflected_light_center_x, reflected_light_center_y),
                        axes_copy,
                        reflected_light_ellipse_angle,
                        0,
                        360,
                        reflected_light_color,
                        step_length,
                    )
                    
                    # Add noise to make the light look more natural
                    indices = image_random > min(1.0, current_reflected_light_alpha)
                    for i in range(3):
                        image_background_new[:, :, i][indices] = 0
                    
                    # Merge current ellipse with main background
                    indices = np.logical_and(image_background_new > 0, image_background == (0, 0, 0))
                    image_background[indices] = image_background_new[indices] * current_reflected_light_alpha
                    
                    # Break when we reach the internal radius
                    if current_diameter <= reflected_light_internal_radius:
                        break
                        
                    # For last iteration, adjust step length to exactly reach internal radius
                    if current_diameter - step_length < reflected_light_internal_radius:
                        step_length = current_diameter - reflected_light_internal_radius
                        current_diameter = reflected_light_internal_radius
                    else:
                        current_diameter -= step_length
                    
                    # Update parameters for next iteration
                    reflected_light_alpha += step_alpha
                    axes_copy = [axes_copy[0] - step_length, axes_copy[1] - step_length]
                
                # Draw internal ellipse with higher brightness if needed
                if reflected_light_internal_radius > 0:
                    image_background_new = np.zeros_like(image, dtype="uint8")
                    
                    # Draw inner ellipse
                    cv2.ellipse(
                        image_background_new,
                        (reflected_light_center_x, reflected_light_center_y),
                        axes_copy,  # Use the final axes size from the loop above
                        reflected_light_ellipse_angle,
                        0,
                        360,
                        reflected_light_color,
                        -1,  # Fill the ellipse
                    )
                    
                    # Merge with main background
                    indices = np.logical_and(image_background_new > 0, image_background == (0, 0, 0))
                    image_background[indices] = image_background_new[indices] * reflected_light_internal_max_brightness
                
                # Apply Gaussian blur to make the light effect more realistic
                try:
                    # Limit kernel size to avoid errors
                    kernel_size = min(reflected_light_gaussian_kernel_value, min(image.shape[0], image.shape[1]) - 1)
                    # Ensure kernel size is odd
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    
                    image_background = cv2.GaussianBlur(
                        image_background,
                        (kernel_size, kernel_size),
                        0,
                    )
                except Exception as e:
                    print(f"Error applying Gaussian blur: {e}")
                
                # Add the light effect to the original image
                image_output = cv2.addWeighted(
                    image_background,
                    1,
                    image,
                    1,
                    0,
                )
                
                # Convert back to original format
                if is_gray:
                    image_output = cv2.cvtColor(image_output, cv2.COLOR_BGR2GRAY)
                if has_alpha:
                    image_output = np.dstack((image_output, image_alpha))
                
                # Update layer with processed image
                layer.image = image_output
                
            except Exception as e:
                print(f"Error applying reflected light effect: {e}")
                # Keep original image if processing fails
        
        return meta