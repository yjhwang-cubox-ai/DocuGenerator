import random

import cv2
import numpy as np

from synthtiger.components.component import Component


class InkColorSwap(Component):
    """Swap color of ink in the image based on detected ink contours.

    :param ink_swap_color: The swapping color (in BGR) of the effect.
    :type ink_swap_color: tuple, optional
    :param ink_swap_sequence_number_range: Pair of ints determing the consecutive swapping number in the detected contours.
            Use "-1" to swap color for all detected contours.
    :type ink_swap_sequence_number_range: tuple, optional
    :param ink_swap_min_width_range: Pair of ints/floats determining the minimum width of the contour.
            If the value is within the range of 0.0 to 1.0 and the value is float,
            the minimum width will be scaled by image width:
            min width (int) = image width  * min width (float and 0.0 - 1.0)
    :type ink_swap_min_width_range: tuple, optional
    :param ink_swap_max_width_range: Pair of ints/floats determining the maximum width of the contour.
            If the value is within the range of 0.0 to 1.0 and the value is float,
            the maximum width will be scaled by image width:
            max width (int) = image width  * max width (float and 0.0 - 1.0)
    :type ink_swap_max_width_range: tuple, optional
    :param ink_swap_min_height_range: Pair of ints/floats determining the minimum height of the contour.
            If the value is within the range of 0.0 to 1.0 and the value is float,
            the minimum height will be scaled by image height:
            min height (int) = image height  * min height (float and 0.0 - 1.0)
    :type ink_swap_min_height_range: tuple, optional
    :param ink_swap_max_height_range: Pair of ints/floats determining the maximum height of the contour.
            If the value is within the range of 0.0 to 1.0 and the value is float,
            the maximum height will be scaled by image height:
            max height (int) = image height  * max height (float and 0.0 - 1.0)
    :type ink_swap_max_height_range: tuple, optional
    :param ink_swap_min_area_range: Pair of ints/floats determining the minimum area of the contour.
            If the value is within the range of 0.0 to 1.0 and the value is float,
            the minimum area will be scaled by image area:
            min area (int) = image area  * min area (float and 0.0 - 1.0)
    :type ink_swap_min_area_range: tuple, optional
    :param ink_swap_max_area_range: Pair of ints/floats determining the maximum area of the contour.
            If the value is within the range of 0.0 to 1.0 and the value is float,
            the maximum area will be scaled by image area:
            max area (int) = image area  * max area (float and 0.0 - 1.0)
    :type ink_swap_max_area_range: tuple, optional
    """

    def __init__(
        self,
        ink_swap_color="random",
        ink_swap_sequence_number_range=(5, 10),
        ink_swap_min_width_range=(2, 3),
        ink_swap_max_width_range=(100, 120),
        ink_swap_min_height_range=(2, 3),
        ink_swap_max_height_range=(100, 120),
        ink_swap_min_area_range=(10, 20),
        ink_swap_max_area_range=(400, 500),
    ):
        """Constructor method"""
        super().__init__()
        self.ink_swap_color = ink_swap_color
        self.ink_swap_sequence_number_range = ink_swap_sequence_number_range
        self.ink_swap_min_width_range = ink_swap_min_width_range
        self.ink_swap_max_width_range = ink_swap_max_width_range
        self.ink_swap_min_height_range = ink_swap_min_height_range
        self.ink_swap_max_height_range = ink_swap_max_height_range
        self.ink_swap_min_area_range = ink_swap_min_area_range
        self.ink_swap_max_area_range = ink_swap_max_area_range

    def sample(self, meta=None):
        if meta is None:
            meta = {}
            
        # Generate min_width
        min_width = meta.get("min_width", None)
        if min_width is None:
            if self.ink_swap_min_width_range[0] <= 1.0 and isinstance(self.ink_swap_min_width_range[0], float):
                min_width = {"is_scaled": True, "value": random.uniform(
                    self.ink_swap_min_width_range[0],
                    self.ink_swap_min_width_range[1],
                )}
            else:
                min_width = {"is_scaled": False, "value": random.randint(
                    self.ink_swap_min_width_range[0], 
                    self.ink_swap_min_width_range[1]
                )}
        
        max_width = meta.get("max_width", None)
        if max_width is None:
            if self.ink_swap_max_width_range[0] <= 1.0 and isinstance(self.ink_swap_max_width_range[0], float):
                max_width = {"is_scaled": True, "value": random.uniform(
                    self.ink_swap_max_width_range[0],
                    self.ink_swap_max_width_range[1],
                )}
            else:
                max_width = {"is_scaled": False, "value": random.randint(
                    self.ink_swap_max_width_range[0], 
                    self.ink_swap_max_width_range[1]
                )}
        
        # Height parameters
        min_height = meta.get("min_height", None)
        if min_height is None:
            if self.ink_swap_min_height_range[0] <= 1.0 and isinstance(self.ink_swap_min_height_range[0], float):
                min_height = {"is_scaled": True, "value": random.uniform(
                    self.ink_swap_min_height_range[0],
                    self.ink_swap_min_height_range[1],
                )}
            else:
                min_height = {"is_scaled": False, "value": random.randint(
                    self.ink_swap_min_height_range[0], 
                    self.ink_swap_min_height_range[1]
                )}
        
        max_height = meta.get("max_height", None)
        if max_height is None:
            if self.ink_swap_max_height_range[0] <= 1.0 and isinstance(self.ink_swap_max_height_range[0], float):
                max_height = {"is_scaled": True, "value": random.uniform(
                    self.ink_swap_max_height_range[0],
                    self.ink_swap_max_height_range[1],
                )}
            else:
                max_height = {"is_scaled": False, "value": random.randint(
                    self.ink_swap_max_height_range[0], 
                    self.ink_swap_max_height_range[1]
                )}
        
        # Area parameters
        min_area = meta.get("min_area", None)
        if min_area is None:
            if self.ink_swap_min_area_range[0] <= 1.0 and isinstance(self.ink_swap_min_area_range[0], float):
                min_area = {"is_scaled": True, "value": random.uniform(
                    self.ink_swap_min_area_range[0],
                    self.ink_swap_min_area_range[1],
                )}
            else:
                min_area = {"is_scaled": False, "value": random.randint(
                    self.ink_swap_min_area_range[0], 
                    self.ink_swap_min_area_range[1]
                )}
        
        max_area = meta.get("max_area", None)
        if max_area is None:
            if self.ink_swap_max_area_range[0] <= 1.0 and isinstance(self.ink_swap_max_area_range[0], float):
                max_area = {"is_scaled": True, "value": random.uniform(
                    self.ink_swap_max_area_range[0],
                    self.ink_swap_max_area_range[1],
                )}
            else:
                max_area = {"is_scaled": False, "value": random.randint(
                    self.ink_swap_max_area_range[0], 
                    self.ink_swap_max_area_range[1]
                )}
        
        # Sequence number
        ink_swap_sequence_number = meta.get("ink_swap_sequence_number", random.randint(
            self.ink_swap_sequence_number_range[0],
            self.ink_swap_sequence_number_range[1],
        ))
        
        # Color
        ink_swap_color = meta.get("ink_swap_color", None)
        if ink_swap_color is None:
            if self.ink_swap_color == "random":
                ink_swap_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            else:
                ink_swap_color = self.ink_swap_color
        
        # Create the metadata
        meta = {
            "min_width": min_width,
            "max_width": max_width,
            "min_height": min_height,
            "max_height": max_height,
            "min_area": min_area,
            "max_area": max_area,
            "ink_swap_sequence_number": ink_swap_sequence_number,
            "ink_swap_color": ink_swap_color,
        }
        
        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        
        for layer in layers:
            image = layer.image.copy().astype(np.uint8)
            
            # get image size and area
            ysize, xsize = image.shape[:2]
            image_area = ysize * xsize
            
            # Check if image is grayscale and convert if needed
            if len(image.shape) > 2 and image.shape[2] >= 3:
                is_gray = 0
            else:
                is_gray = 1
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            # Generate mask
            image_mask = np.zeros((ysize, xsize), dtype="uint8")
            
            # Calculate actual width, height, area parameters based on image size
            min_width_data = meta["min_width"]
            if min_width_data["is_scaled"]:
                min_width = int(min_width_data["value"] * xsize)
            else:
                min_width = min_width_data["value"]
                
            max_width_data = meta["max_width"]
            if max_width_data["is_scaled"]:
                max_width = int(max_width_data["value"] * xsize)
            else:
                max_width = max_width_data["value"]
                
            min_height_data = meta["min_height"]
            if min_height_data["is_scaled"]:
                min_height = int(min_height_data["value"] * ysize)
            else:
                min_height = min_height_data["value"]
                
            max_height_data = meta["max_height"]
            if max_height_data["is_scaled"]:
                max_height = int(max_height_data["value"] * ysize)
            else:
                max_height = max_height_data["value"]
                
            min_area_data = meta["min_area"]
            if min_area_data["is_scaled"]:
                min_area = int(min_area_data["value"] * image_area)
            else:
                min_area = min_area_data["value"]
                
            max_area_data = meta["max_area"]
            if max_area_data["is_scaled"]:
                max_area = int(max_area_data["value"] * image_area)
            else:
                max_area = max_area_data["value"]
            
            # Get sequence number and color
            ink_swap_sequence_number = meta["ink_swap_sequence_number"]
            ink_swap_color = meta["ink_swap_color"]
            
            # Convert input image to gray
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # float32 타입인 경우 uint8로 변환
            # if image_gray.dtype == np.float32:
            #     image_gray = (image_gray * 255).astype(np.uint8)
            
            # Convert image into binary
            _, image_binary = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            
            
            # Find contours of image
            contours, _ = cv2.findContours(
                image_binary,
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            
            color_mode = 1
            
            # Draw mask
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                if (
                    w < max_width
                    and w > min_width
                    and h < max_height
                    and h > min_height
                    and area < max_area
                    and area > min_area
                ):
                    # Draw contour for swap color
                    if color_mode:
                        cv2.drawContours(image_mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
                    
                    # Reduce count for contour, and change color when count <= 0
                    if ink_swap_sequence_number == 0:
                        ink_swap_sequence_number = random.randint(
                            self.ink_swap_sequence_number_range[0],
                            self.ink_swap_sequence_number_range[1],
                        )
                        color_mode = 1 - color_mode
                    elif ink_swap_sequence_number != -1:
                        ink_swap_sequence_number -= 1
            
            # Add alpha value if needed
            if image.shape[2] == 4:
                ink_swap_color = (ink_swap_color[0], ink_swap_color[1], ink_swap_color[2], 255)
            
            # Create a mask of swap color
            image_color = np.full_like(image, fill_value=ink_swap_color, dtype="uint8")
            
            # Update alpha if needed
            if image.shape[2] == 4:
                image_color[:, :, 3] = image[:, :, 3].copy()
            
            # Ensure both images are of the same type before blending
            image = image.astype(np.uint8)  # Ensure image is uint8
            image_color = image_color.astype(np.uint8)  # Ensure image_color is uint8
            
            # Blend image with swap color
            image_color = cv2.addWeighted(image, 1.0, image_color, 1.0, 0)
            
            # Update image to blended image in the contour area
            image[image_mask > 0] = image_color[image_mask > 0]
            
            # Return image follows the input image color channel
            if is_gray:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
            # Update the layer's image
            layer.image = image
            
        return meta