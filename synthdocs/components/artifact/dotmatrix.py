import random

import cv2
import numpy as np
from numba import config
from numba import jit
from PIL import Image

from synthdocs.components.component import Component


class DotMatrix(Component):
    """Creates dot matrix effect by drawing dots of mean color in the detected contours."""

    def __init__(
        self,
        dot_matrix_shape="random",
        dot_matrix_dot_width_range=(3, 19),
        dot_matrix_dot_height_range=(3, 19),
        dot_matrix_min_width_range=(1, 2),
        dot_matrix_max_width_range=(150, 200),
        dot_matrix_min_height_range=(1, 2),
        dot_matrix_max_height_range=(150, 200),
        dot_matrix_min_area_range=(10, 20),
        dot_matrix_max_area_range=(2000, 5000),
        dot_matrix_median_kernel_value_range=(128, 255),
        dot_matrix_gaussian_kernel_value_range=(1, 3),
        dot_matrix_rotate_value_range=(0, 360),
        numba_jit=1,
    ):
        """Constructor method"""
        super().__init__()
        self.dot_matrix_shape = dot_matrix_shape
        self.dot_matrix_dot_width_range = dot_matrix_dot_width_range
        self.dot_matrix_dot_height_range = dot_matrix_dot_height_range
        self.dot_matrix_min_width_range = dot_matrix_min_width_range
        self.dot_matrix_max_width_range = dot_matrix_max_width_range
        self.dot_matrix_min_height_range = dot_matrix_min_height_range
        self.dot_matrix_max_height_range = dot_matrix_max_width_range
        self.dot_matrix_min_area_range = dot_matrix_min_area_range
        self.dot_matrix_max_area_range = dot_matrix_max_area_range
        self.dot_matrix_median_kernel_value_range = dot_matrix_median_kernel_value_range
        self.dot_matrix_gaussian_kernel_value_range = dot_matrix_gaussian_kernel_value_range
        self.dot_matrix_rotate_value_range = dot_matrix_rotate_value_range
        self.numba_jit = numba_jit
        config.DISABLE_JIT = bool(1 - numba_jit)

    @staticmethod
    @jit(nopython=True, cache=True)
    def fill_dot(
        image,
        image_dot_matrix,
        image_dot,
        image_mask,
        dot_matrix_dot_width,
        dot_matrix_dot_height,
        n_dot_x,
        n_dot_y,
        remainder_x,
        remainder_y,
    ):
        """The core function to fill output image with each dot image."""

        # fill in image_dot
        for y in range(n_dot_y):
            cy = y * dot_matrix_dot_height
            for x in range(n_dot_x):
                cx = x * dot_matrix_dot_width
                # non empty contour area
                if np.sum(image_mask[cy : cy + dot_matrix_dot_height, cx : cx + dot_matrix_dot_width]) > 0:
                    # mean of current dot color
                    image_patch = image[cy : cy + dot_matrix_dot_height, cx : cx + dot_matrix_dot_width]
                    dot_color = np.array(
                        [np.mean(image_patch[:, :, 0]), np.mean(image_patch[:, :, 1]), np.mean(image_patch[:, :, 2])],
                    )

                    # indices of shape mapping
                    indices = np.logical_or(
                        np.logical_or(image_dot[:, :, 0], image_dot[:, :, 1]),
                        image_dot[:, :, 2],
                    )

                    # map dot to image
                    for i in range(3):
                        dot_color_patch = ((image_dot / 255.0) * dot_color[i])[:, :, i]
                        dot_matrix_patch = image_dot_matrix[
                            cy : cy + dot_matrix_dot_height,
                            cx : cx + dot_matrix_dot_width,
                            i,
                        ]
                        for y in range(dot_color_patch.shape[0]):
                            for x in range(dot_color_patch.shape[1]):
                                if indices[y, x]:
                                    dot_matrix_patch[y, x] = np.uint8(dot_color_patch[y, x])

        # remaining last column
        if remainder_y > 0:
            for x in range(n_dot_x):
                cx = x * dot_matrix_dot_width
                start_y = n_dot_y * dot_matrix_dot_height
                # non empty contour area
                if np.sum(image_mask[start_y : start_y + remainder_y, cx : cx + dot_matrix_dot_width]) > 0:
                    # mean of current dot color
                    image_patch = image[start_y : start_y + remainder_y, cx : cx + dot_matrix_dot_width]
                    dot_color = np.array(
                        [np.mean(image_patch[:, :, 0]), np.mean(image_patch[:, :, 1]), np.mean(image_patch[:, :, 2])],
                    )

                    # indices of shape mapping
                    indices = np.logical_or(
                        np.logical_or(image_dot[:remainder_y, :, 0], image_dot[:remainder_y, :, 1]),
                        image_dot[:remainder_y, :, 2],
                    )

                    # map dot to image
                    for i in range(3):
                        dot_color_patch = ((image_dot[:remainder_y, :] / 255.0) * dot_color[i])[:, :, i]
                        dot_matrix_patch = image_dot_matrix[
                            start_y : start_y + remainder_y,
                            cx : cx + dot_matrix_dot_width,
                            i,
                        ]
                        for y in range(dot_color_patch.shape[0]):
                            for x in range(dot_color_patch.shape[1]):
                                if indices[y, x]:
                                    dot_matrix_patch[y, x] = np.uint8(dot_color_patch[y, x])

        # remaining last row
        if remainder_x > 0:
            for y in range(n_dot_y):
                cy = y * dot_matrix_dot_height
                start_x = n_dot_x * dot_matrix_dot_width
                # non empty contour area
                if np.sum(image_mask[cy : cy + dot_matrix_dot_height, start_x : start_x + remainder_x]) > 0:
                    # mean of current dot color
                    image_patch = image[cy : cy + dot_matrix_dot_height, start_x : start_x + remainder_x]
                    dot_color = np.array(
                        [np.mean(image_patch[:, :, 0]), np.mean(image_patch[:, :, 1]), np.mean(image_patch[:, :, 2])],
                    )

                    # indices of shape mapping
                    indices = np.logical_or(
                        np.logical_or(image_dot[:, :remainder_x, 0], image_dot[:, :remainder_x, 1]),
                        image_dot[:, :remainder_x, 2],
                    )

                    # map dot to image
                    for i in range(3):
                        dot_color_patch = ((image_dot[:, :remainder_x] / 255.0) * dot_color[i])[:, :, i]
                        dot_matrix_patch = image_dot_matrix[
                            cy : cy + dot_matrix_dot_height,
                            start_x : start_x + remainder_x,
                            i,
                        ]
                        for y in range(dot_color_patch.shape[0]):
                            for x in range(dot_color_patch.shape[1]):
                                if indices[y, x]:
                                    dot_matrix_patch[y, x] = np.uint8(dot_color_patch[y, x])

        # last dot (bottom right)
        if remainder_x and remainder_y > 0:
            if remainder_x > 0:
                length_x = remainder_x
            else:
                length_x = dot_matrix_dot_width
            if remainder_y > 0:
                length_y = remainder_y
            else:
                length_y = dot_matrix_dot_height

            start_x = n_dot_x * dot_matrix_dot_width
            start_y = n_dot_y * dot_matrix_dot_height
            # non empty contour area
            if np.sum(image_mask[start_y : start_y + length_y, start_x : start_x + length_x]) > 0:
                # mean of current dot color
                image_patch = image[start_y : start_y + length_y, start_x : start_x + length_x]
                dot_color = np.array(
                    [np.mean(image_patch[:, :, 0]), np.mean(image_patch[:, :, 1]), np.mean(image_patch[:, :, 2])],
                )

                # indices of shape mapping
                indices = np.logical_or(
                    np.logical_or(image_dot[:length_y, :length_x, 0], image_dot[:length_y, :length_x, 1]),
                    image_dot[:length_y, :length_x, 2],
                )

                # map dot to image
                for i in range(3):
                    dot_color_patch = ((image_dot[:length_y, :length_x] / 255.0) * dot_color[i])[:, :, i]
                    dot_matrix_patch = image_dot_matrix[
                        start_y : start_y + length_y,
                        start_x : start_x + length_x,
                        i,
                    ]
                    for y in range(dot_color_patch.shape[0]):
                        for x in range(dot_color_patch.shape[1]):
                            if indices[y, x]:
                                dot_matrix_patch[y, x] = np.uint8(dot_color_patch[y, x])

    def sample(self, meta=None):
        """Sample random parameters for this effect.
        
        :param meta: Optional metadata dictionary with parameters to use
        :type meta: dict, optional
        :return: Metadata dictionary with sampled parameters
        :rtype: dict
        """
        if meta is None:
            meta = {}
            
        meta["run"] = True
        
        # Choose dot shape
        if self.dot_matrix_shape == "random":
            meta["dot_matrix_shape"] = random.choice(["circle", "rectangle", "triangle", "diamond"])
        else:
            meta["dot_matrix_shape"] = self.dot_matrix_shape
            
        # Sample dot dimensions
        meta["dot_matrix_dot_width"] = random.randint(
            self.dot_matrix_dot_width_range[0],
            self.dot_matrix_dot_width_range[1]
        )
        meta["dot_matrix_dot_height"] = random.randint(
            self.dot_matrix_dot_height_range[0],
            self.dot_matrix_dot_height_range[1]
        )
        
        # Make sure dimensions are odd for better centering
        if meta["dot_matrix_dot_width"] % 2 == 0:
            meta["dot_matrix_dot_width"] += 1
        if meta["dot_matrix_dot_height"] % 2 == 0:
            meta["dot_matrix_dot_height"] += 1
        
        # Sample kernel values
        meta["median_kernel_value"] = random.randint(
            self.dot_matrix_median_kernel_value_range[0],
            self.dot_matrix_median_kernel_value_range[1]
        )
        if meta["median_kernel_value"] % 2 == 0:
            meta["median_kernel_value"] += 1
            
        meta["gaussian_kernel_value"] = random.randint(
            self.dot_matrix_gaussian_kernel_value_range[0],
            self.dot_matrix_gaussian_kernel_value_range[1]
        )
        if meta["gaussian_kernel_value"] % 2 == 0:
            meta["gaussian_kernel_value"] += 1
            
        # Sample rotation angle
        meta["dot_matrix_rotate_value"] = random.randint(
            self.dot_matrix_rotate_value_range[0],
            self.dot_matrix_rotate_value_range[1]
        )
        
        return meta

    def apply(self, layers, meta=None):
        """Apply the dot matrix effect to the layers.
        
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
        
        for layer in layers:
            # Get a copy of the image to work with
            image = layer.image.copy()
            
            # Check and convert image into BGR format if needed
            is_gray = False
            if len(image.shape) < 3:
                is_gray = True
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                
            try:
                # Get image dimensions
                ysize, xsize = image.shape[:2]
                
                # Apply median filter
                median_kernel_value = meta["median_kernel_value"]
                # 1. 이미지가 uint8 타입인지 확인하고 변환
                image = image.astype(np.uint8)

                # 2. 커널 값이 홀수인지 확인 (medianBlur는 홀수 커널 크기만 허용)
                if median_kernel_value % 2 == 0:
                    median_kernel_value += 1

                # 3. 커널 크기가 255를 초과하지 않도록 제한 (OpenCV는 더 작은 값을 선호)
                if median_kernel_value > 255:
                    median_kernel_value = 255

                try:
                    # 4. 이미지 크기에 따라 다른 처리 방법 적용
                    if median_kernel_value > 127:  # 큰 커널 값은 이미지 크기 축소 후 처리
                        scale = 127 / median_kernel_value
                        image_resize = cv2.resize(image, (int(xsize * scale), int(ysize * scale)), interpolation=cv2.INTER_AREA)
                        # 타입과 채널 수 확인
                        if image_resize.dtype != np.uint8:
                            image_resize = image_resize.astype(np.uint8)
                        image_median = cv2.medianBlur(image_resize, 127)  # 더 작은 커널 사용
                        image_median = cv2.resize(image_median, (xsize, ysize), interpolation=cv2.INTER_LINEAR)
                    else:
                        # 이미지 채널 확인
                        if len(image.shape) == 2 or image.shape[2] in [1, 3, 4]:
                            image_median = cv2.medianBlur(image, median_kernel_value)
                        else:  # 지원되지 않는 채널 수일 경우
                            # BGR로 변환하여 처리
                            temp_image = cv2.cvtColor(image, cv2.COLOR_BGR2BGR)
                            image_median = cv2.medianBlur(temp_image, median_kernel_value)
                except Exception as e:
                    print(f"Error in median blur: {e}")
                    # 실패할 경우 원본 이미지 사용
                    image_median = image.copy()
                
                # Init binary image for edge detection purpose
                image_binary = np.zeros((ysize, xsize), dtype="int32")
                contours = []
                
                # Get contours from all three channels
                for i in range(3):
                    # Get binary of current channel
                    _, image_binary_single_channel = cv2.threshold(
                        image[:, :, i],
                        0,
                        255,
                        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                    )
                    
                    # Sum of binary images across all channels
                    image_binary += image_binary_single_channel.astype("int32")
                    
                    # Find contours of current channel
                    contours_single, _ = cv2.findContours(
                        image_binary_single_channel,
                        cv2.RETR_LIST,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )
                    
                    # Merge contours in each channel
                    contours.extend(contours_single)
                
                # Convert back to uint8
                image_binary[image_binary > 255] = 255
                image_binary = image_binary.astype("uint8")
                
                # Find contours of merged binary
                contours_single, _ = cv2.findContours(
                    image_binary,
                    cv2.RETR_LIST,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                # Merge contours of binary image
                contours.extend(contours_single)
                
                # Determine contour width/height/area constraints
                # Width
                if self.dot_matrix_min_width_range[0] <= 1.0 and isinstance(self.dot_matrix_min_width_range[0], float):
                    min_width = random.randint(
                        int(self.dot_matrix_min_width_range[0] * xsize),
                        int(self.dot_matrix_min_width_range[1] * xsize),
                    )
                else:
                    min_width = random.randint(
                        self.dot_matrix_min_width_range[0],
                        self.dot_matrix_min_width_range[1],
                    )
                    
                if self.dot_matrix_max_width_range[0] <= 1.0 and isinstance(self.dot_matrix_max_width_range[0], float):
                    max_width = random.randint(
                        int(self.dot_matrix_max_width_range[0] * xsize),
                        int(self.dot_matrix_max_width_range[1] * xsize),
                    )
                else:
                    max_width = random.randint(
                        self.dot_matrix_max_width_range[0],
                        self.dot_matrix_max_width_range[1],
                    )
                
                # Height
                if self.dot_matrix_min_height_range[0] <= 1.0 and isinstance(self.dot_matrix_min_height_range[0], float):
                    min_height = random.randint(
                        int(self.dot_matrix_min_height_range[0] * ysize),
                        int(self.dot_matrix_min_height_range[1] * ysize),
                    )
                else:
                    min_height = random.randint(
                        self.dot_matrix_min_height_range[0],
                        self.dot_matrix_min_width_range[1],
                    )
                    
                if self.dot_matrix_max_height_range[0] <= 1.0 and isinstance(self.dot_matrix_max_height_range[0], float):
                    max_height = random.randint(
                        int(self.dot_matrix_max_height_range[0] * ysize),
                        int(self.dot_matrix_max_height_range[1] * ysize),
                    )
                else:
                    max_height = random.randint(
                        self.dot_matrix_max_height_range[0],
                        self.dot_matrix_max_height_range[1],
                    )
                
                # Area
                if self.dot_matrix_min_area_range[0] <= 1.0 and isinstance(self.dot_matrix_min_area_range[0], float):
                    min_area = random.randint(
                        int(self.dot_matrix_min_area_range[0] * xsize * ysize),
                        int(self.dot_matrix_min_area_range[1] * xsize * ysize),
                    )
                else:
                    min_area = random.randint(
                        self.dot_matrix_min_area_range[0],
                        self.dot_matrix_min_area_range[1],
                    )
                    
                if self.dot_matrix_max_area_range[0] <= 1.0 and isinstance(self.dot_matrix_max_area_range[0], float):
                    max_area = random.randint(
                        int(self.dot_matrix_max_area_range[0] * xsize * ysize),
                        int(self.dot_matrix_max_area_range[1] * xsize * ysize),
                    )
                else:
                    max_area = random.randint(
                        self.dot_matrix_max_area_range[0],
                        self.dot_matrix_max_area_range[1],
                    )
                
                # Create mask from contours
                image_mask = np.zeros_like(image_binary, dtype="uint8")
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
                        cv2.drawContours(image_mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
                
                # Create dot shape
                dot_matrix_shape = meta["dot_matrix_shape"]
                dot_matrix_dot_width = meta["dot_matrix_dot_width"]
                dot_matrix_dot_height = meta["dot_matrix_dot_height"]
                
                # Adjust dimensions based on shape
                if dot_matrix_shape == "circle":
                    # Min size of dot is 3 pixels for circle
                    dot_matrix_dot_width = max(3, dot_matrix_dot_width)
                    dot_matrix_dot_height = max(3, dot_matrix_dot_height)
                    # Initialize dot image
                    image_dot = np.zeros((dot_matrix_dot_height, dot_matrix_dot_width, 3), dtype="uint8")
                    # Draw shape
                    center_x = int(np.floor(dot_matrix_dot_width / 2))
                    center_y = int(np.floor(dot_matrix_dot_height / 2))
                    radius = int(np.floor(min(dot_matrix_dot_width / 2, dot_matrix_dot_height / 2)))
                    cv2.circle(image_dot, (center_x, center_y), radius, (255, 255, 255), -1)
                
                elif dot_matrix_shape == "rectangle":
                    # Min size of dot is 3 pixels for rectangle
                    dot_matrix_dot_width = max(3, dot_matrix_dot_width)
                    dot_matrix_dot_height = max(3, dot_matrix_dot_height)
                    # Initialize dot image
                    image_dot = np.zeros((dot_matrix_dot_height, dot_matrix_dot_width, 3), dtype="uint8")
                    # Draw shape
                    image_dot[1:-1, 1:-1] = 255
                
                elif dot_matrix_shape == "triangle":
                    # Min size of dot is 5 pixels for triangle
                    dot_matrix_dot_width = max(5, dot_matrix_dot_width)
                    dot_matrix_dot_height = max(5, dot_matrix_dot_height)
                    # Initialize dot image
                    image_dot = np.zeros((dot_matrix_dot_height, dot_matrix_dot_width, 3), dtype="uint8")
                    # Draw shape
                    y0 = 0
                    yn = dot_matrix_dot_height - 1
                    x0 = 0
                    xmid = int(np.floor(dot_matrix_dot_width / 2))
                    xn = dot_matrix_dot_width - 1
                    triangle_points = np.array([(x0, yn), (xmid, y0), (xn, yn)])
                    cv2.drawContours(image_dot, [triangle_points], 0, (255, 255, 255), -1)
                    # Mirror left right for consistent shape
                    if xmid + 1 < image_dot.shape[1]:  # Check to prevent index error
                        image_dot[:, :xmid] = np.fliplr(image_dot[:, xmid + 1:])
                
                elif dot_matrix_shape == "diamond":
                    # Min size of dot is 5 pixels for diamond
                    dot_matrix_dot_width = max(5, dot_matrix_dot_width)
                    dot_matrix_dot_height = max(5, dot_matrix_dot_height)
                    # Initialize dot image
                    image_dot = np.zeros((dot_matrix_dot_height, dot_matrix_dot_width, 3), dtype="uint8")
                    # Draw shape
                    y0 = 0
                    ymid = int(np.floor(dot_matrix_dot_height / 2))
                    yn = dot_matrix_dot_height - 1
                    x0 = 0
                    xmid = int(np.floor(dot_matrix_dot_width / 2))
                    xn = dot_matrix_dot_width - 1
                    triangle_points = np.array([(x0, ymid), (xmid, y0), (xn, ymid)])
                    cv2.drawContours(image_dot, [triangle_points], 0, (255, 255, 255), -1)
                    # Mirror left right for consistent shape
                    if xmid + 1 < image_dot.shape[1]:  # Check to prevent index error
                        image_dot[:, :xmid] = np.fliplr(image_dot[:, xmid + 1:])
                    # Mirror up down to create diamond shape
                    if ymid + 1 < image_dot.shape[0]:  # Check to prevent index error
                        image_dot[ymid:, :] = np.flipud(image_dot[:ymid + 1, :])
                
                # Rotate dot image
                dot_matrix_rotate_value = meta["dot_matrix_rotate_value"]
                if dot_matrix_rotate_value != 0:
                    try:
                        image_dot_PIL = Image.fromarray(image_dot)
                        rotated_image_dot_PIL = image_dot_PIL.rotate(dot_matrix_rotate_value)
                        image_dot = np.array(rotated_image_dot_PIL)
                    except Exception as e:
                        print(f"Error rotating dot image: {e}")
                
                # Calculate dot parameters
                div_x = xsize / dot_matrix_dot_width
                div_y = ysize / dot_matrix_dot_height
                
                n_dot_x = int(np.floor(div_x))
                n_dot_y = int(np.floor(div_y))
                
                remainder_x = xsize % dot_matrix_dot_width
                remainder_y = ysize % dot_matrix_dot_height
                
                # Create output image
                image_dot_matrix = image.copy()
                
                # Apply median image to contour areas
                for i in range(3):
                    image_dot_matrix[:, :, i][image_mask > 0] = image_median[:, :, i][image_mask > 0]
                
                # Fill image with dots
                self.fill_dot(
                    image,
                    image_dot_matrix,
                    image_dot.astype("float"),
                    image_mask,
                    dot_matrix_dot_width,
                    dot_matrix_dot_height,
                    n_dot_x,
                    n_dot_y,
                    remainder_x,
                    remainder_y,
                )
                
                # Apply Gaussian Blur on dot image
                gaussian_kernel_value = meta["gaussian_kernel_value"]
                image_dot_matrix_blur = cv2.GaussianBlur(
                    image_dot_matrix,
                    (gaussian_kernel_value, gaussian_kernel_value),
                    0,
                )
                
                # Perform blur on detected contours only
                image_dot_matrix[image_mask > 0] = image_dot_matrix_blur[image_mask > 0]
                
                # Convert back to grayscale if input was grayscale
                if is_gray:
                    image_dot_matrix = cv2.cvtColor(image_dot_matrix, cv2.COLOR_BGR2GRAY)
                
                # Update layer with processed image
                layer.image = image_dot_matrix
                
            except Exception as e:
                print(f"Error applying dot matrix effect: {e}")
                # Keep original image in case of error
                
        return meta