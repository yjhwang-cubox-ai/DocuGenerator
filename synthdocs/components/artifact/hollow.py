import random

import cv2
import numpy as np

from synthdocs.components.component import Component


class Hollow(Component):
    """Creates hollow effect by replacing detected contours with edges.
       The detected contours are removed by using median filter operation.

    :param hollow_median_kernel_value_range: Pair of ints determining the median filter kernel value.
    :type hollow_median_kernel_value_range: tuple, optional
    :param hollow_min_width_range: Pair of ints/floats determining the minimum width of the contour.
            If the value is within the range of 0.0 to 1.0 and the value is float,
            the minimum width will be scaled by image width:
            min width (int) = image width  * min width (float and 0.0 - 1.0)
    :type hollow_min_width_range: tuple, optional
    :param hollow_max_width_range: Pair of ints/floats determining the maximum width of the contour.
            If the value is within the range of 0.0 to 1.0 and the value is float,
            the maximum width will be scaled by image width:
            max width (int) = image width  * max width (float and 0.0 - 1.0)
    :type hollow_max_width_range: tuple, optional
    :param hollow_min_height_range: Pair of ints/floats determining the minimum height of the contour.
            If the value is within the range of 0.0 to 1.0 and the value is float,
            the minimum height will be scaled by image height:
            min height (int) = image height  * min height (float and 0.0 - 1.0)
    :type hollow_min_height_range: tuple, optional
    :param hollow_max_height_range: Pair of ints/floats determining the maximum height of the contour.
            If the value is within the range of 0.0 to 1.0 and the value is float,
            the maximum height will be scaled by image height:
            max height (int) = image height  * max height (float and 0.0 - 1.0)
    :type hollow_max_height_range: tuple, optional
    :param hollow_min_area_range: Pair of ints/floats determining the minimum area of the contour.
            If the value is within the range of 0.0 to 1.0 and the value is float,
            the minimum area will be scaled by image area:
            min area (int) = image area  * min area (float and 0.0 - 1.0)
    :type hollow_min_area_range: tuple, optional
    :param hollow_max_area_range: Pair of ints/floats determining the maximum area of the contour.
            If the value is within the range of 0.0 to 1.0 and the value is float,
            the maximum area will be scaled by image area:
            max area (int) = image area  * max area (float and 0.0 - 1.0)
    :type hollow_max_area_range: tuple, optional
    :param hollow_dilation_kernel_size_range: Pair of ints determining the kernel value of the dilation.
            The dilation affect the final thickness of the hollow efect.
    :type hollow_dilation_kernel_size_range: tuple, optional
    """

    def __init__(
        self,
        hollow_median_kernel_value_range=(71, 101),
        hollow_min_width_range=(1, 2),
        hollow_max_width_range=(150, 200),
        hollow_min_height_range=(1, 2),
        hollow_max_height_range=(150, 200),
        hollow_min_area_range=(10, 20),
        hollow_max_area_range=(2000, 5000),
        hollow_dilation_kernel_size_range=(1, 2),
    ):
        super().__init__()
        self.hollow_median_kernel_value_range = hollow_median_kernel_value_range
        self.hollow_min_width_range = hollow_min_width_range
        self.hollow_max_width_range = hollow_max_width_range
        self.hollow_min_height_range = hollow_min_height_range
        self.hollow_max_height_range = hollow_max_height_range
        self.hollow_min_area_range = hollow_min_area_range
        self.hollow_max_area_range = hollow_max_area_range
        self.hollow_dilation_kernel_size_range = hollow_dilation_kernel_size_range

    def sample(self, meta=None):
        if meta is None:
            meta = {}
        
        # Get upscale factor
        upscale = meta.get("upscale", 2)
        upscale_area = upscale * 2
        
        # Sample median kernel value
        median_kernel_value = meta.get("median_kernel_value", None)
        if median_kernel_value is None:
            median_kernel_value = upscale * random.randint(
                self.hollow_median_kernel_value_range[0],
                self.hollow_median_kernel_value_range[1],
            )
            # median kernel value must be odd
            if not median_kernel_value % 2:
                median_kernel_value += 1
        
        # Sample dilation kernel value
        dilation_kernel_value = meta.get("dilation_kernel_value", None)
        if dilation_kernel_value is None:
            dilation_kernel_value = upscale * random.randint(
                self.hollow_dilation_kernel_size_range[0],
                self.hollow_dilation_kernel_size_range[1],
            )
            
        # Build metadata
        meta = {
            "upscale": upscale,
            "upscale_area": upscale_area,
            "median_kernel_value": median_kernel_value,
            "dilation_kernel_value": dilation_kernel_value,
            "hollow_min_width_range": self.hollow_min_width_range,
            "hollow_max_width_range": self.hollow_max_width_range,
            "hollow_min_height_range": self.hollow_min_height_range,
            "hollow_max_height_range": self.hollow_max_height_range,
            "hollow_min_area_range": self.hollow_min_area_range,
            "hollow_max_area_range": self.hollow_max_area_range
        }
        
        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        
        upscale = meta["upscale"]
        upscale_area = meta["upscale_area"]
        median_kernel_value = meta["median_kernel_value"]
        dilation_kernel_value = meta["dilation_kernel_value"]
        
        for layer in layers:
            image = layer.image.copy().astype(np.uint8)
            
            # Convert and make sure image is color image
            if len(image.shape) > 2:
                is_gray = 0
            else:
                is_gray = 1
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            # Original image size
            ysize, xsize = image.shape[:2]
            uysize, uxsize = ysize * upscale, xsize * upscale
            image_area = ysize * xsize

            # Upscale image to enable a better edge detection
            image = cv2.resize(image, (uxsize, uysize), 0)

            # Init binary image for edge detection purpose
            image_binary = np.zeros((uysize, uxsize), dtype="int32")
            contours = []
            # Get better contours by getting contours from all three channels
            for i in range(3):
                # Get binary of current channel and sum to binary image
                _, image_binary_single_channel = cv2.threshold(
                    image[:, :, i],
                    0,
                    255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                )

                # Sum of binary to get binary images across all channels
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

            # Calculate contour size parameters based on image dimensions
            # Width
            if meta["hollow_min_width_range"][0] <= 1.0 and isinstance(meta["hollow_min_width_range"][0], float):
                min_width = random.randint(
                    int(meta["hollow_min_width_range"][0] * xsize * upscale),
                    int(meta["hollow_min_width_range"][1] * xsize * upscale),
                )
            else:
                min_width = random.randint(
                    meta["hollow_min_width_range"][0] * upscale,
                    meta["hollow_min_width_range"][1] * upscale,
                )
                
            if meta["hollow_max_width_range"][0] <= 1.0 and isinstance(meta["hollow_max_width_range"][0], float):
                max_width = random.randint(
                    int(meta["hollow_max_width_range"][0] * xsize * upscale),
                    int(meta["hollow_max_width_range"][1] * xsize * upscale),
                )
            else:
                max_width = random.randint(
                    meta["hollow_max_width_range"][0] * upscale,
                    meta["hollow_max_width_range"][1] * upscale,
                )

            # Height
            if meta["hollow_min_height_range"][0] <= 1.0 and isinstance(meta["hollow_min_height_range"][0], float):
                min_height = random.randint(
                    int(meta["hollow_min_height_range"][0] * ysize * upscale),
                    int(meta["hollow_min_height_range"][1] * ysize * upscale),
                )
            else:
                min_height = random.randint(
                    meta["hollow_min_height_range"][0] * upscale,
                    meta["hollow_min_width_range"][1] * upscale,
                )
                
            if meta["hollow_max_height_range"][0] <= 1.0 and isinstance(meta["hollow_max_height_range"][0], float):
                max_height = random.randint(
                    int(meta["hollow_max_height_range"][0] * ysize * upscale),
                    int(meta["hollow_max_height_range"][1] * ysize * upscale),
                )
            else:
                max_height = random.randint(
                    meta["hollow_max_height_range"][0] * upscale,
                    meta["hollow_max_height_range"][1] * upscale,
                )

            # Area
            if meta["hollow_min_area_range"][0] <= 1.0 and isinstance(meta["hollow_min_area_range"][0], float):
                min_area = random.randint(
                    int(meta["hollow_min_area_range"][0] * image_area * upscale_area),
                    int(meta["hollow_min_area_range"][1] * image_area * upscale_area),
                )
            else:
                min_area = random.randint(
                    meta["hollow_min_area_range"][0] * upscale_area,
                    meta["hollow_min_area_range"][1] * upscale_area,
                )
                
            if meta["hollow_max_area_range"][0] <= 1.0 and isinstance(meta["hollow_max_area_range"][0], float):
                max_area = random.randint(
                    int(meta["hollow_max_area_range"][0] * image_area * upscale_area),
                    int(meta["hollow_max_area_range"][1] * image_area * upscale_area),
                )
            else:
                max_area = random.randint(
                    meta["hollow_max_area_range"][0] * upscale_area,
                    meta["hollow_max_area_range"][1] * upscale_area,
                )

            # Find contours of image
            image_mask = np.zeros_like(image_binary, dtype="uint8")
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
                    cv2.drawContours(image_mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

            # Apply canny edge detection in binary image
            image_canny_binary = cv2.Canny(image_binary, threshold1=0, threshold2=255)
            # Apply canny edge detection in the contour image
            image_canny_contour = cv2.Canny(image_mask, threshold1=0, threshold2=255)
            # Merge both canny images
            image_canny = np.add(image_canny_binary.astype("int32"), image_canny_contour.astype("int32"))
            image_canny[image_canny > 255] = 255
            image_canny = image_canny.astype("uint8")

            # Apply median filter            
            # Median kernel max value is 255
            image_median = cv2.medianBlur(image, min(255, median_kernel_value))

            # Get background by removing the detected contours
            image_output = image.copy()
            for i in range(3):
                image_output[:, :, i][image_mask > 0] = image_median[:, :, i][image_mask > 0]

            # Create a random mask
            image_random = np.random.randint(0, 255, size=image_mask.shape, dtype="uint8")
            image_random = cv2.GaussianBlur(image_random, (3, 3), 0)

            # Apply dilation in the hollow edges
            dilation_kernel = np.ones((dilation_kernel_value, dilation_kernel_value), np.uint8)
            image_canny = cv2.dilate(image_canny, dilation_kernel, iterations=1)

            # Remove some edge based on the random mask
            image_canny[image_random < 128] = 0

            # Update output with edge image
            for i in range(3):
                image_output[:, :, i][image_canny > 0] = image[:, :, i][image_canny > 0]

            # Downscale to original input size
            image_output = cv2.resize(image_output, (xsize, ysize), 0)

            # Return image follows the input image color channel
            if is_gray:
                image_output = cv2.cvtColor(image_output, cv2.COLOR_BGR2GRAY)

            # Update layer's image
            layer.image = image_output
            
        return meta
