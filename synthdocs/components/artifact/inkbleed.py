import random

import cv2
import numpy as np

from synthtiger.components.component import Component


class InkBleed(Component):
    """Uses Sobel edge detection to create a mask of all edges, then applies
    random noise to those edges. When followed by a blur, this creates a
    fuzzy edge that emulates an ink bleed effect.

    :param intensity_range: Pair of floats determining the intensity of the
           ink bleeding effect.
    :type intensity: tuple, optionall
    :param kernel_size: Kernel size to determine area of inkbleed effect.
    :type kernel_size: tuple, optional
    :param severity: Severity to determine concentration of inkbleed effect.
    :type severity: tuple, optional
    """

    def __init__(
        self,
        intensity_range=(0.4, 0.7),
        kernel_size=(5, 5),
        severity=(0.3, 0.4),
    ):
        """Constructor method"""
        super().__init__()
        self.intensity_range = intensity_range
        self.kernel_size = kernel_size
        self.severity = severity

    def sobel(self, image):
        """Applies a Sobel filter to an image to detect edges.

        :param image: The image to apply the Sobel filter to.
        :type image: numpy.ndarray
        :return: The edges detected by the Sobel filter.
        :rtype: numpy.ndarray
        """
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
        abs_sobel_x = cv2.convertScaleAbs(sobel_x)
        abs_sobel_y = cv2.convertScaleAbs(sobel_y)
        edges = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return edges

    def sample(self, meta=None):
        if meta is None:
            meta = {}
        
        # Sample intensity
        intensity = meta.get("intensity", None)
        if intensity is None:
            intensity = random.uniform(self.intensity_range[0], self.intensity_range[1])
        
        # Sample severity
        severity = meta.get("severity", None)
        if severity is None:
            severity = random.uniform(self.severity[0], self.severity[1])
        
        # Build metadata
        meta = {
            "intensity": intensity,
            "severity": severity,
            "kernel_size": self.kernel_size
        }
        
        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        
        intensity = meta["intensity"]
        severity = meta["severity"]
        kernel_size = meta["kernel_size"]
        
        for layer in layers:
            image = layer.image.copy().astype(np.uint8)
            
            # Check for alpha channel
            has_alpha = 0
            if len(image.shape) > 2:
                is_gray = 0
                if image.shape[2] == 4:
                    has_alpha = 1
                    image, image_alpha = image[:, :, :3], image[:, :, 3]
            else:
                is_gray = 1
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            # Create output image
            image_output = image.copy()

            # Apply sobel filter and dilate image
            sobelized = self.sobel(image)
            kernel = np.ones(random.choice(kernel_size), dtype="uint8")
            sobelized_dilated = cv2.dilate(sobelized, kernel, iterations=1)

            # Create grayscale from the dilated edge image
            sobelized_dilated_gray = cv2.cvtColor(sobelized_dilated, cv2.COLOR_BGR2GRAY)

            # Dilation on the darker ink area, which is erosion here
            dilated = cv2.erode(image, kernel, iterations=1)

            # Create a random mask
            image_random = np.random.randint(0, 255, size=image.shape[:2]).astype("uint8")

            # Based on the provided severity value, update image edges randomly into the dilated edge image
            severity_value = severity * 255
            indices = np.logical_and(image_random < severity_value, sobelized_dilated_gray > 0)
            image_output[indices] = dilated[indices]

            # Blur image and blend output based on input intensity
            image_output = cv2.GaussianBlur(image_output, (3, 3), 0)
            image_output = cv2.addWeighted(image_output, intensity, image, 1 - intensity, 0)

            # Return image follows the input image color channel
            if is_gray:
                image_output = cv2.cvtColor(image_output, cv2.COLOR_BGR2GRAY)
            if has_alpha:
                image_output = np.dstack((image_output, image_alpha))

            # Update the layer's image
            layer.image = image_output
            
        return meta
