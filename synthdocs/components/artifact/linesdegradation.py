import random

import cv2
import numpy as np

from synthdocs.components.component import Component


class LinesDegradation(Component):
    """Degrades lines by replacing lines formed by image gradients with a different value.

    :param line_roi: Tuple of 4 (x0, y0, xn, yn) to determine the region of interest of the augmentation effect.
             The value will be in percentage of the image size if the value is float and in between 0.0 - 1.0:
             x0 (int) = image width  * x0 (float and 0.0 - 1.0);
             y0 (int) = image height * y0 (float and 0.0 - 1.0);
             xn (int) = image width  * xn (float and 0.0 - 1.0);
             yn (int) = image height * yn (float and 0.0 - 1.0)
    :type line_roi: tuple, optional
    :param line_gradient_range: Pair of ints determining range of gradient values (low, high) in detecting the lines.
    :type line_gradient_range: tuple, optional
    :param line_gradient_direction: Set value to 0 for horizontal gradients, 1 for vertical gradients and 2 for both.
    :type line_gradient_direction: tuple, optional
    :param line_split_probability: Pair of floats determining the probability to split long line into shorter lines.
    :type line_split_probability: tuple, optional
    :param line_replacement_value: Pair of ints determining the new value of the detected lines.
    :type line_replacement_value: tuple, optional
    :param line_min_length: Pair of ints determining the minimum length of detected lines.
    :type line_min_length: tuple, optional
    :param line_long_to_short_ratio: Pair of ints determining the threshold ratio of major axis to minor axis of the detected lines.
    :type line_long_to_short_ratio: tuple, optional
    :param line_replacement_probability: Pair of floats determining the probability to replace the detected lines with new value.
    :type line_replacement_probability: tuple, optional
    :param line_replacement_thickness: Pair of ints determining the thickness of replaced lines.
    :type line_replacement_thickness: tuple, optional
    """

    def __init__(
        self,
        line_roi=(0.0, 0.0, 1.0, 1.0),
        line_gradient_range=(32, 255),
        line_gradient_direction=(0, 2),
        line_split_probability=(0.2, 0.4),
        line_replacement_value=(250, 255),
        line_min_length=(30, 40),
        line_long_to_short_ratio=(5, 7),
        line_replacement_probability=(0.4, 0.5),
        line_replacement_thickness=(1, 3),
    ):
        """Constructor method"""
        super().__init__()
        self.line_roi = line_roi
        self.line_gradient_range = line_gradient_range
        self.line_gradient_direction = line_gradient_direction
        self.line_split_probability = line_split_probability
        self.line_replacement_value = line_replacement_value
        self.line_min_length = line_min_length
        self.line_long_to_short_ratio = line_long_to_short_ratio
        self.line_replacement_probability = line_replacement_probability
        self.line_replacement_thickness = line_replacement_thickness

    def sample(self, meta=None):
        if meta is None:
            meta = {}
            
        # Sample ROI
        line_roi = meta.get("line_roi", self.line_roi)
        
        # Sample gradient parameters
        line_gradient_range = meta.get("line_gradient_range", self.line_gradient_range)
        gradient_direction = meta.get("gradient_direction", random.randint(
            self.line_gradient_direction[0], 
            self.line_gradient_direction[1]
        ))
        
        # Sample line split parameters
        line_split_probability = meta.get("line_split_probability", np.random.uniform(
            self.line_split_probability[0], 
            self.line_split_probability[1]
        ))
        
        # Sample replacement parameters
        line_replacement_value = meta.get("line_replacement_value", self.line_replacement_value)
        line_replacement_probability = meta.get("line_replacement_probability", np.random.uniform(
            self.line_replacement_probability[0],
            self.line_replacement_probability[1],
        ))
        
        # Sample line dimension parameters
        line_min_length = meta.get("line_min_length", random.randint(
            self.line_min_length[0], 
            self.line_min_length[1]
        ))
        long_to_short_ratio = meta.get("long_to_short_ratio", random.randint(
            self.line_long_to_short_ratio[0], 
            self.line_long_to_short_ratio[1]
        ))
        
        # Sample thickness
        line_replacement_thickness = meta.get("line_replacement_thickness", self.line_replacement_thickness)
        
        # Create the metadata
        meta = {
            "line_roi": line_roi,
            "line_gradient_range": line_gradient_range,
            "gradient_direction": gradient_direction,
            "line_split_probability": line_split_probability,
            "line_replacement_value": line_replacement_value,
            "line_min_length": line_min_length,
            "long_to_short_ratio": long_to_short_ratio,
            "line_replacement_probability": line_replacement_probability,
            "line_replacement_thickness": line_replacement_thickness,
        }
        
        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        
        for layer in layers:
            image = layer.image.copy()
            
            # Get parameters from meta
            line_roi = meta["line_roi"]
            line_gradient_range = meta["line_gradient_range"]
            gradient_direction = meta["gradient_direction"]
            line_split_probability = meta["line_split_probability"]
            line_replacement_value = meta["line_replacement_value"]
            line_min_length = meta["line_min_length"]
            long_to_short_ratio = meta["long_to_short_ratio"]
            line_replacement_probability = meta["line_replacement_probability"]
            line_replacement_thickness = meta["line_replacement_thickness"]
            
            # ROI calculation
            ysize, xsize = image.shape[:2]
            xstart, ystart, xend, yend = line_roi
            
            # When value is float and in between 0-1, scale it with image size
            if xstart >= 0 and xstart <= 1 and isinstance(xstart, float):
                xstart = int(xstart * xsize)
            if ystart >= 0 and ystart <= 1 and isinstance(ystart, float):
                ystart = int(ystart * ysize)
            if xend >= 0 and xend <= 1 and isinstance(xend, float):
                xend = int(xend * xsize)
            if yend >= 0 and yend <= 1 and isinstance(yend, float):
                yend = int(yend * ysize)
                
            # Extract ROI
            image_roi = image[ystart:yend, xstart:xend]
            
            # Convert to grayscale if needed
            if len(image.shape) > 2:
                image_gray = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
            else:
                image_gray = image_roi
            
            # Create random mask for line splitting
            image_random = np.random.uniform(0, 1, size=(image_gray.shape[0], image_gray.shape[1]))
            
            # Calculate gradients
            gx, gy = np.gradient(image_gray, edge_order=1)
            
            # Process horizontal or both gradients
            if gradient_direction != 1:
                # Process horizontal gradients
                gx = abs(gx)  # Remove negative values
                
                # Remove gradients beyond the selected range
                gx[gx <= line_gradient_range[0]] = 0
                gx[gx > line_gradient_range[1]] = 0
                
                # Randomly remove line value
                gx[image_random < line_split_probability] = 0
                
                # Get contours of lines
                contours_x, _ = cv2.findContours(
                    gx.astype("uint8"),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_NONE,
                )
                
                # Get mask of horizontal lines
                mask_x = np.zeros_like(image_gray)
                for contour in contours_x:
                    x, y, w, h = cv2.boundingRect(contour)
                    # Check for horizontal line criteria
                    if (
                        w > h * long_to_short_ratio
                        and w > line_min_length
                        and np.random.random() < line_replacement_probability
                    ):
                        thickness = random.randint(line_replacement_thickness[0], line_replacement_thickness[1])
                        cv2.drawContours(
                            mask_x,
                            [contour],
                            -1,
                            (255, 255, 255),
                            thickness,
                        )
            
            # Process vertical or both gradients
            if gradient_direction != 0:
                # Process vertical gradients
                gy = abs(gy)  # Remove negative values
                
                # Remove gradients beyond the selected range
                gy[gy <= line_gradient_range[0]] = 0
                gy[gy > line_gradient_range[1]] = 0
                
                # Randomly remove line value
                gy[image_random < line_split_probability] = 0
                
                # Get contours of lines
                contours_y, _ = cv2.findContours(
                    gy.astype("uint8"),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_NONE,
                )
                
                # Get mask of vertical lines
                mask_y = np.zeros_like(image_gray)
                for contour in contours_y:
                    x, y, w, h = cv2.boundingRect(contour)
                    # Check for vertical line criteria
                    if (
                        h > w * long_to_short_ratio
                        and h > line_min_length
                        and np.random.random() < line_replacement_probability
                    ):
                        thickness = random.randint(line_replacement_thickness[0], line_replacement_thickness[1])
                        cv2.drawContours(
                            mask_y,
                            [contour],
                            -1,
                            (255, 255, 255),
                            thickness,
                        )
            
            # Merge masks based on gradient direction
            if gradient_direction == 2:
                mask_xy = mask_x + mask_y
            elif gradient_direction == 1:
                mask_xy = mask_y
            else:
                mask_xy = mask_x
                
            # Normalize mask values
            mask_xy[mask_xy > 0] = 1
            
            # Create replacement values mask
            replacement_mask = np.random.randint(
                line_replacement_value[0],
                line_replacement_value[1] + 1,
                size=(yend - ystart, xend - xstart),
            )
            
            # Apply replacement to the image
            if len(image.shape) > 2:
                # For color images, apply to each channel except alpha
                channels = 3 if image.shape[2] >= 3 else image.shape[2]
                for i in range(channels):
                    image[ystart:yend, xstart:xend, i][mask_xy > 0] = replacement_mask[mask_xy > 0]
            else:
                # For grayscale images
                image[ystart:yend, xstart:xend][mask_xy > 0] = replacement_mask[mask_xy > 0]
            
            # Update the layer's image
            layer.image = image
            
        return meta