import random

import cv2
import numpy as np

from synthdocs.components.artifact.noisylines import NoisyLines
from synthdocs.components.component import Component


class Squish(Component):
    """Creates a squish effect by removing a fixed horizontal or vertical section of the image.

    :param squish_direction: Direction of the squish effect.
        Use 0 for horizontal squish, 1 for vertical squish, 2 for both directions.
        Use "random" to generate random direction.
    :type squish_direction: int or string, optional
    :param squish_location: List of ints determining the location of squish effect.
        If direction of squish effect is horizontal, the value determines the row coordinate of the lines.
        If direction of squish effect is vertical, the value determines the column coordinate of the lines.
        If both directions are selected, the value determines both row and column coordinate of the lines.
    :type squish_location: list, optional
    :param squish_number_range: Tuple of ints determining the number of squish effect.
    :type squish_number_range: tuple, optional
    :param squish_distance_range: Tuple of ints determining the distance of squish effect.
    :type squish_distance_range: tuple, optional
    :param squish_line: Flag to enable drawing of line in each squish effect.
    :type squish_line: int or string, optional
    :param squish_line_thickness_range: Tuple of ints determing the thickness of squish line.
    :type squish_line_thickness_range: tuple, optional
    :param p: The probability that this Component will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        squish_direction="random",
        squish_location="random",
        squish_number_range=(5, 10),
        squish_distance_range=(5, 7),
        squish_line="random",
        squish_line_thickness_range=(1, 1)
    ):
        """Constructor method"""
        super().__init__()
        self.squish_direction = squish_direction
        self.squish_location = squish_location
        self.squish_number_range = squish_number_range
        self.squish_distance_range = squish_distance_range
        self.squish_line = squish_line
        self.squish_line_thickness_range = squish_line_thickness_range

    def apply_squish(self, image, mask=None, keypoints=None, bounding_boxes=None, squish_direction=0, squish_params=None):
        """Core function to apply the squish effect.

        :param image: The input image.
        :type image: numpy array
        :param mask: The mask of labels for each pixel. Mask value should be in range of 1 to 255.
            Value of 0 will be assigned to the filled area after the transformation.
        :type mask: numpy array (uint8), optional
        :param keypoints: A dictionary of single or multiple labels where each label is a nested list of points coordinate.
        :type keypoints: dictionary, optional
        :param bounding_boxes: A nested list where each nested list contains box location (x1, y1, x2, y2).
        :type bounding_boxes: list, optional
        :param squish_direction: Direction of squish effect, where 0 = horizontal, 1 = vertical.
        :type squish_direction: int
        :param squish_params: Dictionary of parameters for the squish effect
        :type squish_params: dict, optional
        """
        ysize, xsize = image.shape[:2]

        # Use passed parameters or instance defaults
        if squish_params is None:
            squish_params = {}
        
        squish_location = squish_params.get("squish_location", self.squish_location)
        squish_number_range = squish_params.get("squish_number_range", self.squish_number_range)
        squish_distance_range = squish_params.get("squish_distance_range", self.squish_distance_range)
        squish_line = squish_params.get("squish_line", self.squish_line)
        squish_line_thickness_range = squish_params.get("squish_line_thickness_range", self.squish_line_thickness_range)

        # generate random squish number
        squish_number = random.randint(squish_number_range[0], squish_number_range[1])

        # generate squish coordinates
        if squish_location == "random":
            # vertical
            if squish_direction:
                squish_coordinates = random.sample(range(0, xsize - 1), min(squish_number, xsize - 1))
            # horizontal
            else:
                squish_coordinates = random.sample(range(0, ysize - 1), min(squish_number, ysize - 1))
        else:
            squish_coordinates = squish_location.copy()  # Copy to avoid modifying original

        # reverse sort to squish from last element so that squish location won't be affected after multiple squish iterations
        squish_coordinates.sort(reverse=True)
        squish_distance_total = 0
        squish_distances = []

        for coordinate in squish_coordinates:
            # apply squish effect based on the distance
            squish_distance = random.randint(squish_distance_range[0], squish_distance_range[1])
            # vertical
            if squish_direction:
                if coordinate < image.shape[1] and squish_distance < image.shape[1] - coordinate:
                    # image
                    image[:, coordinate:-squish_distance] = image[:, coordinate + squish_distance :]
                    # mask
                    if mask is not None:
                        mask[:, coordinate:-squish_distance] = mask[:, coordinate + squish_distance :]
                    # keypoints
                    if keypoints is not None:
                        for name, points in keypoints.items():
                            remove_indices = []
                            for i, (xpoint, ypoint) in enumerate(points):
                                # remove keypoints in squish box
                                if xpoint >= coordinate and xpoint < coordinate + squish_distance:
                                    remove_indices.append(i)
                                # reduce coordinate value if points > coordinate + squish_distance
                                elif xpoint >= coordinate + squish_distance:
                                    xpoint -= squish_distance
                                    points[i] = [xpoint, ypoint]
                            # remove points
                            for idx in sorted(remove_indices, reverse=True):
                                points.pop(idx)
                    # bounding boxes
                    if bounding_boxes is not None:
                        remove_indices = []
                        for i, bounding_box in enumerate(bounding_boxes):
                            xspoint, yspoint, xepoint, yepoint = bounding_box
                            # both x points are inside squish coordinate
                            if (
                                xspoint >= coordinate
                                and xspoint < coordinate + squish_distance
                                and xepoint >= coordinate
                                and xepoint < coordinate + squish_distance
                            ):
                                remove_indices.append(i)
                            # start point is in the squish box
                            elif xspoint >= coordinate and xspoint < coordinate + squish_distance:
                                xspoint = coordinate
                            # end point is in the squish box
                            elif xepoint >= coordinate and xepoint < coordinate + squish_distance:
                                xepoint = coordinate
                            # reduce value by squish distance
                            if xspoint >= coordinate + squish_distance:
                                xspoint -= squish_distance
                            if xepoint >= coordinate + squish_distance:
                                xepoint -= squish_distance
                            bounding_boxes[i] = [xspoint, yspoint, xepoint, yepoint]
                        # remove boxes
                        for idx in sorted(remove_indices, reverse=True):
                            bounding_boxes.pop(idx)

            # horizontal
            else:
                if coordinate < image.shape[0] and squish_distance < image.shape[0] - coordinate:
                    # image
                    image[coordinate:-squish_distance, :] = image[coordinate + squish_distance :, :]
                    # mask
                    if mask is not None:
                        mask[coordinate:-squish_distance, :] = mask[coordinate + squish_distance :, :]
                    # keypoints
                    if keypoints is not None:
                        for name, points in keypoints.items():
                            remove_indices = []
                            for i, (xpoint, ypoint) in enumerate(points):
                                # remove keypoints in squish box
                                if ypoint >= coordinate and ypoint < coordinate + squish_distance:
                                    remove_indices.append(i)
                                # reduce coordinate value if points > coordinate + squish_distance
                                elif ypoint >= coordinate + squish_distance:
                                    ypoint -= squish_distance
                                    points[i] = [xpoint, ypoint]
                            # remove points
                            for idx in sorted(remove_indices, reverse=True):
                                points.pop(idx)
                    # bounding boxes
                    if bounding_boxes is not None:
                        remove_indices = []
                        for i, bounding_box in enumerate(bounding_boxes):
                            xspoint, yspoint, xepoint, yepoint = bounding_box
                            # both y points are inside squish coordinate
                            if (
                                yspoint >= coordinate
                                and yspoint < coordinate + squish_distance
                                and yepoint >= coordinate
                                and yepoint < coordinate + squish_distance
                            ):
                                remove_indices.append(i)
                            # start point is in the squish box
                            elif yspoint >= coordinate and yspoint < coordinate + squish_distance:
                                yspoint = coordinate
                            # end point is in the squish box
                            elif yepoint >= coordinate and yepoint < coordinate + squish_distance:
                                yepoint = coordinate
                            # reduce value by squish distance
                            if yspoint >= coordinate + squish_distance:
                                yspoint -= squish_distance
                            if yepoint >= coordinate + squish_distance:
                                yepoint -= squish_distance
                            bounding_boxes[i] = [xspoint, yspoint, xepoint, yepoint]
                        # remove boxes
                        for idx in sorted(remove_indices, reverse=True):
                            bounding_boxes.pop(idx)

                    squish_distances.append(squish_distance)
                    # add total squish distance so that we can remove it later
                    squish_distance_total += squish_distance

        # Crop the image to remove empty space
        # vertical
        if squish_direction and squish_distance_total > 0:
            image = image[:, :-squish_distance_total]
            if mask is not None:
                mask = mask[:, :-squish_distance_total]
        # horizontal
        elif squish_distance_total > 0:
            image = image[:-squish_distance_total, :]
            if mask is not None:
                mask = mask[:-squish_distance_total, :]

        # generate flag for squish line
        if squish_line == "random":
            draw_squish_line = random.choice([0, 1]) > 0
        else:
            draw_squish_line = bool(squish_line)
            
        # generate lines
        if draw_squish_line:
            squish_lines_coordinates = []
            # reduce y location when there's multiple squishes
            for i, coordinate in enumerate(squish_coordinates, start=1):
                if i <= len(squish_distances):
                    squish_lines_coordinate = coordinate - sum(squish_distances[i-1:])
                    if squish_line == "random":
                        if random.choice([0, 1]) > 0:
                            squish_lines_coordinates.append(squish_lines_coordinate)
                    else:
                        squish_lines_coordinates.append(squish_lines_coordinate)
            
            # Only apply noisy lines if we have valid coordinates
            if squish_lines_coordinates:
                try:
                    noisy_lines = NoisyLines(
                        noisy_lines_direction=squish_direction,
                        noisy_lines_location=squish_lines_coordinates,
                        noisy_lines_number_range=(1, 1),
                        noisy_lines_color=(0, 0, 0),
                        noisy_lines_thickness_range=squish_line_thickness_range,
                        noisy_lines_random_noise_intensity_range=(0.01, 0.1),
                        noisy_lines_length_interval_range=(0, 0),
                        noisy_lines_gaussian_kernel_value_range=(1, 1),
                        noisy_lines_overlay_method="ink_to_paper",
                    )
                    image = noisy_lines(image)
                except Exception as e:
                    # If NoisyLines fails, we'll just continue without the lines
                    print(f"Warning: Couldn't apply noisy lines: {str(e)}")

        return image, mask, keypoints, bounding_boxes

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
        
        # Sample squish direction
        if self.squish_direction == "random":
            squish_direction = random.choice([0, 1, 2])
        else:
            squish_direction = self.squish_direction
        
        # Sample squish line flag
        if self.squish_line == "random":
            squish_line = random.choice([0, 1])
        else:
            squish_line = self.squish_line
            
        # Store sampled parameters
        meta.update({
            "squish_direction": squish_direction,
            "squish_location": self.squish_location,
            "squish_number_range": self.squish_number_range,
            "squish_distance_range": self.squish_distance_range,
            "squish_line": squish_line,
            "squish_line_thickness_range": self.squish_line_thickness_range
        })
        
        return meta

    def apply(self, layers, meta=None):
        """Apply the squish effect to layers.
        
        :param layers: List of layers to process
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
        
        squish_direction = meta["squish_direction"]
        squish_params = meta
        
        for layer in layers:
            image = layer.image.copy()
            mask = layer.mask if hasattr(layer, 'mask') else None
            keypoints = layer.keypoints if hasattr(layer, 'keypoints') else None
            bounding_boxes = layer.bounding_boxes if hasattr(layer, 'bounding_boxes') else None
            
            # convert and make sure image is color image
            is_gray = False
            if len(image.shape) < 3:
                is_gray = True
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            try:
                # Apply squish effect
                if squish_direction == 2:
                    # Apply in both directions
                    random_direction = random.randint(0, 1)
                    # First direction
                    image, mask, keypoints, bounding_boxes = self.apply_squish(
                        image,
                        mask,
                        keypoints,
                        bounding_boxes,
                        squish_direction=random_direction,
                        squish_params=squish_params,
                    )
                    # Second direction
                    image, mask, keypoints, bounding_boxes = self.apply_squish(
                        image,
                        mask,
                        keypoints,
                        bounding_boxes,
                        squish_direction=1 - random_direction,
                        squish_params=squish_params,
                    )
                else:
                    # Apply in single direction
                    image, mask, keypoints, bounding_boxes = self.apply_squish(
                        image,
                        mask,
                        keypoints,
                        bounding_boxes,
                        squish_direction=squish_direction,
                        squish_params=squish_params,
                    )
                
                # return image follows the input image color channel
                if is_gray:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # Update layer properties
                layer.image = image
                if mask is not None and hasattr(layer, 'mask'):
                    layer.mask = mask
                if keypoints is not None and hasattr(layer, 'keypoints'):
                    layer.keypoints = keypoints
                if bounding_boxes is not None and hasattr(layer, 'bounding_boxes'):
                    layer.bounding_boxes = bounding_boxes
                    
            except Exception as e:
                print(f"Error applying squish effect: {str(e)}")
                # If an error occurs, we keep the original image
                pass
        
        return meta