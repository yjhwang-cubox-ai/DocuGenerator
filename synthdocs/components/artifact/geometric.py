import random

import cv2
import numpy as np

from synthdocs.utils.lib import rotate_bounding_boxes
from synthdocs.utils.lib import rotate_image_PIL
from synthdocs.utils.lib import rotate_keypoints
from synthdocs.utils.lib import update_mask_labels
from synthdocs.components.component import Component


class Geometric(Component):
    """Applies basic geometric transformations such as resizing, flips and rotation."""

    def __init__(
        self,
        scale=(1, 1),
        translation=(0, 0),
        fliplr=[True, False],
        flipud=[True, False],
        crop=(),
        rotate_range=(0, 0),
        padding=[0, 0, 0, 0],
        padding_type="fill",
        padding_value=(255, 255, 255),
        randomize=0,
    ):
        """Constructor method"""
        super().__init__()
        self.scale = scale
        self.translation = translation
        self.fliplr = random.choice(fliplr)
        self.flipud = random.choice(flipud)
        self.crop = crop
        self.rotate_range = rotate_range
        self.randomize = randomize
        self.padding = padding
        self.padding_type = padding_type
        self.padding_value = padding_value

    def randomize_parameters(self, image):
        """Randomize parameters for random geometrical effect.

        :param image: The input image.
        :type image: numpy array
        :return: Dictionary of randomized parameters
        :rtype: dict
        """
        params = {}
        
        # Get image dimensions
        ysize, xsize = image.shape[:2]
        
        # Randomize scale
        params["scale"] = (random.uniform(0.5, 1), random.uniform(1, 1.5))

        # Randomize translation value
        params["translation"] = (random.randint(0, int(xsize * 0.1)), random.randint(0, int(ysize * 0.1)))

        # Randomize flip
        params["fliplr"] = random.choice([0, 1])
        params["flipud"] = random.choice([0, 1])

        # Randomize crop
        cx1 = random.randint(0, int(xsize / 5))
        cx2 = random.randint(int(xsize / 2), xsize - 1)
        cy1 = random.randint(0, int(ysize / 5))
        cy2 = random.randint(int(ysize / 2), ysize - 1)
        params["crop"] = (cx1, cy1, cx2, cy2)

        # Randomize rotate
        params["rotate_range"] = (-10, 10)
        params["angle"] = random.uniform(-10, 10)

        # Randomize padding
        params["padding"] = [
            random.randint(0, int(xsize / 5)),
            random.randint(0, int(xsize / 5)),
            random.randint(0, int(ysize / 5)),
            random.randint(0, int(ysize / 5)),
        ]
        params["padding_value"] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        params["padding_type"] = random.choice(["fill", "mirror", "duplicate"])
        
        return params

    def run_crop(self, image, mask=None, keypoints=None, bounding_boxes=None, crop=None):
        """Crop image based on the input cropping box."""
        # Use passed crop or instance crop
        crop_to_use = crop if crop is not None else self.crop

        # Make sure there's only 4 inputs, x0, y0, xn, yn
        if len(crop_to_use) == 4:
            ysize, xsize = image.shape[:2]
            xstart, ystart, xend, yend = crop_to_use

            # When value is float and in between 0-1, scale it with image size
            if xstart >= 0 and xstart <= 1 and isinstance(xstart, float):
                xstart = int(xstart * xsize)
            if ystart >= 0 and ystart <= 1 and isinstance(ystart, float):
                ystart = int(ystart * ysize)
            if xend >= 0 and xend <= 1 and isinstance(xend, float):
                xend = int(xend * xsize)
            if yend >= 0 and yend <= 1 and isinstance(yend, float):
                yend = int(yend * ysize)

            # When value is set to -1, it takes image size
            if yend == -1:
                yend = ysize
            if xend == -1:
                xend = xsize
                
            # Condition to make sure cropping range is valid
            check_y = yend > ystart and ystart >= 0
            check_x = xend > xstart and xstart >= 0

            if check_y and check_x:
                # Crop image
                image = image[ystart:yend, xstart:xend]

                # Crop mask
                if mask is not None:
                    mask = mask[ystart:yend, xstart:xend]

                # Remove keypoints outside the cropping boundary
                if keypoints is not None:
                    for name, points in keypoints.items():
                        remove_indices = []
                        # Check and save the indices to be removed
                        for i, (xpoint, ypoint) in enumerate(points):
                            if xpoint < xstart or xpoint >= xend or ypoint < ystart or ypoint >= yend:
                                remove_indices.append(i)
                        # Remove points (in reverse order to avoid index issues)
                        for idx in sorted(remove_indices, reverse=True):
                            points.pop(idx)
                        # Update points location after the cropping process
                        for i, (xpoint, ypoint) in enumerate(points):
                            points[i] = [xpoint - xstart, ypoint - ystart]

                # Remove and limit bounding boxes to the cropped boundary
                if bounding_boxes is not None:
                    remove_indices = []
                    for i, bounding_box in enumerate(bounding_boxes):
                        xspoint, yspoint, xepoint, yepoint = bounding_box
                        # Both start and end points are outside the cropped area
                        if (xspoint < xstart and xepoint < xstart) or (xspoint >= xend and xepoint >= xend) or \
                           (yspoint < ystart and yepoint < ystart) or (yspoint >= yend and yepoint >= yend):
                            remove_indices.append(i)
                        else:
                            # Clip box coordinates to crop boundaries
                            xspoint = max(xspoint, xstart)
                            yspoint = max(yspoint, ystart)
                            xepoint = min(xepoint, xend - 1)
                            yepoint = min(yepoint, yend - 1)
                            
                            # Adjust coordinates relative to the new cropped image
                            bounding_boxes[i] = [
                                xspoint - xstart,
                                yspoint - ystart,
                                xepoint - xstart,
                                yepoint - ystart,
                            ]
                    
                    # Remove boxes outside crop area
                    for idx in sorted(remove_indices, reverse=True):
                        bounding_boxes.pop(idx)

        return image, mask, keypoints, bounding_boxes

    def run_padding(self, image, mask=None, keypoints=None, bounding_boxes=None, padding=None, padding_type=None, padding_value=None):
        """Apply padding to image based on the input padding value."""
        # Use passed parameters or instance variables
        padding_to_use = padding if padding is not None else self.padding
        padding_type_to_use = padding_type if padding_type is not None else self.padding_type
        padding_value_to_use = padding_value if padding_value is not None else self.padding_value
        
        # Make sure padding is a list for modification
        padding_to_use = list(padding_to_use)
        
        # Track padding applied to each side
        left_pad = right_pad = top_pad = bottom_pad = 0
        
        # Get image dimensions
        ysize, xsize = image.shape[:2]
        
        # Apply left padding
        if padding_to_use[0] > 0:
            # Convert percentage to pixels if needed
            if padding_to_use[0] <= 1 and isinstance(padding_to_use[0], float):
                padding_to_use[0] = int(padding_to_use[0] * xsize)
                
            left_pad = padding_to_use[0]
            
            # Create padding based on type
            if padding_type_to_use == "duplicate":
                if padding_to_use[0] < image.shape[1]:
                    image_padding = image[:, :padding_to_use[0]].copy()
                    if mask is not None:
                        mask_padding = mask[:, :padding_to_use[0]].copy()
                else:
                    # If padding larger than image, just use solid color
                    padding_type_to_use = "fill"
            
            # If using mirror or fill
            if padding_type_to_use == "mirror":
                image_padding = np.fliplr(image[:, :padding_to_use[0]].copy())
                if mask is not None:
                    mask_padding = np.fliplr(mask[:, :padding_to_use[0]].copy())
            elif padding_type_to_use == "fill":
                # Create solid color padding
                if len(image.shape) > 2:
                    image_padding = np.full((ysize, padding_to_use[0], image.shape[2]), 
                                          padding_value_to_use, dtype="uint8")
                else:
                    image_padding = np.full((ysize, padding_to_use[0]), 
                                          padding_value_to_use, dtype="uint8")
                
                if mask is not None:
                    mask_padding = np.zeros((ysize, padding_to_use[0]), dtype="uint8")
            
            # Combine padding with image
            image = np.concatenate([image_padding, image], axis=1)
            if mask is not None:
                mask = np.concatenate([mask_padding, mask], axis=1)
        
        # Apply right padding (similar to left padding)
        if padding_to_use[1] > 0:
            # Convert percentage to pixels if needed
            if padding_to_use[1] <= 1 and isinstance(padding_to_use[1], float):
                padding_to_use[1] = int(padding_to_use[1] * xsize)
                
            right_pad = padding_to_use[1]
            
            # Create padding based on type
            if padding_type_to_use == "duplicate":
                if padding_to_use[1] < image.shape[1]:
                    image_padding = image[:, -padding_to_use[1]:].copy()
                    if mask is not None:
                        mask_padding = mask[:, -padding_to_use[1]:].copy()
                else:
                    padding_type_to_use = "fill"
            
            if padding_type_to_use == "mirror":
                image_padding = np.fliplr(image[:, -padding_to_use[1]:].copy())
                if mask is not None:
                    mask_padding = np.fliplr(mask[:, -padding_to_use[1]:].copy())
            elif padding_type_to_use == "fill":
                if len(image.shape) > 2:
                    image_padding = np.full((ysize, padding_to_use[1], image.shape[2]), 
                                          padding_value_to_use, dtype="uint8")
                else:
                    image_padding = np.full((ysize, padding_to_use[1]), 
                                          padding_value_to_use, dtype="uint8")
                
                if mask is not None:
                    mask_padding = np.zeros((ysize, padding_to_use[1]), dtype="uint8")
            
            # Combine padding with image
            image = np.concatenate([image, image_padding], axis=1)
            if mask is not None:
                mask = np.concatenate([mask, mask_padding], axis=1)
        
        # Update dimensions after horizontal padding
        ysize, xsize = image.shape[:2]
        
        # Apply top padding
        if padding_to_use[2] > 0:
            # Convert percentage to pixels if needed
            if padding_to_use[2] <= 1 and isinstance(padding_to_use[2], float):
                padding_to_use[2] = int(padding_to_use[2] * ysize)
                
            top_pad = padding_to_use[2]
            
            # Create padding based on type
            if padding_type_to_use == "duplicate":
                if padding_to_use[2] < image.shape[0]:
                    image_padding = image[:padding_to_use[2], :].copy()
                    if mask is not None:
                        mask_padding = mask[:padding_to_use[2], :].copy()
                else:
                    padding_type_to_use = "fill"
            
            if padding_type_to_use == "mirror":
                image_padding = np.flipud(image[:padding_to_use[2], :].copy())
                if mask is not None:
                    mask_padding = np.flipud(mask[:padding_to_use[2], :].copy())
            elif padding_type_to_use == "fill":
                if len(image.shape) > 2:
                    image_padding = np.full((padding_to_use[2], xsize, image.shape[2]), 
                                          padding_value_to_use, dtype="uint8")
                else:
                    image_padding = np.full((padding_to_use[2], xsize), 
                                          padding_value_to_use, dtype="uint8")
                
                if mask is not None:
                    mask_padding = np.zeros((padding_to_use[2], xsize), dtype="uint8")
            
            # Combine padding with image
            image = np.concatenate([image_padding, image], axis=0)
            if mask is not None:
                mask = np.concatenate([mask_padding, mask], axis=0)
        
        # Apply bottom padding (similar to top padding)
        if padding_to_use[3] > 0:
            # Convert percentage to pixels if needed
            if padding_to_use[3] <= 1 and isinstance(padding_to_use[3], float):
                padding_to_use[3] = int(padding_to_use[3] * ysize)
                
            bottom_pad = padding_to_use[3]
            
            # Create padding based on type
            if padding_type_to_use == "duplicate":
                if padding_to_use[3] < image.shape[0]:
                    image_padding = image[-padding_to_use[3]:, :].copy()
                    if mask is not None:
                        mask_padding = mask[-padding_to_use[3]:, :].copy()
                else:
                    padding_type_to_use = "fill"
            
            if padding_type_to_use == "mirror":
                image_padding = np.flipud(image[-padding_to_use[3]:, :].copy())
                if mask is not None:
                    mask_padding = np.flipud(mask[-padding_to_use[3]:, :].copy())
            elif padding_type_to_use == "fill":
                if len(image.shape) > 2:
                    image_padding = np.full((padding_to_use[3], xsize, image.shape[2]), 
                                          padding_value_to_use, dtype="uint8")
                else:
                    image_padding = np.full((padding_to_use[3], xsize), 
                                          padding_value_to_use, dtype="uint8")
                
                if mask is not None:
                    mask_padding = np.zeros((padding_to_use[3], xsize), dtype="uint8")
            
            # Combine padding with image
            image = np.concatenate([image, image_padding], axis=0)
            if mask is not None:
                mask = np.concatenate([mask, mask_padding], axis=0)

        # Update keypoints and bounding boxes after padding
        if keypoints is not None:
            for name, points in keypoints.items():
                for i, (xpoint, ypoint) in enumerate(points):
                    points[i] = [xpoint + left_pad, ypoint + top_pad]

        if bounding_boxes is not None:
            for i, bounding_box in enumerate(bounding_boxes):
                xspoint, yspoint, xepoint, yepoint = bounding_box
                bounding_boxes[i] = [
                    xspoint + left_pad,
                    yspoint + top_pad,
                    xepoint + left_pad,
                    yepoint + top_pad,
                ]

        return image, mask, keypoints, bounding_boxes

    def run_flip(self, image, mask=None, keypoints=None, bounding_boxes=None, fliplr=None, flipud=None):
        """Flip image left-right or up-down based on the input flipping flags."""
        # Use passed parameters or instance variables
        fliplr_to_use = fliplr if fliplr is not None else self.fliplr
        flipud_to_use = flipud if flipud is not None else self.flipud
        
        # Get image dimensions
        ysize, xsize = image.shape[:2]
        
        # Apply horizontal (left-right) flip
        if fliplr_to_use:
            # Flip the image
            image = np.fliplr(image)
            
            # Flip the mask
            if mask is not None:
                mask = np.fliplr(mask)
            
            # Update keypoints
            if keypoints is not None:
                for name, points in keypoints.items():
                    for i, (xpoint, ypoint) in enumerate(points):
                        points[i] = [xsize - 1 - xpoint, ypoint]
            
            # Update bounding boxes
            if bounding_boxes is not None:
                for i, bounding_box in enumerate(bounding_boxes):
                    xspoint, yspoint, xepoint, yepoint = bounding_box
                    # Swap and invert x-coordinates
                    bounding_boxes[i] = [
                        xsize - 1 - xepoint, 
                        yspoint, 
                        xsize - 1 - xspoint, 
                        yepoint
                    ]
        
        # Apply vertical (up-down) flip
        if flipud_to_use:
            # Flip the image
            image = np.flipud(image)
            
            # Flip the mask
            if mask is not None:
                mask = np.flipud(mask)
            
            # Update keypoints
            if keypoints is not None:
                for name, points in keypoints.items():
                    for i, (xpoint, ypoint) in enumerate(points):
                        points[i] = [xpoint, ysize - 1 - ypoint]
            
            # Update bounding boxes
            if bounding_boxes is not None:
                for i, bounding_box in enumerate(bounding_boxes):
                    xspoint, yspoint, xepoint, yepoint = bounding_box
                    # Swap and invert y-coordinates
                    bounding_boxes[i] = [
                        xspoint, 
                        ysize - 1 - yepoint, 
                        xepoint, 
                        ysize - 1 - yspoint
                    ]
                    
        return image, mask, keypoints, bounding_boxes

    def run_scale(self, image, mask=None, keypoints=None, bounding_boxes=None, scale=None):
        """Scale image size based on the input scaling ratio."""
        # Use passed scale or instance scale
        scale_to_use = scale if scale is not None else self.scale
        
        # Convert to list and ensure positive values
        scale_to_use = list(scale_to_use)
        scale_to_use[0] = abs(scale_to_use[0])
        scale_to_use[1] = abs(scale_to_use[1])
        
        # Only apply scaling if range is not 1.0
        if scale_to_use[0] != 1 or scale_to_use[1] != 1:
            # Sample scale factor from range
            scale_factor = random.uniform(scale_to_use[0], scale_to_use[1])
            
            if scale_factor != 1:
                # Calculate new dimensions
                new_width = max(1, int(image.shape[1] * scale_factor))
                new_height = max(1, int(image.shape[0] * scale_factor))
                new_size = (new_width, new_height)
                
                # Resize image
                image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

                # Resize and update mask
                if mask is not None:
                    mask_labels = np.unique(mask).tolist() + [0]
                    mask = cv2.resize(mask, new_size, interpolation=cv2.INTER_AREA)
                    update_mask_labels(mask, mask_labels)

                # Scale keypoints
                if keypoints is not None:
                    for name, points in keypoints.items():
                        for i, (xpoint, ypoint) in enumerate(points):
                            points[i] = [int(xpoint * scale_factor), int(ypoint * scale_factor)]

                # Scale bounding boxes
                if bounding_boxes is not None:
                    for i, bounding_box in enumerate(bounding_boxes):
                        xspoint, yspoint, xepoint, yepoint = bounding_box
                        bounding_boxes[i] = [
                            int(xspoint * scale_factor),
                            int(yspoint * scale_factor),
                            int(xepoint * scale_factor),
                            int(yepoint * scale_factor),
                        ]

        return image, mask, keypoints, bounding_boxes

    def run_translation(self, image, mask=None, keypoints=None, bounding_boxes=None, translation=None):
        """Translate image based on the input translation value."""
        # Use passed translation or instance translation
        translation_to_use = translation if translation is not None else self.translation
        translation_to_use = list(translation_to_use)  # Convert to list for modification
        
        # Get image dimensions
        ysize, xsize = image.shape[:2]
        
        # Convert percentage values to pixels
        if translation_to_use[0] <= 1 and translation_to_use[0] >= -1 and isinstance(translation_to_use[0], float):
            translation_to_use[0] = int(translation_to_use[0] * xsize)
        if translation_to_use[1] <= 1 and translation_to_use[1] >= -1 and isinstance(translation_to_use[1], float):
            translation_to_use[1] = int(translation_to_use[1] * ysize)

        # Get translation offsets
        offset_x = translation_to_use[0]
        offset_y = translation_to_use[1]
        
        # Skip if no translation
        if offset_x == 0 and offset_y == 0:
            return image, mask, keypoints, bounding_boxes
        
        # Create new blank images filled with white
        image_new = np.full_like(image, fill_value=255, dtype="uint8")
        if mask is not None:
            mask_new = np.zeros_like(mask, dtype="uint8")
        
        # Apply horizontal translation
        if offset_x > 0:
            image_new[:, offset_x:] = image[:, :-offset_x]
            if mask is not None:
                mask_new[:, offset_x:] = mask[:, :-offset_x]
        elif offset_x < 0:
            image_new[:, :offset_x] = image[:, abs(offset_x):]
            if mask is not None:
                mask_new[:, :offset_x] = mask[:, abs(offset_x):]
        else:
            image_new = image.copy()
            if mask is not None:
                mask_new = mask.copy()
        
        # Store horizontally translated results
        image = image_new.copy()
        if mask is not None:
            mask = mask_new.copy()
        
        # Create new blank images for vertical translation
        image_new = np.full_like(image, fill_value=255, dtype="uint8")
        if mask is not None:
            mask_new = np.zeros_like(mask, dtype="uint8")
        
        # Apply vertical translation
        if offset_y > 0:
            image_new[offset_y:, :] = image[:-offset_y, :]
            if mask is not None:
                mask_new[offset_y:, :] = mask[:-offset_y, :]
        elif offset_y < 0:
            image_new[:offset_y, :] = image[abs(offset_y):, :]
            if mask is not None:
                mask_new[:offset_y, :] = mask[abs(offset_y):, :]
        else:
            image_new = image.copy()
            if mask is not None:
                mask_new = mask.copy()
        
        # Store final results
        image = image_new
        if mask is not None:
            mask = mask_new

        # Update keypoints
        if keypoints is not None:
            for name, points in keypoints.items():
                for i, (xpoint, ypoint) in enumerate(points):
                    points[i] = [xpoint + offset_x, ypoint + offset_y]

        # Update bounding boxes
        if bounding_boxes is not None:
            for i, bounding_box in enumerate(bounding_boxes):
                xspoint, yspoint, xepoint, yepoint = bounding_box
                bounding_boxes[i] = [
                    xspoint + offset_x,
                    yspoint + offset_y,
                    xepoint + offset_x,
                    yepoint + offset_y,
                ]

        return image, mask, keypoints, bounding_boxes

    def run_rotation(self, image, mask=None, keypoints=None, bounding_boxes=None, rotate_range=None, padding_value=None):
        """Rotate image based on the input rotation angle."""
        # Use passed parameters or instance variables
        rotate_range_to_use = rotate_range if rotate_range is not None else self.rotate_range
        padding_value_to_use = padding_value if padding_value is not None else self.padding_value
        
        # Skip if no rotation to apply
        if rotate_range_to_use[0] == 0 and rotate_range_to_use[1] == 0:
            return image, mask, keypoints, bounding_boxes
            
        # Sample angle from range
        angle = random.uniform(rotate_range_to_use[0], rotate_range_to_use[1])
        
        # Skip if angle is zero
        if angle == 0:
            return image, mask, keypoints, bounding_boxes
            
        # Store original image dimensions
        ysize, xsize = image.shape[:2]
        
        try:
            # Rotate image using PIL
            image = rotate_image_PIL(
                image, angle, expand=1, background_value=padding_value_to_use
            )
            
            # Rotate mask
            if mask is not None:
                mask_labels = np.unique(mask).tolist() + [0]
                mask = rotate_image_PIL(mask, angle, expand=1)
                update_mask_labels(mask, mask_labels)
            
            # Get new image dimensions
            new_ysize, new_xsize = image.shape[:2]
            
            # Calculate rotation offset
            cx = int(xsize / 2)  # Center x
            cy = int(ysize / 2)  # Center y
            x_offset = (new_xsize / 2) - cx
            y_offset = (new_ysize / 2) - cy
            
            # Rotate keypoints
            if keypoints is not None:
                rotate_keypoints(keypoints, cx, cy, x_offset, y_offset, -angle)
            
            # Rotate bounding boxes
            if bounding_boxes is not None:
                rotate_bounding_boxes(bounding_boxes, cx, cy, x_offset, y_offset, -angle)
        
        except Exception as e:
            print(f"Error during rotation: {e}")
            # If rotation fails, return original image
            
        return image, mask, keypoints, bounding_boxes

    def sample(self, meta=None):
        """Sample random parameters for geometric transformations.
        
        :param meta: Optional metadata dictionary with parameters to use
        :type meta: dict, optional
        :return: Metadata dictionary with sampled parameters
        :rtype: dict
        """
        if meta is None:
            meta = {}
            
        meta["run"] = True
        
        # Sample image from first layer if available
        image = None
        if "layers" in meta and meta["layers"] and hasattr(meta["layers"][0], "image"):
            image = meta["layers"][0].image
        
        # If randomize is enabled, generate random parameters
        if self.randomize and image is not None:
            params = self.randomize_parameters(image)
            meta.update(params)
        else:
            # Otherwise use the instance variables
            meta.update({
                "scale": self.scale,
                "translation": self.translation,
                "fliplr": self.fliplr,
                "flipud": self.flipud,
                "crop": self.crop,
                "rotate_range": self.rotate_range,
                "padding": self.padding.copy() if hasattr(self.padding, "copy") else self.padding,
                "padding_type": self.padding_type,
                "padding_value": self.padding_value,
            })
            
            # Sample angle from range if needed
            if self.rotate_range[0] != self.rotate_range[1]:
                meta["angle"] = random.uniform(self.rotate_range[0], self.rotate_range[1])
            else:
                meta["angle"] = self.rotate_range[0]
        
        return meta

    def apply(self, layers, meta=None):
        """Apply geometric transformations to layers.
        
        :param layers: List of layers to process
        :type layers: list
        :param meta: Optional metadata with parameters
        :type meta: dict, optional
        :return: Updated metadata
        :rtype: dict
        """
        # Sample parameters if not provided
        meta = self.sample(meta)
        
        # Skip processing if run is False
        if not meta.get("run", True):
            return meta
        
        # Process each layer
        for layer in layers:
            try:
                # Get image and optional metadata from layer
                image = layer.image.copy()
                mask = layer.mask if hasattr(layer, 'mask') else None
                keypoints = layer.keypoints if hasattr(layer, 'keypoints') else None
                bounding_boxes = layer.bounding_boxes if hasattr(layer, 'bounding_boxes') else None
                
                # Apply operations in sequence
                
                # 1. Crop
                if meta.get("crop"):
                    image, mask, keypoints, bounding_boxes = self.run_crop(
                        image, mask, keypoints, bounding_boxes, meta.get("crop")
                    )
                
                # 2. Padding
                if any(meta.get("padding", [0, 0, 0, 0])):
                    image, mask, keypoints, bounding_boxes = self.run_padding(
                        image, mask, keypoints, bounding_boxes,
                        meta.get("padding"),
                        meta.get("padding_type"),
                        meta.get("padding_value")
                    )
                
                # 3. Scale
                if meta.get("scale") != (1, 1):
                    image, mask, keypoints, bounding_boxes = self.run_scale(
                        image, mask, keypoints, bounding_boxes, meta.get("scale")
                    )
                
                # 4. Translation
                if meta.get("translation") != (0, 0):
                    image, mask, keypoints, bounding_boxes = self.run_translation(
                        image, mask, keypoints, bounding_boxes, meta.get("translation")
                    )
                
                # 5. Flip
                if meta.get("fliplr") or meta.get("flipud"):
                    image, mask, keypoints, bounding_boxes = self.run_flip(
                        image, mask, keypoints, bounding_boxes,
                        meta.get("fliplr"), meta.get("flipud")
                    )
                
                # 6. Rotation
                if meta.get("rotate_range") != (0, 0) or meta.get("angle", 0) != 0:
                    image, mask, keypoints, bounding_boxes = self.run_rotation(
                        image, mask, keypoints, bounding_boxes,
                        meta.get("rotate_range"), meta.get("padding_value")
                    )
                
                # Update layer with transformed data
                layer.image = image
                
                # Update optional layer attributes if they exist
                if mask is not None and hasattr(layer, 'mask'):
                    layer.mask = mask
                if keypoints is not None and hasattr(layer, 'keypoints'):
                    layer.keypoints = keypoints
                if bounding_boxes is not None and hasattr(layer, 'bounding_boxes'):
                    layer.bounding_boxes = bounding_boxes
                    
            except Exception as e:
                print(f"Error applying geometric transformations to layer: {str(e)}")
                # Keep original layer if transformation fails
        
        return meta