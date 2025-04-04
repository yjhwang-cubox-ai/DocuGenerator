import random

import cv2
import numpy as np

from synthdocs.components.component import Component


class InkMottling(Component):
    """Create a random pattern effect in the detected ink by blending a layer of random Gaussian noise.

    :param ink_mottling_alpha_range: Tuple of floats determining the alpha value of the added effect.
    :type ink_mottling_alpha_range: tuple, optional
    :param ink_mottling_noise_scale_range: Tuple of ints determining the size of Gaussian noise pattern.
    :type ink_mottling_noise_scale_range: tuple, optional
    :param ink_mottling_gaussian_kernel_range: Tuple of ints determining the Gaussian kernel value.
    :type ink_mottling_gaussian_kernel_range: tuple, optional
    :param p: The probability that this Component will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        ink_mottling_alpha_range=(0.2, 0.3),
        ink_mottling_noise_scale_range=(2, 2),
        ink_mottling_gaussian_kernel_range=(3, 5),
    ):
        """Constructor method"""
        super().__init__()
        self.ink_mottling_alpha_range = ink_mottling_alpha_range
        self.ink_mottling_noise_scale_range = ink_mottling_noise_scale_range
        self.ink_mottling_gaussian_kernel_range = ink_mottling_gaussian_kernel_range

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
        
        # Sample alpha value for blending
        meta["ink_mottling_alpha"] = random.uniform(
            self.ink_mottling_alpha_range[0],
            self.ink_mottling_alpha_range[1]
        )
        
        # Sample noise scale (controls the size of the noise pattern)
        meta["ink_mottling_noise_scale"] = random.randint(
            self.ink_mottling_noise_scale_range[0],
            self.ink_mottling_noise_scale_range[1]
        )
        
        # Sample Gaussian kernel size for blurring the noise
        meta["ink_mottling_gaussian_kernel"] = random.randint(
            self.ink_mottling_gaussian_kernel_range[0],
            self.ink_mottling_gaussian_kernel_range[1]
        )
        
        # Ensure kernel value is odd (required by Gaussian blur)
        if meta["ink_mottling_gaussian_kernel"] % 2 == 0:
            meta["ink_mottling_gaussian_kernel"] += 1
            
        return meta

    def apply(self, layers, meta=None):
        """Apply the ink mottling effect to the layers.
        
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
        ink_mottling_alpha = meta.get("ink_mottling_alpha")
        ink_mottling_noise_scale = meta.get("ink_mottling_noise_scale")
        ink_mottling_gaussian_kernel = meta.get("ink_mottling_gaussian_kernel")
        
        for layer in layers:
            try:
                # Get a copy of the image to work with
                image = layer.image.copy()

                image = image.astype(np.uint8)
                
                # Get image dimensions
                ysize, xsize = image.shape[:2]
                
                # Check if image is grayscale and convert to color if needed
                is_gray = False
                if len(image.shape) <= 2:
                    is_gray = True
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                
                # Create a mask to identify ink areas (dark regions)
                image_mask = np.zeros((ysize, xsize), dtype="uint8")
                
                # Get ink area from each channel
                for i in range(3):
                    # Convert image into binary
                    _, image_binary = cv2.threshold(
                        image[:, :, i], 
                        0, 
                        255, 
                        cv2.THRESH_BINARY + cv2.THRESH_OTSU
                    )
                    image_mask += image_binary
                
                # Invert ink area (dark area)
                image_mask = 255 - image_mask
                
                # Generate random noise pattern
                try:
                    # Calculate size for the noise pattern
                    noise_height = max(1, int(ysize / ink_mottling_noise_scale))
                    noise_width = max(1, int(xsize / ink_mottling_noise_scale))
                    
                    # Create random noise
                    image_random = np.random.randint(
                        0,
                        255,
                        size=(noise_height, noise_width)
                    ).astype("uint8")
                    
                    # Convert to BGR
                    image_random = cv2.cvtColor(image_random, cv2.COLOR_GRAY2BGR)
                    
                    # Apply Gaussian blur to the noise
                    image_random = cv2.GaussianBlur(
                        image_random, 
                        (ink_mottling_gaussian_kernel, ink_mottling_gaussian_kernel), 
                        0
                    )
                    
                    # Resize to match input image size
                    if ink_mottling_noise_scale > 1:
                        image_random = cv2.resize(
                            image_random,
                            (xsize, ysize),
                            interpolation=cv2.INTER_AREA
                        )
                    
                    # Handle alpha channel if present
                    if image.shape[2] == 4:
                        image_random = np.dstack((image_random, image[:, :, 3]))
                    
                    # Blend noise with the original image
                    image_blend = cv2.addWeighted(
                        image, 
                        (1 - ink_mottling_alpha), 
                        image_random, 
                        ink_mottling_alpha, 
                        0
                    )
                    
                    # Apply blended noise only to ink areas
                    image[image_mask > 128] = image_blend[image_mask > 128]
                    
                except Exception as e:
                    print(f"Error generating noise pattern: {e}")
                
                # Convert back to grayscale if input was grayscale
                if is_gray:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # Update layer with processed image
                layer.image = image
                
            except Exception as e:
                print(f"Error applying ink mottling effect: {e}")
                # Keep original image if processing fails
        
        return meta