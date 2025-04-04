import random

import cv2
import numpy as np
from numba import config
from numba import jit

from synthdocs.utils.lib import binary_threshold
from synthdocs.components.component import Component


class Faxify(Component):
    """Emulates faxify effect in the image.

    :param scale_range: Pair of floats determining the range from which to
           divide the resolution by.
    :type scale_range: tuple, optional
    :param monochrome: Flag to enable monochrome effect.
    :type monochrome: int, optional
    :param monochrome_method: Monochrome thresholding method.
    :type monochrome_method: string, optional
    :param monochrome_arguments: A dictionary contains argument to monochrome
            thresholding method.
    :type monochrome_arguments: dict, optional
    :param halftone: Flag to enable halftone effect.
    :type halftone: int, optional
    :param invert: Flag to invert grayscale value in halftone effect.
    :type invert: int, optional
    :param half_kernel_size: Pair of ints to determine half size of gaussian kernel for halftone effect.
    :type half_kernel_size: tuple, optional
    :param angle: Pair of ints to determine angle of halftone effect.
    :type angle: tuple, optional
    :param sigma: Pair of ints to determine sigma value of gaussian kernel in halftone effect.
    :type sigma: tuple, optional
    :param numba_jit: The flag to enable numba jit to speed up the processing in the augmentation.
    :type numba_jit: int, optional
    :param p: The probability that this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        scale_range=(1.0, 1.25),
        monochrome=[0, 1],
        monochrome_method="random",
        monochrome_arguments={},
        halftone=[0, 1],
        invert=1,
        half_kernel_size=[(1, 1), (2, 2)],
        angle=(0, 360),
        sigma=(1, 3),
        numba_jit=1,
    ):

        """Constructor method"""
        super().__init__()
        self.scale_range = scale_range
        self.monochrome = random.choice(monochrome)
        self.monochrome_method = monochrome_method
        self.monochrome_arguments = monochrome_arguments
        self.halftone = random.choice(halftone)
        self.invert = invert
        self.half_kernel_size = random.choice(half_kernel_size)
        self.angle = angle
        self.sigma = sigma
        self.numba_jit = numba_jit
        config.DISABLE_JIT = bool(1 - numba_jit)

    def cv_rotate(self, image, angle):
        """Rotate image based on the input angle.

        :param image: The image to apply the function.
        :type image: numpy.array (numpy.uint8)
        :param angle: The angle of the rotation.
        :type angle: int
        :return: Rotated image
        :rtype: numpy.array (numpy.uint8)
        """
        # image shape
        ysize, xsize = image.shape[:2]
        # center of rotation
        cx, cy = xsize // 2, ysize // 2
        # rotation matrix
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1)

        # rotation calculates the cos and sin, taking absolutes of those
        abs_cos = abs(M[0, 0])
        abs_sin = abs(M[0, 1])

        # find the new x and y bounds
        bound_x = int(ysize * abs_sin + xsize * abs_cos)
        bound_y = int(ysize * abs_cos + xsize * abs_sin)

        # subtract old image center (bringing image back to origin) and adding the new image center coordinates
        M[0, 2] += bound_x / 2 - cx
        M[1, 2] += bound_y / 2 - cy

        # warp and rotate the image
        image_rotated = cv2.warpAffine(image, M, (bound_x, bound_y))

        return image_rotated

    @staticmethod
    @jit(nopython=True, cache=True)
    def apply_halftone(image_halftone, gaussian_kernel, rotated, ysize, xsize, kernel_size):
        """Run loops and apply halftone effect in the input image.

        :param image_halftone: The image with halftone effect.
        :type image_halftone: numpy.array (numpy.uint8)
        :param gaussian_kernel: Gaussian kernel to generate the halftone effect.
        :type gaussian_kernel: numpy.array (numpy.uint8)
        :param rotated: The rotated input image
        :type rotated: numpy.array (numpy.uint8)
        :param ysize: Row numbers of the input image.
        :type ysize: int
        :param xsize: Column numbers of the input image.
        :type xsize: int
        :param kernel_size: Kernel size to generate the halftone effect.
        :type kernel_size: int
        """

        for y in range(0, ysize - kernel_size + 1, kernel_size):
            for x in range(0, xsize - kernel_size + 1, kernel_size):
                image_halftone[y : y + kernel_size, x : x + kernel_size] = (
                    np.mean(rotated[y : y + kernel_size, x : x + kernel_size]) * gaussian_kernel
                )

    # generate halftone effect
    def generate_halftone(self, image, half_kernel_size=2, angle=45, sigma=2):
        """Generate halftone effect in the input image.

        :param image: The image to apply the function.
        :type image: numpy.array (numpy.uint8)
        :param half_kernel_size: Half value of kernel size to generate halftone effect.
        :type half_kernel_size: int
        :param angle: The angle of the halftone effect.
        :type angle: int
        :param sigma: Sigma value of gaussian kernel for halftone effect.
        :type sigma: int
        :return: Image with halftone effect
        :rtype: numpy.array (numpy.uint8)
        """
        # get total width of the kernel
        kernel_size = kernel_size_x = kernel_size_y = 2 * half_kernel_size + 1

        # rotate image based on the angle
        rotated = self.cv_rotate(image, angle)

        # get new rotated image size
        ysize, xsize = rotated.shape[:2]

        # generate gaussian kernel
        image_kernel = np.zeros((kernel_size_x, kernel_size_x))
        image_kernel[half_kernel_size, half_kernel_size] = 1
        gaussian_kernel = cv2.GaussianBlur(
            image_kernel,
            (kernel_size_x, kernel_size_y),
            sigmaX=sigma,
            sigmaY=sigma,
        )
        gaussian_kernel *= 1 / np.max(gaussian_kernel)

        # initialize empty image
        image_halftone = np.zeros((ysize, xsize))

        # apply halftone effect to image
        self.apply_halftone(image_halftone, gaussian_kernel, rotated, ysize, xsize, kernel_size)

        # rotate back using negative angle
        image_halftone = self.cv_rotate(image_halftone, -angle)

        # crop the center section of image
        ysize_out, xsize_out = image_halftone.shape[:2]
        ysize_in, xsize_in = image.shape[:2]
        y_start = int((ysize_out - ysize_in) / 2)
        y_end = y_start + ysize_in
        x_start = int((xsize_out - xsize_in) / 2)
        x_end = x_start + xsize_in
        
        # Make sure indices are within bounds
        y_start = max(0, y_start)
        y_end = min(ysize_out, y_end)
        x_start = max(0, x_start)
        x_end = min(xsize_out, x_end)
        
        image_halftone = image_halftone[y_start:y_end, x_start:x_end]
        
        # Resize to match original image if needed
        if image_halftone.shape[:2] != (ysize_in, xsize_in):
            image_halftone = cv2.resize(image_halftone, (xsize_in, ysize_in))

        return image_halftone

    def complement_rgb_to_gray(self, img, invert=1, gray_level=255, max_value=255):
        """Convert RGB/BGR image to single channel grayscale image.

        :param img: The image to apply the function.
        :type img: numpy.array (numpy.uint8)
        :param invert: Flag to invert the generated grayscale value.
        :type invert: int
        :param gray_level: The selected gray value.
        :type gray_level: int
        :param max_value: Maximum value of gray value.
        :type max_value: int
        :return: Grayscale image
        :rtype: numpy.array (float)
        """

        img_complement = max_value - img

        if len(img.shape) > 2:
            img_gray = np.min(img_complement, axis=2) * (gray_level / max_value)
            img_gray[np.where(np.sum(img, axis=2) == 0)] = max_value  # if there is no color, set it to max value
        else:
            img_gray = img_complement * (gray_level / max_value)
            img_gray[np.where(img == 0)] = max_value  # if there is no color, set it to max value

        if invert:
            return (img_gray / 255).astype("float")
        else:
            return (1 - (img_gray / 255)).astype("float")

    def downscale(self, image, scale=None):
        """Downscale image based on the user input scale value.

        :param image: The image to apply the function.
        :type image: numpy.array (numpy.uint8)
        :param scale: Scale factor, if None uses random value from scale_range
        :type scale: float, optional
        :return: Downscaled image
        :rtype: numpy.array (numpy.uint8)
        """

        ysize, xsize = image.shape[:2]
        if scale is None:
            scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        new_size = (max(1, int(xsize // scale)), max(1, int(ysize // scale)))
        image_downscaled = cv2.resize(image, new_size)

        return image_downscaled

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
        
        # Determine if we should use monochrome effect
        if self.monochrome == -1:
            meta["monochrome"] = random.choice([0, 1])
        else:
            meta["monochrome"] = self.monochrome
            
        # Determine if we should use halftone effect
        if self.halftone == -1:
            meta["halftone"] = random.choice([0, 1])
        else:
            meta["halftone"] = self.halftone
            
        # Sample scale factor for downscaling
        meta["scale"] = np.random.uniform(self.scale_range[0], self.scale_range[1])
        
        # Sample monochrome parameters if needed
        if meta["monochrome"]:
            # Randomly select threshold method
            if self.monochrome_method == "random":
                all_monochrome_method = [
                    "threshold_li",
                    "threshold_mean",
                    "threshold_otsu",
                    "threshold_sauvola",
                    "threshold_triangle",
                ]
                meta["monochrome_method"] = random.choice(all_monochrome_method)
            else:
                meta["monochrome_method"] = self.monochrome_method
                
            # Copy monochrome arguments
            meta["monochrome_arguments"] = self.monochrome_arguments.copy()
            
            # Handle specific threshold methods
            if meta["monochrome_method"] == "threshold_local" and "block_size" not in meta["monochrome_arguments"]:
                # Min image size is 30
                block_size = random.randint(3, 29)
                # Block size must be odd
                if not block_size % 2:
                    block_size += 1
                meta["monochrome_arguments"]["block_size"] = block_size
                
            # Window size of niblack and sauvola must be odd
            if (meta["monochrome_method"] == "threshold_niblack") or (meta["monochrome_method"] == "threshold_sauvola"):
                if meta["monochrome_arguments"] and "window_size" in meta["monochrome_arguments"]:
                    if not meta["monochrome_arguments"]["window_size"] % 2:
                        meta["monochrome_arguments"]["window_size"] += 1
                        
            # CV2 threshold parameters
            if meta["monochrome_method"] == "cv2.threshold":
                if "thresh" not in meta["monochrome_arguments"]:
                    meta["monochrome_arguments"]["thresh"] = random.randint(64, 128)
                if "maxval" not in meta["monochrome_arguments"]:
                    meta["monochrome_arguments"]["maxval"] = 255
                if "type" not in meta["monochrome_arguments"]:
                    meta["monochrome_arguments"]["type"] = cv2.THRESH_BINARY
                    
            # CV2 adaptive threshold parameters
            if meta["monochrome_method"] == "cv2.adaptiveThreshold":
                if "maxValue" not in meta["monochrome_arguments"]:
                    meta["monochrome_arguments"]["maxValue"] = 255
                if "adaptiveMethod" not in meta["monochrome_arguments"]:
                    meta["monochrome_arguments"]["adaptiveMethod"] = random.choice(
                        (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.ADAPTIVE_THRESH_MEAN_C)
                    )
                if "thresholdType" not in meta["monochrome_arguments"]:
                    meta["monochrome_arguments"]["thresholdType"] = cv2.THRESH_BINARY
                if "blockSize" not in meta["monochrome_arguments"]:
                    block_size = random.randint(5, 29)
                    if not block_size % 2:
                        block_size += 1
                    meta["monochrome_arguments"]["blockSize"] = block_size
                if "C" not in meta["monochrome_arguments"]:
                    meta["monochrome_arguments"]["C"] = random.randint(1, 3)
        
        # Sample halftone parameters if needed
        if meta["halftone"]:
            meta["half_kernel_size"] = random.randint(
                self.half_kernel_size[0],
                self.half_kernel_size[1]
            )
            meta["angle"] = random.randint(self.angle[0], self.angle[1])
            meta["sigma"] = random.randint(self.sigma[0], self.sigma[1])
            meta["invert"] = self.invert
            
        return meta

    def apply(self, layers, meta=None):
        """Apply the faxify effect to the layers.
        
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
            try:
                # Get a copy of the image to work with
                image = layer.image.copy()
                
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
                
                # Get parameters from metadata
                monochrome = meta.get("monochrome", False)
                halftone = meta.get("halftone", False)
                scale = meta.get("scale")
                
                # Apply downscaling
                image_out = self.downscale(image, scale)
                
                # Apply monochrome effect if needed
                if monochrome:
                    monochrome_method = meta.get("monochrome_method")
                    monochrome_arguments = meta.get("monochrome_arguments", {})
                    
                    # Convert image to binary
                    try:
                        image_out = binary_threshold(
                            image_out,
                            monochrome_method,
                            monochrome_arguments,
                        )
                    except Exception as e:
                        print(f"Error applying binary threshold: {e}")
                
                # Apply halftone effect if needed
                if halftone:
                    try:
                        # Convert to gray
                        image_out = self.complement_rgb_to_gray(
                            image_out, 
                            invert=meta.get("invert", self.invert)
                        )
                        
                        # Generate halftone effect
                        image_out = self.generate_halftone(
                            image_out,
                            meta.get("half_kernel_size"),
                            meta.get("angle"),
                            meta.get("sigma")
                        )
                        
                        # Check and invert image, then return image as uint8
                        if meta.get("invert", self.invert):
                            image_out = ((1 - image_out) * 255).astype("uint8")
                        else:
                            image_out = (image_out * 255).astype("uint8")
                    except Exception as e:
                        print(f"Error applying halftone effect: {e}")
                
                # Resize back to original size
                image_faxify = cv2.resize(image_out, (image.shape[1], image.shape[0]))
                
                # Convert back to original format
                if is_gray and len(image_faxify.shape) > 2:
                    image_faxify = cv2.cvtColor(image_faxify, cv2.COLOR_BGR2GRAY)
                if has_alpha:
                    # Convert to BGRA if input has alpha layer
                    if len(image_faxify.shape) < 3:
                        image_faxify = cv2.cvtColor(image_faxify, cv2.COLOR_GRAY2BGR)
                    image_faxify = np.dstack((image_faxify, image_alpha))
                
                # Update layer with processed image
                layer.image = image_faxify
                
            except Exception as e:
                print(f"Error applying faxify effect: {e}")
                # Keep original image if processing fails
        
        return meta