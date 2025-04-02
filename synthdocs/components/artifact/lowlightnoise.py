"""
version: 0.0.1

Dependencies:
    - opencv :
    - numpy :

Documentation:
    - OpenCV : https://docs.opencv.org/4.x/
    - Numpy : https://numpy.org/doc/
    - Brightness Adjustment : https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
"""
import random

import cv2
import numpy as np

from synthdocs.components.component import Component


class LowLightNoise(Component):
    """
    Simulate low-light conditions in an image by applying various noise effects.

    :param num_photons_range: Range of the number of photons to simulate.
    :type num_photons_range: tuple, optional
    :param alpha_range: Range of alpha values for brightness adjustment.
    :type alpha_range: tuple, optional
    :param beta_range: Range of beta values for brightness adjustment.
    :type beta_range: tuple, optional
    :param gamma_range: Range of gamma values for contrast adjustment.
    :type gamma_range: tuple, optional
    :param bias_range: Range of bias values to add.
    :type bias_range: tuple, optional
    :param dark_current_value: Value for dark current simulation.
    :type dark_current_value: float, optional
    :param exposure_time: Length of the simulated exposure in seconds.
    :type exposure_time: float, optional
    :param gain: Gain of the camera.
    :type gain: float, optional
    :param p: Probability of applying the effect.
    :type p: float, optional
    """

    def __init__(
        self,
        num_photons_range=(50, 100),
        alpha_range=(0.7, 1.0),
        beta_range=(10, 30),
        gamma_range=(1, 1.8),
        bias_range=(20, 40),
        dark_current_value=1.0,
        exposure_time=0.2,
        gain=0.1,
        p=1,
    ):
        """Constructor method"""
        super().__init__()
        self.num_photons_range = num_photons_range
        self.alpha_range = alpha_range
        self.beta_range = beta_range
        self.gamma_range = gamma_range
        self.bias_range = bias_range
        self.dark_current_value = dark_current_value
        self.exposure_time = exposure_time
        self.gain = gain
        self.p = p

    def _add_bias(self, image, value):
        """
        Add bias noise to the image.

        :param image: The input image.
        :type image: numpy.ndarray
        :param value: The bias value to add.
        :type value: float
        :return: Image with bias noise
        :rtype: numpy.ndarray
        """
        # Setting seed for random number generation
        shape = image.shape
        columns = np.random.randint(0, shape[1], size=15)
        # Adding constant offset to the image
        bias_im = np.zeros_like(image) + value
        # Add random column noise to the image
        col_pattern = np.random.uniform(0, int(value * 0.1) // 2, size=shape[0])
        if len(shape) > 2:
            # Skip alpha channel
            for channel in range(3):
                for column in columns:
                    bias_im[:, column, channel] = value + col_pattern
        else:
            for column in columns:
                bias_im[:, column] = value + col_pattern

        return bias_im

    def _apply_filter(self, image):
        """
        Apply a filter to the image.

        :param image: The input image.
        :type image: numpy.ndarray
        :return: Filtered image
        :rtype: numpy.ndarray
        """
        kernel = np.ones((5, 5), np.float32) / 25
        dst = cv2.filter2D(image, -1, kernel)
        return dst

    def _sensor_noise(self, image, current, exposure_time, value, gain=0.1):
        """
        Simulate sensor noise in the image.

        :param image: The input image.
        :type image: numpy.ndarray
        :param current: The dark current value.
        :type current: float
        :param exposure_time: Length of the simulated exposure in seconds.
        :type exposure_time: float
        :param value: The bias value.
        :type value: float
        :param gain: The gain of the camera.
        :type gain: float
        :return: Sensor noise
        :rtype: numpy.ndarray
        """
        base_current = current * exposure_time / gain  # Noise due to thermal heat of the sensor
        dark_im = np.random.poisson(base_current, size=image.shape)
        y_max, x_max = dark_im.shape[:2]
        n_hot = int(0.00001 * x_max * y_max)
        hot_x = np.random.randint(0, x_max, size=n_hot)
        hot_y = np.random.randint(0, y_max, size=n_hot)
        if len(dark_im.shape) > 2:
            for channel in range(3):
                dark_im[hot_y, hot_x, channel] = current * exposure_time / gain
        else:
            dark_im[hot_y, hot_x] = current * exposure_time / gain
        bias_im = self._add_bias(image.copy(), value)  # Noise due to accumulation of photon on the screen
        noise = 0.1 * bias_im + 0.1 * dark_im
        return noise

    def _simulate_low_light_image(self, image, alpha, beta, gamma, bias, photons):
        """
        Simulate a low-light image with various noise effects.

        :param image: The input image.
        :type image: numpy.ndarray
        :param alpha: The alpha value for brightness adjustment.
        :type alpha: float
        :param beta: The beta value for brightness adjustment.
        :type beta: float
        :param gamma: The gamma value for contrast adjustment.
        :type gamma: float
        :param bias: The bias value to add.
        :type bias: float
        :param photons: The number of photons.
        :type photons: int
        :return: Low-light image
        :rtype: numpy.ndarray
        """
        image = image.astype(np.float32)
        new_img = cv2.convertScaleAbs(
            image.copy(),
            alpha=alpha,
            beta=-beta,
        )  # Reducing the brightness of the image by applying a linear function
        quantum_efficiency = random.random()
        noise = np.random.poisson(quantum_efficiency * photons, size=image.shape) + self._sensor_noise(
            image,
            self.dark_current_value,
            self.exposure_time,
            bias,
            self.gain,
        )
        noise_im = np.clip(new_img.copy() + noise, 0, 255).astype(np.uint8)
        lookUpTable = np.empty((1, 256), np.uint8)
        for i in range(256):  # Reducing the contrast using gamma adjustment
            lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        res = cv2.LUT(noise_im, lookUpTable)
        output_img = self._apply_filter(res)
        return output_img

    def sample(self, meta=None):
        """Sample random parameters for the low-light noise effect.
        
        :param meta: Optional metadata dictionary with parameters to use.
        :type meta: dict, optional
        :return: Dictionary with sampled parameters.
        :rtype: dict
        """
        if meta is None:
            meta = {}
            
        # Check if we should run based on probability
        if random.random() > self.p:
            meta["run"] = False
            return meta
            
        meta["run"] = True
        
        # Sample parameters
        photons = random.randint(self.num_photons_range[0], self.num_photons_range[1])
        alpha = random.uniform(self.alpha_range[0], self.alpha_range[1])
        beta = random.uniform(self.beta_range[0], self.beta_range[1])
        gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
        bias = random.uniform(self.bias_range[0], self.bias_range[1])
        
        # Build metadata
        meta.update({
            "photons": photons,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "bias": bias,
            "dark_current_value": self.dark_current_value,
            "exposure_time": self.exposure_time,
            "gain": self.gain
        })
        
        return meta

    def apply(self, layers, meta=None):
        """Apply the low-light noise effect to layers.
        
        :param layers: The layers to apply the effect to.
        :type layers: list of Layer objects
        :param meta: Optional metadata with parameters.
        :type meta: dict, optional
        :return: Updated metadata.
        :rtype: dict
        """
        meta = self.sample(meta)
        
        # Skip processing if run is False
        if not meta.get("run", True):
            return meta
            
        # Get parameters from metadata
        photons = meta["photons"]
        alpha = meta["alpha"]
        beta = meta["beta"]
        gamma = meta["gamma"]
        bias = meta["bias"]
        
        for layer in layers:
            image = layer.image.copy()
            
            image = image.astype(np.uint8)
            
            # Handle alpha channel
            has_alpha = 0
            if len(image.shape) > 2 and image.shape[2] == 4:
                has_alpha = 1
                image, image_alpha = image[:, :, :3], image[:, :, 3]
                
            # Apply low-light simulation
            result = self._simulate_low_light_image(
                image,
                alpha,
                beta,
                gamma,
                bias,
                photons,
            )
            
            # Restore alpha channel if needed
            if has_alpha:
                result = np.dstack((result, image_alpha))
                
            # Update the layer's image
            layer.image = result
            
        return meta