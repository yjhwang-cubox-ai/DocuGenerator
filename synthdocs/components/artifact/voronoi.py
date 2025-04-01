"""
version: 0.0.1
*********************************

Dependencies
- numpy
- PIL
- numba
- opencv

*********************************

References:

- Numpy Documentation: https://numpy.org/doc/
- PIL Documentation: https://pillow.readthedocs.io/en/stable/

- Numba Documentation: https://numba.readthedocs.io/en/stable/

- OpenCV Documentation:  https://docs.opencv.org/4.x/

- Voronoi Tessellation: a. https://en.wikipedia.org/wiki/Voronoi_diagram
                        b. https://www.generativehut.com/post/robots-and-generative-art-and-python-oh-my
                        c. https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Voronoi.html

- Perlin Noise: https://iq.opengenus.org/perlin-noise/

*********************************
"""
import os
import random
import warnings

import cv2
import numba as nb
import numpy as np
from numba import config
from numba import jit
from PIL import Image

from synthdocs.components.component import Component
from synthdocs.utils.meshgenerator import Noise
from synthdocs.utils.slidingwindow import PatternMaker

warnings.filterwarnings("ignore")


class VoronoiTessellation(Component):
    """
    This script generates a Voronoi Tessellation based on a set of random points in a plane. The tessellation
    is visualized by coloring or shading the region around each point with the color or shade of the corresponding
    random point. By default, Perlin Noise is added to the distances between each point and its closest random
    point to create a smoother, more organic looking tessellation.
    The class inherits methods and properties from the Component base class.

    :param mult_range: range for amplification factor to generate Perlin noise , default lies between 50 and 80
    :type mult_range: tuple (int), optional
    :param seed: The seed value for generating the Perlin Noise, default value is 19829813472
    :type seed: int, optional
    :param num_cells_range: Range for the number of cells used to generate the Voronoi Tessellation. Default
                            lies between 1000 and 9000.
    :type num_cells_range: tuple (int), optional
    :param noise_type: If "random", integration of Perlin Noise in the pipeline is randomly selected.
                       If noise_type is "perlin", perlin noise is added to the background pattern,
                       otherwise no Perlin Noise is added.Perlin Noise is added to the image to create a smoother,
                       more organic looking tessellation.
    :type noise_type: string, optional
    :param background_value: Range for background color assigned to each point
    :type background_value: tuple (int)
    :param numba_jit: The flag to enable numba jit to speed up the processing in the augmentation.
    :type numba_jit: int, optional
    :param p: The probability of applying the augmentation to an input image. Default value is 1.0
    :type p: float
    """

    def __init__(
        self,
        mult_range=(50, 80),
        seed=19829813472,
        num_cells_range=(500, 1000),
        noise_type="random",
        background_value=(200, 255),
        numba_jit=1,
        p=1,
    ):
        """Constructor method"""
        super().__init__()
        self.mult_range = mult_range
        self.seed = seed
        self.num_cells_range = num_cells_range
        self.noise_type = noise_type
        self.background_value = background_value
        self.numba_jit = numba_jit
        self.p = p
        config.DISABLE_JIT = bool(1 - numba_jit)

    @staticmethod
    @jit(nopython=True, cache=True, parallel=True)
    def generate_voronoi(width, height, num_cells, nsize, pixel_data, perlin_noise_2d):
        """
        Generates Voronoi Tessellation

        :param width: Width of the image
        :type width: int
        :param height: Height of the image
        :type height: int
        :param num_cells: Number of cells for tessellation
        :type num_cells: int
        :param nsize: Array to store cell sizes
        :type nsize: numpy.ndarray
        :param pixel_data: Tuple of (x_coordinates, y_coordinates, colors)
        :type pixel_data: tuple
        :param perlin_noise_2d: Tuple of (x_noise, y_noise)
        :type perlin_noise_2d: tuple
        :return: Generated Voronoi tessellation image
        :rtype: numpy.ndarray
        """
        img_array = np.zeros((width, height), dtype=np.uint8)
        for y in nb.prange(width):
            for x in nb.prange(height):
                dmin = np.hypot(height, width)
                for i in nb.prange(num_cells):
                    d = np.hypot(
                        (pixel_data[0][i] - x + perlin_noise_2d[0][x][y]),
                        (pixel_data[1][i] - y + perlin_noise_2d[1][x][y]),
                    )
                    if d < dmin:
                        dmin = d
                        j = i
                    nsize[j] += 1
                img_array[y][x] = pixel_data[2][j]
        return img_array

    def apply_augmentation(self, width, height, mult, num_cells, perlin, background_value, seed):
        """
        Apply the Voronoi tessellation augmentation to create a texture image.
        
        :param width: Width of the image
        :type width: int
        :param height: Height of the image
        :type height: int
        :param mult: Amplification factor for Perlin noise
        :type mult: int
        :param num_cells: Number of cells for tessellation
        :type num_cells: int
        :param perlin: Whether to apply Perlin noise
        :type perlin: bool
        :param background_value: Range for background color
        :type background_value: tuple
        :param seed: Seed value for Perlin noise
        :type seed: int
        :return: Generated tessellation image
        :rtype: numpy.ndarray
        """
        obj_noise = Noise()
        perlin_x = np.zeros((height, width))
        perlin_y = np.zeros((height, width))
        if perlin:
            perlin_x = np.array(
                [
                    [obj_noise.noise2D(x / 100, y / 100) * mult for y in range(height)]
                    for x in range(width)
                ],
            )
            perlin_y = np.array(
                [
                    [
                        obj_noise.noise2D((x + seed) / 100, (y + seed) / 100) * mult
                        for y in range(height)
                    ]
                    for x in range(width)
                ],
            )
        nx = [random.randrange(width) for _ in range(num_cells)]  # x-coordinates of random points
        ny = [random.randrange(height) for _ in range(num_cells)]  # y-coordinates of random points
        ng = [
            random.randrange(background_value[0], background_value[1]) for _ in range(num_cells)
        ]  # go through the number of cells and assign color
        nsize = np.zeros(num_cells, dtype=np.int32)
        img_array = self.generate_voronoi(
            width,
            height,
            num_cells,
            nsize,
            (nx, ny, ng),
            (perlin_x, perlin_y),
        )
        # try if it is able to save and read image properly, might facing permission issue
        try:
            image = Image.fromarray(img_array)
            image.save(os.getcwd() + "/Voronoi_example.png", "PNG", dpi=(300, 300))
            # reads it in a format so that it can be applied as a background pattern to the original image
            mesh = cv2.imread(os.getcwd() + "/Voronoi_example.png")
            os.remove(os.getcwd() + "/Voronoi_example.png")
        except Exception:
            mesh = img_array
        return mesh

    def sample(self, meta=None):
        """Sample random parameters for the augmentation.
        
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
        
        # Determine if Perlin noise should be applied
        perlin = meta.get("perlin", None)
        if perlin is None:
            if self.noise_type == "random":
                perlin = random.choice([True, False])
            elif self.noise_type == "perlin":
                perlin = True
            else:
                perlin = False
        
        # Set width and height based on perlin flag
        if perlin:
            width = height = meta.get("size", random.choice([100, 120, 140, 160, 180, 200]))
            lst = [50, 70, 80, 90]
        else:
            width = height = meta.get("size", random.choice(
                [200, 210, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400]
            ))
            lst = [100, 120, 140, 150, 160]
            
        # Find random divisor for window size
        find_random_divisor = (
            lambda lst, b: random.choice([x for x in lst if x != 0 and b % x == 0])
            if any(x != 0 and b % x == 0 for x in lst)
            else 40
        )
        ws = meta.get("ws", find_random_divisor(lst, width))
        
        # Sample other parameters
        mult = meta.get("mult", random.randint(self.mult_range[0], self.mult_range[1]))
        num_cells = meta.get("num_cells", random.randint(
            self.num_cells_range[0], 
            self.num_cells_range[1]
        ))
        
        # Build metadata
        meta.update({
            "perlin": perlin,
            "width": width,
            "height": height,
            "mult": mult,
            "num_cells": num_cells,
            "ws": ws,
            "seed": self.seed,
            "background_value": self.background_value
        })
        
        return meta

    def apply(self, layers, meta=None):
        """Apply the VoronoiTessellation effect to layers.
        
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
        perlin = meta["perlin"]
        width = meta["width"]
        height = meta["height"]
        mult = meta["mult"]
        num_cells = meta["num_cells"]
        ws = meta["ws"]
        seed = meta["seed"]
        background_value = meta["background_value"]
        
        for layer in layers:
            image = layer.image.copy()
            
            # Check for alpha channel
            has_alpha = 0
            if len(image.shape) > 2 and image.shape[2] == 4:
                has_alpha = 1
                image, image_alpha = image[:, :, :3], image[:, :, 3]
            
            h, w = image.shape[:2]
            
            # Generate the Voronoi mesh
            voronoi_mesh = self.apply_augmentation(
                width, height, mult, num_cells, perlin, background_value, seed
            )
            
            # Resize mesh to window size
            voronoi_mesh = cv2.resize(voronoi_mesh, (ws, ws), interpolation=cv2.INTER_LINEAR)
            
            # Adjust mesh format to match image
            if len(image.shape) < 3 and len(voronoi_mesh.shape) > 2:
                voronoi_mesh = cv2.cvtColor(voronoi_mesh, cv2.COLOR_RGB2GRAY)
            elif len(image.shape) > 2 and len(voronoi_mesh.shape) < 3:
                voronoi_mesh = cv2.cvtColor(voronoi_mesh, cv2.COLOR_GRAY2BGR)
            
            # Apply pattern to image
            sw = PatternMaker()
            # to ensure the voronoi tessellation covers the whole image,
            # original image is padded and voronoi_mesh passes through it like a sliding window
            result = sw.make_patterns(image, voronoi_mesh, ws)
            result = result[ws : h + ws, ws : w + ws]
            
            # Restore alpha channel if needed
            if has_alpha:
                result = np.dstack((result, image_alpha))
                
            # Update the layer's image
            layer.image = result
            
        return meta