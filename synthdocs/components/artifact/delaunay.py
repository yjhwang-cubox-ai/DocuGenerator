"""
version: 0.0.1
*********************************

Dependencies
- numpy
- opencv

*********************************

References:

- Scipy Documentation: https://docs.scipy.org/doc/scipy/
- Numpy Documentation: https://numpy.org/doc/

- OpenCV Documentation:  https://docs.opencv.org/4.x/

- Delaunay Tessellation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html

- Perlin Noise: https://iq.opengenus.org/perlin-noise/

*********************************

"""
import random
import warnings

import cv2
import numpy as np
from scipy import ndimage

from synthdocs.components.component import Component
from synthdocs.utils.meshgenerator import Noise
from synthdocs.utils.slidingwindow import PatternMaker

warnings.filterwarnings("ignore")


class DelaunayTessellation(Component):
    """
    The Delaunay Tessellation is a method of dividing a geometric space
    into a set of triangles. This implementation generate a Delaunay Tessellation of an image with Perlin Noise by default to create smoother,
    more organic looking tessellations.
    The Delaunay Tessellation algorithm is a method of traingulating a set of points in a planar space such that the minimum angle of
    each triangle is maximized. This ensures that the triangles are as close to equilateral as possible.
    The Delaunay Tessellation is defined as the triangulation of a set of points such that no point is inside the circumcircle of any triangle.
    The algorithm works by iteratively adding points to the triangulation and re-triangulating the set of points after each point is added.
    The algorithm ensures that the triangulation remains a Delaunay tessellation by checking for the Delaunay condition after each point is added.
    The Delaunay Condition states that the circumcircle of each triangle in the triangulation must contain no other points in its interior.
    The class inherits methods and properties from the Component base class.

    :param n_points_range: Range for the number of triangulating points from 500 to 800. Randomly selected.
    :type n_points_range: tuple (int), optional
    :param n_horizontal_points_range: Range for the number of points in the horizontal edge, from 500 to 800. The value is randomly selected.
    :type n_horizontal_points_range: tuple (int), optional
    :param n_vertical_points_range: Range for the number of points in the vertical edge, from 500 to 800. The value is randomly selected.
    :type n_vertical_points_range: tuple (int), optional
    :param noise_type: If "random", integration of Perlin Noise in the pipeline is randomly selected.
        If noise_type is "perlin", perlin noise is added to the background pattern,
        otherwise no Perlin Noise is added.
        Perlin Noise is added to the image to create a smoother, more organic looking tessellation.
    :type noise_type: string, optional
    :param color_list: A single list contains a collection of colors (in BGR) where the color of the effect will be randomly selected from it.
        Use "default" for default color or "random" for random colors.
    :type color_list: list, optional
    :param color_list_alternate: A single list contains a collection of colors (in BGR) where the alternate color of the effect will be randomly selected from it.
        Use "default" for default color or "random" for random colors.
    :type color_list_alternate: list, optional
    :param p: The probability of applying the augmentation to an input image. Default value is 1.0
    :type p: float, optional
    """

    def __init__(
        self,
        n_points_range=(500, 800),
        n_horizontal_points_range=(500, 800),
        n_vertical_points_range=(500, 800),
        noise_type="random",
        color_list="default",
        color_list_alternate="default",
        p=1,
    ):
        """Constructor method"""
        super().__init__()
        self.n_points_range = n_points_range  # no. of random points generated on the geometric plane
        self.n_horizontal_points_range = n_horizontal_points_range  # no. of horizontal edge points
        self.n_vertical_points_range = n_vertical_points_range  # no. of edge vertical points
        self.noise_type = noise_type  # apply perlin or not
        self.color_list = color_list
        self.color_list_alternate = color_list_alternate
        self.p = p

    def _edge_points(self, image):
        """
        Generate Random Points on the edge of an document image

        :param image: opencv image array
        :param length_scale: how far to space out the points in the goemetric
                             document image plane
        :param n_horizontal_points: number of points in the horizontal edge
                                    Leave as None to use length_scale to determine
                                    the value.
        :param n_vertical_points: number of points in the vertical edge
                                  Leave as None to use length_scale to determine
                                  the value
        :return: array of coordinates
        """
        ymax, xmax = image.shape[:2]
        if self.n_horizontal_points is None:
            self.n_horizontal_points = int(xmax / 200)

        if self.n_vertical_points is None:
            self.n_vertical_points = int(ymax / 200)

        delta_x = 4
        delta_y = 4

        return np.array(
            [[0, 0], [xmax - 1, 0], [0, ymax - 1], [xmax - 1, ymax - 1]]
            + [[delta_x * i, 0] for i in range(1, self.n_horizontal_points)]
            + [[delta_x * i, ymax] for i in range(1, self.n_horizontal_points)]
            + [[0, delta_y * i] for i in range(1, self.n_vertical_points)]
            + [[xmax, delta_y * i] for i in range(1, self.n_vertical_points)]
            + [[xmax - delta_x * i, ymax] for i in range(1, self.n_vertical_points)],
        )

    def apply_augmentation(self, width, height, n_points, n_horizontal_points, n_vertical_points, perlin, colors, alt_colors):
        """
        Apply the Delaunay tessellation augmentation to create a texture image.
        
        :param width: Width of the image
        :param height: Height of the image
        :param n_points: Number of points for triangulation
        :param n_horizontal_points: Number of horizontal edge points
        :param n_vertical_points: Number of vertical edge points
        :param perlin: Whether to apply Perlin noise
        :param colors: List of colors for triangles
        :param alt_colors: List of alternate colors for triangles
        :return: Generated tessellation image
        """
        # Create an empty numpy array of zeros with the given size
        img = np.ones((height, width, 3), np.uint8) * 255
        # Define some points to use for the Delaunay triangulation
        points = np.array(
            [(random.uniform(0, width), random.uniform(0, height)) for i in range(n_points)],
        )
        
        # Store the current values in instance variables for _edge_points method
        self.n_horizontal_points = n_horizontal_points
        self.n_vertical_points = n_vertical_points
        
        points = np.concatenate([points, self._edge_points(img)])
        # Perform the Delaunay triangulation on the points
        rect = (0, 0, width, height)
        subdiv = cv2.Subdiv2D(rect)
        for p in points:
            if p[0] >= 0 and p[0] < width and p[1] >= 0 and p[1] < height:
                subdiv.insert((int(p[0]), int(p[1])))

        triangles = subdiv.getTriangleList()
        triangles = triangles.astype(np.int32)

        # adding perlin noise
        if perlin:
            obj_noise = Noise()
            noise = np.array(
                [
                    [obj_noise.noise2D(j / 200, i / 200) * 50 + 200 for j in range(width)]
                    for i in range(height)
                ],
                np.float32,
            )
            # noise = np.array((noise - np.min(noise)) / (np.max(noise) - np.min(noise)) * 255 , np.uint8)
            nh, nw = noise.shape
            # Convert the blue texture to grayscale
            gray_texture = np.dot(noise[..., :3], [0.299, 0.587, 0.114])
            white_texture = np.zeros((nh, nw, 3), dtype=np.uint8)
            white_texture[..., 0] = gray_texture
            white_texture[..., 1] = gray_texture
            white_texture[..., 2] = gray_texture
            img = cv2.addWeighted(
                white_texture,
                0.1,
                img,
                0.9,
                0,
            )  # creating a white texture from the perlin noise mesh
            img = ndimage.gaussian_filter(img, sigma=(3, 3, 0), order=0)  # applying gaussian filter

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Draw the Delaunay triangulation on the empty numpy array

            for t in triangles:
                pt1 = (t[0], t[1])
                pt2 = (t[2], t[3])
                pt3 = (t[4], t[5])
                if (
                    pt1[0]
                    and pt2[0]
                    and pt3[0] <= width * 0.80
                    and (pt1[0] and pt2[0] and pt3[0] >= width * 0.40)
                ):
                    color = colors[np.random.randint(len(colors))]  # choose from colors

                elif pt1[0] and pt2[0] and pt3[0] <= width * 0.40:
                    color = alt_colors[np.random.randint(len(alt_colors))]
                else:
                    color = alt_colors[np.random.randint(len(alt_colors))]

                cv2.fillConvexPoly(img, np.array([pt1, pt2, pt3]), color)
        else:
            for t in triangles:
                pt1 = (t[0], t[1])
                pt2 = (t[2], t[3])
                pt3 = (t[4], t[5])

                if (
                    pt1[0]
                    and pt2[0]
                    and pt3[0] <= width * 0.80
                    and (pt1[0] and pt2[0] and pt3[0] >= width * 0.40)
                ):
                    color = colors[np.random.randint(len(colors))]  # choose from colors

                elif pt1[0] and pt2[0] and pt3[0] <= width * 0.40:
                    color = alt_colors[np.random.randint(len(alt_colors))]
                else:
                    color = alt_colors[np.random.randint(len(alt_colors))]
                color = colors[np.random.randint(len(colors))]  # choose from colors
                cv2.fillConvexPoly(img, np.array([pt1, pt2, pt3]), color)
        return img

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
        
        # Randomly select width and height
        width = height = meta.get("size", random.choice([400, 480, 500, 600, 640, 720]))
        
        # Sample number of points
        n_points = meta.get("n_points", random.randint(
            self.n_points_range[0],
            self.n_points_range[1],
        ))
        
        # Sample number of horizontal points
        n_horizontal_points = meta.get("n_horizontal_points", random.randint(
            self.n_horizontal_points_range[0],
            self.n_horizontal_points_range[1],
        ))
        
        # Sample number of vertical points
        n_vertical_points = meta.get("n_vertical_points", random.randint(
            self.n_vertical_points_range[0],
            self.n_vertical_points_range[1],
        ))
        
        # Determine if Perlin noise should be applied
        perlin = meta.get("perlin", None)
        if perlin is None:
            if self.noise_type == "random":
                perlin = random.choice([True, False])
            elif self.noise_type == "perlin":
                perlin = True
            else:
                perlin = False
        
        # Determine window size for pattern
        lst = [100, 120, 160]
        find_random_divisor = (
            lambda lst, b: random.choice([x for x in lst if x != 0 and b % x == 0])
            if any(x != 0 and b % x == 0 for x in lst)
            else 40
        )
        ws = meta.get("ws", find_random_divisor(lst, width))
        
        # Get color lists
        if self.color_list == "default":
            colors = [
                (250, 235, 215),
                (240, 240, 230),
                (253, 245, 230),
                (255, 245, 238),
                (255, 248, 220),
                (248, 248, 255),
                (255, 240, 245),
                (245, 255, 250),
                (255, 250, 250),
                (240, 248, 255),
                (240, 255, 255),
                (240, 255, 240),
                (255, 245, 238),
                (243, 229, 171),
                (250, 250, 210),
            ]
        elif self.color_list == "random":
            colors = [[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for _ in range(15)]
        else:
            colors = self.color_list

        if self.color_list_alternate == "default":
            alt_colors = [
                (255, 255, 240),
                (255, 250, 205),
                (238, 232, 170),
                (255, 255, 224),
                (255, 239, 213),
            ]
        elif self.color_list_alternate == "random":
            alt_colors = [[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for _ in range(5)]
        else:
            alt_colors = self.color_list_alternate
        
        # Build metadata dictionary
        meta.update({
            "width": width,
            "height": height,
            "n_points": n_points,
            "n_horizontal_points": n_horizontal_points,
            "n_vertical_points": n_vertical_points,
            "perlin": perlin,
            "ws": ws,
            "colors": colors,
            "alt_colors": alt_colors
        })
        
        return meta

    def apply(self, layers, meta=None):
        """Apply the DelaunayTessellation effect to layers.
        
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
        
        width = meta["width"]
        height = meta["height"]
        n_points = meta["n_points"]
        n_horizontal_points = meta["n_horizontal_points"]
        n_vertical_points = meta["n_vertical_points"]
        perlin = meta["perlin"]
        ws = meta["ws"]
        colors = meta["colors"]
        alt_colors = meta["alt_colors"]
        
        for layer in layers:
            image = layer.image.copy()
            
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

            # Generate the delaunay mesh
            delaunay_mesh = self.apply_augmentation(
                width, height, n_points, n_horizontal_points, n_vertical_points, perlin, colors, alt_colors
            )
            
            # Apply the mesh to the image
            h, w = image.shape[:2]
            threshold = ws // 20
            delaunay_mesh = delaunay_mesh[threshold : h - threshold, threshold : w - threshold]
            delaunay_mesh = cv2.resize(delaunay_mesh, (ws, ws), interpolation=cv2.INTER_LINEAR)
            
            sw = PatternMaker(alpha=0.49)
            result = sw.make_patterns(image=image, mesh_img=delaunay_mesh, window_size=ws)
            result = result[ws : h + ws, ws : w + ws]

            # Restore original color format
            if is_gray:
                result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            if has_alpha:
                result = np.dstack((result, image_alpha))

            # Update the layer's image
            layer.image = result
            
        return meta