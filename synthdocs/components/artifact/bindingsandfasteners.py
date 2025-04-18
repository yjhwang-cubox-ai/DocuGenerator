import os
import random

import cv2
import numpy as np

# from augraphy import *
from synthdocs.utils.figsharedownloader import FigshareDownloader
from synthdocs.utils.lib import rotate_image_PIL
from synthdocs.components.component import Component
from synthdocs.utils import *


class BindingsAndFasteners(Component):
    """Creates binding and fastener mark in the input image."""

    def __init__(
        self,
        overlay_types="random",
        foreground=None,
        effect_type="random",
        width_range="random",
        height_range="random",
        angle_range=(-30, 30),
        ntimes=(2, 6),
        nscales=(1.0, 1.5),
        edge="random",
        edge_offset=(5, 20),
        use_figshare_library=0,
    ):
        """Constructor method"""
        super().__init__()
        self.overlay_types = overlay_types
        self.foreground = foreground
        self.effect_type = effect_type
        self.width_range = width_range
        self.height_range = height_range
        self.angle_range = angle_range
        self.ntimes = ntimes
        self.nscales = nscales
        self.edge = edge
        self.edge_offset = edge_offset
        self.use_figshare_library = use_figshare_library

    def add_noise(self, image, noise_probability, noise_value_range):
        """Add noise to black pixels of the image.

        :param image: The image to apply the function.
        :type image: numpy.array (numpy.uint8)
        :param noise_probability: The probability of applied noise.
        :type noise_probability: float
        :param noise_value: The value of applied noise.
        :type noise_value: tuple
        """

        # create mask of noise
        noise_mask = np.random.randint(noise_value_range[0], noise_value_range[1], size=image.shape).astype("uint8")
        noise_mask = cv2.GaussianBlur(noise_mask, random.choice([[5, 5], [7, 7]]), 0)

        # create probability mask
        probability_mask = np.random.random(image.shape)

        # get indices of mask
        indices = np.logical_and(probability_mask < noise_probability, image < 255)

        # apply noise mask to image
        image[indices] = noise_mask[indices]

    def create_punch_holes(self, ysize, xsize, ntimes):
        """Create effect of punch holes mark.

        :param ysize: The height of the input image.
        :type ysize: int
        :param xsize: The width of the input image.
        :type xsize: int
        :param ntimes: The number of applied binding effect.
        :type ntimes: int
        """

        # reset
        self.foreground = []

        template_size = template_size_ori = 60
        # scale template size based on image size
        # 1000 * 800 is normal image size for template size = 30
        # use max to prevent small template and min to prevent large template
        template_size = int(
            max(template_size_ori / 4, 30 * ((ysize * xsize) / (900 * 700))),
        )
        template_size = int(min(template_size, template_size_ori * 2))

        for _ in range(ntimes):
            current_template_size = random.randint(int(template_size * 0.75), int(template_size * 1.25))

            if self.width_range == "random":
                template_size_x = current_template_size
            else:
                template_size_x = random.randint(self.width_range[0], self.width_range[1])

            if self.height_range == "random":
                template_size_y = int(current_template_size / 3)
            else:
                template_size_y = random.randint(self.height_range[0], self.height_range[1])

            # create random location to merge 2 circles
            min_value = min(10, int(min(template_size_x, template_size_y) / 2))
            random_x = random.randint(min_value, template_size_x - min_value)
            random_y = random.randint(min_value, template_size_y - min_value)

            # draw circle
            image_circle = np.full(
                (template_size, template_size),
                fill_value=255,
                dtype="uint8",
            )

            circle_centroid = (int(template_size / 2), int(template_size / 2))
            circle_radius = max(int(template_size / 4) - 5, 5)
            cv2.circle(image_circle, circle_centroid, circle_radius, 0, -1)

            # add small blob noise effect
            if random.random() > 0.7:
                angle = random.randint(0, 360)
                circle_centroid_small = (
                    int(
                        circle_centroid[0] + circle_radius * np.cos(np.radians(angle)),
                    ),
                    int(
                        circle_centroid[1] + circle_radius * np.sin(np.radians(angle)),
                    ),
                )
                circle_radius_small = max(
                    int(circle_radius * random.uniform(0.1, 0.5)),
                    2,
                )
                cv2.circle(
                    image_circle,
                    circle_centroid_small,
                    circle_radius_small,
                    0,
                    -1,
                )

            # random ring effect
            new_centroid = [
                circle_centroid[0] + (random.choice([1, -1]) * random.randint(1, 5)),
                circle_centroid[1] + (random.choice([1, -1]) * random.randint(1, 5)),
            ]
            image_circle = cv2.circle(
                image_circle,
                new_centroid,
                int(circle_radius * random.uniform(0.8, 0.99)),
                random.randint(0, 10),
                random.randint(1, 5),
            )

            # add noise
            self.add_noise(
                image_circle,
                random.uniform(0.01, 0.21),
                (0, 255),
            )

            # create another copy of complement image
            image_circle_complement = 255 - image_circle

            # merge 2 circles to create non-perfect circle effect
            image_circle[random_y:, random_x:] = np.maximum(
                image_circle[random_y:, random_x:],
                image_circle_complement[:-random_y, :-random_x],
            )

            # randomly rotate to get different direction effect
            image_circle = np.rot90(image_circle, random.randint(1, 3))

            # gaussian blur
            image_circle = cv2.GaussianBlur(image_circle, (3, 3), 0)

            # convert to bgr
            image_circle_bgr = cv2.cvtColor(image_circle, cv2.COLOR_GRAY2BGR)

            self.foreground.append(image_circle_bgr)

    def create_binding_holes(self, edge, ysize, xsize, ntimes):
        """Create effect of binding holes mark.

        :param edge: The side of the binding effect.
        :type edge: string
        :param ysize: The height of the input image.
        :type ysize: int
        :param xsize: The width of the input image.
        :type xsize: int
        :param ntimes: The number of applied binding effect.
        :type ntimes: int
        """

        # reset
        self.foreground = []

        template_size = template_size_ori = 40
        # scale template size based on image size
        # 1000 * 800 is normal image size for template size = 40
        # use max to prevent small template and min to prevent large template
        template_size = int(
            max(template_size_ori / 2, 40 * ((ysize * xsize) / (1000 * 800))),
        )
        template_size = int(min(template_size, template_size_ori * 2))

        binding_effect = random.randint(0, 8)
        offset_direction = random.choice([1, -1])

        for i in range(ntimes):
            random_scale = random.uniform(1, 2)
            current_template_size = random.randint(int(template_size * 0.75), int(template_size * 1.25))

            if self.width_range == "random":
                template_size_x = current_template_size
            else:
                template_size_x = random.randint(self.width_range[0], self.width_range[1])

            if self.height_range == "random":
                template_size_y = int(current_template_size / 3)
            else:
                template_size_y = random.randint(self.height_range[0], self.height_range[1])

            # draw rectangle
            offset = int(min(template_size_x, template_size_y) / random.uniform(7, 8))
            image_rectangle = np.full(
                (template_size_y, int(template_size_x / random_scale)),
                fill_value=255,
                dtype="uint8",
            )
            image_rectangle[offset:-offset:, offset:-offset] = 0

            new_offset = offset + random.randint(3, 6)
            image_rectangle_complement = np.full(
                (template_size_y, int(template_size_x / random_scale)),
                fill_value=0,
                dtype="uint8",
            )

            if binding_effect == 0:
                x1 = new_offset + (offset_direction * random.randint(1, 3))
                x2 = image_rectangle_complement.shape[1]
                y1 = new_offset + (offset_direction * random.randint(1, 3))
                y2 = image_rectangle_complement.shape[1]
            elif binding_effect == 1:
                x1 = new_offset + (offset_direction * random.randint(1, 3))
                x2 = image_rectangle_complement.shape[1]
                y1 = 0
                y2 = -new_offset + (offset_direction * random.randint(1, 3))
            elif binding_effect == 2:
                x1 = 0
                x2 = -new_offset + (offset_direction * random.randint(1, 3))
                y1 = new_offset + (offset_direction * random.randint(1, 3))
                y2 = image_rectangle_complement.shape[1]
            elif binding_effect == 3:
                x1 = 0
                x2 = -new_offset + (offset_direction * random.randint(1, 3))
                y1 = 0
                y2 = -new_offset + (offset_direction * random.randint(1, 3))
            elif binding_effect == 4:
                x1 = new_offset + (offset_direction * random.randint(1, 3))
                x2 = -new_offset + (offset_direction * random.randint(1, 3))
                y1 = new_offset + (offset_direction * random.randint(1, 3))
                y2 = -new_offset + (offset_direction * random.randint(1, 3))
            elif binding_effect == 5:
                x1 = new_offset
                x2 = -new_offset
                y1 = new_offset
                y2 = -new_offset
            else:
                x1 = 0
                x2 = 0
                y1 = 0
                y2 = 0
            image_rectangle_complement[y1:y2:, x1:x2] = 255

            self.add_noise(
                image_rectangle,
                random.uniform(0.01, 0.21),
                (0, 255),
            )

            # merge 2 image to create offset effect
            image_rectangle = np.maximum(
                image_rectangle,
                image_rectangle_complement,
            )

            # add noise and apply blur
            self.add_noise(
                image_rectangle,
                random.uniform(0.01, 0.21),
                (0, 255),
            )
            image_rectangle = cv2.GaussianBlur(
                image_rectangle.astype("uint8"),
                (3, 3),
                cv2.BORDER_DEFAULT,
            )

            # rotate rectangle to create a more proper binding holes
            if edge == "left":
                image_rectangle = np.rot90(image_rectangle, 1)
            elif edge == "right":
                image_rectangle = np.rot90(image_rectangle, 1)

            # convert to bgr
            image_rectangle_bgr = cv2.cvtColor(image_rectangle, cv2.COLOR_GRAY2BGR)

            self.foreground.append(image_rectangle_bgr)

    def create_clips(self, edge, ysize, xsize, ntimes, edge_offset):
        """Create effect of clip mark.

        :param edge: The side of the binding effect.
        :type edge: string
        :param ysize: The height of the input image.
        :type ysize: int
        :param xsize: The width of the input image.
        :type xsize: int
        :param ntimes: The number of applied binding effect.
        :type ntimes: int
        :param  edge_offset: Offset value from each edge.
        :type  edge_offset: int
        """

        # reset
        self.foreground = []

        # minimum size
        template_size = template_size_ori = 120
        # scale template size based on image size
        # 1000 * 800 is normal image size for template size = 60
        # use max to prevent small template and min to prevent large template
        template_size = int(
            max(template_size_ori / 2, 120 * ((ysize * xsize) / (1000 * 800))),
        )
        template_size = int(min(template_size, template_size_ori * 2))

        for _ in range(ntimes):
            current_template_size = random.randint(int(template_size * 0.75), int(template_size * 1.25))

            if self.width_range == "random":
                template_size_x = current_template_size
            else:
                template_size_x = random.randint(self.width_range[0], self.width_range[1])

            if self.height_range == "random":
                template_size_y = int(current_template_size / 3)
            else:
                template_size_y = random.randint(self.height_range[0], self.height_range[1])

            # canvas
            image_clip = np.full(
                (template_size_y, template_size_x),
                fill_value=255,
                dtype="uint8",
            )

            # canvas for inner circle
            image_clip_inner = np.full(
                (template_size_y, template_size_x),
                fill_value=255,
                dtype="uint8",
            )

            cy1 = int(template_size_y / 6)
            cy2 = int(template_size_y * 5 / 6)
            cx = int(template_size_x / 2)

            # random thickness
            current_thickness = random.randint(1, 4)
            thickness_range = (current_thickness, current_thickness + 1)
            # draw circle
            circle_radius = int(min(template_size_y / 6, cx)) - 1

            # top circle
            cv2.circle(image_clip, (cx, cy1), circle_radius, 0, random.randint(thickness_range[0], thickness_range[1]))
            # bottom circle
            cv2.circle(image_clip, (cx, cy2), circle_radius, 0, random.randint(thickness_range[0], thickness_range[1]))

            # create half circle
            image_clip[cy1:cy2, :] = 255

            # inner circle centroid
            icx = cx
            icy = cy2 - int(circle_radius / 2)
            i_circle_radius = int(circle_radius * (2 / 3))
            cv2.circle(
                image_clip_inner,
                (icx, icy),
                i_circle_radius,
                0,
                random.randint(thickness_range[0], thickness_range[1]),
            )
            image_clip_inner[cy1:icy, :] = 255

            # draw lines connecting 2 circles
            left_x = int(np.floor(cx - circle_radius))
            right_x = int(np.ceil(cx + circle_radius))
            half_y_distance = int((cy2 - cy1) / 2)
            x1 = right_x
            x2 = left_x
            y11 = cy1
            y12 = cy2
            y21 = cy1
            y22 = cy2
            top_bottom = random.choice([0, 1]) > 0
            top_bottom = 0
            if top_bottom:
                y11 += half_y_distance
            else:
                y21 += half_y_distance
            image_clip = cv2.line(
                image_clip,
                (x1, y11),
                (x1, y12),
                0,
                random.randint(thickness_range[0], thickness_range[1]),
            )
            image_clip = cv2.line(
                image_clip,
                (x2, y21),
                (x2, y22),
                0,
                random.randint(thickness_range[0], thickness_range[1]),
            )

            # draw inner circle lines
            if top_bottom:
                x11 = right_x
                x12 = icx + i_circle_radius
                y11 = cy1
                y12 = icy
                x21 = icx - i_circle_radius
                x22 = icx - i_circle_radius
                y21 = icy
                y22 = cy1
            else:
                x11 = left_x
                x12 = icx - i_circle_radius
                y11 = cy1
                y12 = icy
                x21 = icx + i_circle_radius
                x22 = icx + i_circle_radius
                y21 = icy
                y22 = cy1
            image_clip_inner = cv2.line(
                image_clip_inner,
                (x11, y11),
                (x12, y12),
                0,
                random.randint(thickness_range[0], thickness_range[1]),
            )
            image_clip_inner = cv2.line(
                image_clip_inner,
                (x21, y21),
                (x22, y22),
                0,
                random.randint(thickness_range[0], thickness_range[1]),
            )

            # different clip mark effect if located at edges of image
            select_inner = 0
            if edge_offset == 0:
                # remove the top part, to create an actual clip clipping the paper if edge offset is <5
                image_clip = image_clip[y11:, :]
                select_inner = random.random() > 0.5
                # randomly choose between inner or exterior clips
                if select_inner:
                    image_clip = image_clip_inner
            else:
                # merge 2 clips mark
                image_clip = cv2.multiply(image_clip, image_clip_inner, scale=1 / 255)

            # add a little bit of noise
            self.add_noise(image_clip, 0.01, (220, 255))

            # rotate the clip to create a more realistic effect
            image_clip = rotate_image_PIL(
                image_clip,
                random.randint(self.angle_range[0], self.angle_range[1]),
                background_value=255,
                expand=1,
            )
            coordinates = np.where(image_clip == 0)
            y_min = np.min(coordinates[0])
            if edge_offset == 0 and select_inner:
                y_min += i_circle_radius
            image_clip = image_clip[y_min:, :]

            # randomly flip
            if random.choice([0, 1]) > 0:
                image_clip = np.fliplr(image_clip)

            # rotate the direction so that it looks like clipping the paper
            if edge == "left":
                image_clip = np.rot90(image_clip, 1)
            elif edge == "right":
                image_clip = np.rot90(image_clip, 3)
            elif edge == "bottom":
                image_clip = np.rot90(image_clip, 2)

            # convert to bgr
            image_clip_bgr = cv2.cvtColor(image_clip, cv2.COLOR_GRAY2BGR)

            self.foreground.append(image_clip_bgr)

    def create_triangle_clips(self, edge, ysize, xsize, ntimes, edge_offset):
        """Create effect of triangle clip mark.

        :param edge: The side of the binding effect.
        :type edge: string
        :param ysize: The height of the input image.
        :type ysize: int
        :param xsize: The width of the input image.
        :type xsize: int
        :param ntimes: The number of applied binding effect.
        :type ntimes: int
        :param  edge_offset: Offset value from each edge.
        :type  edge_offset: int
        """

        # reset
        self.foreground = []

        # minimum size
        template_size = template_size_ori = 80
        # scale template size based on image size
        # 1000 * 800 is normal image size for template size = 60
        # use max to prevent small template and min to prevent large template
        template_size = int(
            max(template_size_ori / 2, 80 * ((ysize * xsize) / (1000 * 800))),
        )
        template_size = int(min(template_size, template_size_ori * 2))

        for _ in range(ntimes):
            current_template_size = random.randint(int(template_size * 0.75), int(template_size * 1.25))

            if self.width_range == "random":
                template_size_x = int(current_template_size / 3)
            else:
                template_size_x = random.randint(self.width_range[0], self.width_range[1])

            if self.height_range == "random":
                template_size_y = current_template_size
            else:
                template_size_y = random.randint(self.height_range[0], self.height_range[1])

            # canvas
            image_clip = np.full(
                (template_size_y, template_size_x),
                fill_value=255,
                dtype="uint8",
            )

            # canvas for inner circle
            image_clip_inner = np.full(
                (template_size_y, template_size_x),
                fill_value=255,
                dtype="uint8",
            )

            cy1 = int(template_size_y * 1 / 6)
            cy2 = int(template_size_y * 5 / 6)
            cx1 = int(template_size_x * 1 / 6)
            cx2 = int(template_size_x * 1 / 2)
            cx3 = int(template_size_x * 5 / 6)

            # random thickness
            current_thickness = random.randint(1, 4)
            thickness_range = (current_thickness, current_thickness + 1)

            # radius of circle
            circle_radius = int(min(template_size_y / 6, template_size_x / 6)) - 1

            # draw 3 circles
            cv2.circle(image_clip, (cx1, cy1), circle_radius, 0, random.randint(thickness_range[0], thickness_range[1]))
            cv2.circle(image_clip, (cx3, cy1), circle_radius, 0, random.randint(thickness_range[0], thickness_range[1]))
            cv2.circle(image_clip, (cx2, cy2), circle_radius, 0, random.randint(thickness_range[0], thickness_range[1]))
            image_clip[cy1 : cy1 + circle_radius + current_thickness + 1, :] = 255
            image_clip[: cy1 + circle_radius + current_thickness + 1, cx1:cx3] = 255
            image_clip[cy1:cy2, :] = 255

            # draw inner circle
            icx = cx2
            icy = cy2 - int(circle_radius / 1)
            i_circle_radius = int(circle_radius * 2 / 3)
            cv2.circle(
                image_clip_inner,
                (icx, icy),
                i_circle_radius,
                0,
                random.randint(thickness_range[0], thickness_range[1]),
            )
            image_clip_inner[cy1:icy, :] = 255

            #  lines
            cv2.line(
                image_clip,
                (cx1, cy1 - circle_radius),
                (cx3, cy1 - circle_radius),
                0,
                random.randint(thickness_range[0], thickness_range[1]),
            )
            cv2.line(
                image_clip,
                (cx1 - circle_radius, cy1 + circle_radius),
                (cx2 - circle_radius, cy2),
                0,
                random.randint(thickness_range[0], thickness_range[1]),
            )
            cv2.line(
                image_clip,
                (cx3 + circle_radius, cy1),
                (cx2 + circle_radius, cy2),
                0,
                random.randint(thickness_range[0], thickness_range[1]),
            )

            # inner lines
            cv2.line(
                image_clip_inner,
                (cx1 - circle_radius, cy1),
                (icx - i_circle_radius, icy),
                0,
                random.randint(thickness_range[0], thickness_range[1]),
            )
            cv2.line(
                image_clip_inner,
                (cx3, cy1 + circle_radius),
                (icx + i_circle_radius, icy),
                0,
                random.randint(thickness_range[0], thickness_range[1]),
            )

            # different clip mark effect if located at edges of image
            select_inner = 0
            if edge_offset == 0:
                image_clip = image_clip[cy1:, :]
                select_inner = random.random() > 0.5
                if select_inner:
                    image_clip = image_clip_inner
            else:
                # merge 2 clips mark
                image_clip = cv2.multiply(image_clip, image_clip_inner)

            # add a little bit of noise
            self.add_noise(image_clip, 0.01, (220, 255))

            # rotate the clip to create a more realistic effect
            image_clip = rotate_image_PIL(
                image_clip,
                random.randint(self.angle_range[0], self.angle_range[1]),
                background_value=255,
                expand=1,
            )
            coordinates = np.where(image_clip == 0)
            y_min = np.min(coordinates[0])
            image_clip = image_clip[y_min:, :]

            # randomly flip
            if random.choice([0, 1]) > 0:
                image_clip = np.fliplr(image_clip)

            # rotate the direction so that it looks like clipping the paper
            if edge == "left":
                image_clip = np.rot90(image_clip, 1)
            elif edge == "right":
                image_clip = np.rot90(image_clip, 3)
            elif edge == "bottom":
                image_clip = np.rot90(image_clip, 2)

            #            image_clip = cv2.GaussianBlur(image_clip, (3,3), 0)

            # convert to bgr
            image_clip_bgr = cv2.cvtColor(image_clip, cv2.COLOR_GRAY2BGR)

            self.foreground.append(image_clip_bgr)
    
    def create_foreground(self, image, edge, edge_offset):
        """Create foreground based on current input effect type."""
        ysize, xsize = image.shape[:2]
        if self.effect_type == "random" or self.effect_type not in (
            "punch_holes",
            "binding_holes",
            "clips",
            "triangle_clips",
        ):
            effect_type = random.choice(("punch_holes", "binding_holes", "clips", "triangle_clips"))
        else:
            effect_type = self.effect_type
        ntimes = random.randint(self.ntimes[0], self.ntimes[1])
        if effect_type == "punch_holes":
            self.create_punch_holes(ysize, xsize, ntimes)
        elif effect_type == "binding_holes":
            self.create_binding_holes(edge, ysize, xsize, ntimes)
        elif effect_type == "clips":
            self.create_clips(edge, ysize, xsize, ntimes, edge_offset)
        elif effect_type == "triangle_clips":
            self.create_triangle_clips(edge, ysize, xsize, ntimes, edge_offset)

    def retrieve_foreground(self):
        """Retrieve template foreground based on current input effect type."""
        # Id for figshare published template files
        article_ID = "16668964"
        # create figshare downloader
        fsdl = FigshareDownloader(directory="figshare_BindingsAndFasteners/")
        # download files
        fsdl.download_all_files_from_article(article_ID)

        if self.effect_type == "random":
            effect_type = random.choice(("punch_holes", "binding_holes", "clips"))
        else:
            effect_type = self.effect_type

        # read foreground
        if self.effect_type == "punch_holes":
            foreground_path = os.path.join(
                os.getcwd() + "/figshare_BindingsAndFasteners/punch_hole.png",
            )
        elif self.effect_type == "binding_holes":
            foreground_path = os.path.join(
                os.getcwd() + "/figshare_BindingsAndFasteners/binding_hole.png",
            )
        elif self.effect_type == "clips" or self.effect_type == "triangle_clips":
            foreground_path = os.path.join(
                os.getcwd() + "/figshare_BindingsAndFasteners/clip.png",
            )
        self.foreground = cv2.imread(foreground_path)

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
        
        # Sample overlay type
        if self.overlay_types == "random":
            overlay_types = random.choice(
                (
                    "min", "max", "mix", "normal", "lighten", "darken", "addition",
                    "screen", "dodge", "multiply", "divide", "hard_light", 
                    "grain_merge", "overlay"
                ),
            )
        else:
            overlay_types = self.overlay_types
            
        # Sample edge
        if self.edge == "random":
            edge = random.choice(("left", "right", "top", "bottom"))
        else:
            edge = self.edge
            
        # Sample number of times
        ntimes = random.randint(self.ntimes[0], self.ntimes[1])
        
        # Process edge offset
        edge_offset_range = list(self.edge_offset)
        if edge_offset_range[0] < 1 and edge_offset_range[1] < 1:
            edge_offset_range[0] = 5  # Default value until we have image dimensions
            edge_offset_range[1] = 20  # Default value until we have image dimensions
            
        # Store sampled parameters
        meta.update({
            "overlay_types": overlay_types,
            "edge": edge,
            "ntimes": ntimes,
            "edge_offset_range": edge_offset_range,
            "nscales": self.nscales,
            "foreground_copy": self.foreground,  # We store a copy to avoid modifying original
            "effect_type": self.effect_type,
            "use_figshare_library": self.use_figshare_library
        })
        
        return meta

    def apply(self, layers, meta=None):
        """Apply the bindings and fasteners effect to the layers.
        
        :param layers: The layers to apply the effect to
        :type layers: list of Layer objects
        :param meta: Optional metadata with parameters
        :type meta: dict, optional
        :return: Updated metadata
        :rtype: dict
        """
        meta = self.sample(meta)
        
        # Skip processing if run is False
        if not meta.get("run", True):
            return meta
        
        # Get parameters from metadata
        overlay_types = meta["overlay_types"]
        edge = meta["edge"]
        ntimes = meta["ntimes"]
        edge_offset_range = meta["edge_offset_range"]
        nscales = meta["nscales"]
        foreground_copy = meta["foreground_copy"]
        effect_type = meta["effect_type"]
        use_figshare_library = meta["use_figshare_library"]
        
        for layer in layers:
            # Get a copy of the image to work with
            image = layer.image.copy()
            image = image.astype(np.uint8)
            ysize, xsize = image.shape[:2]
            
            # Update edge offset range with image dimensions if needed
            if edge_offset_range[0] < 1 and edge_offset_range[1] < 1:
                edge_offset_range[0] = np.ceil(edge_offset_range[0] * min(ysize, xsize))
                edge_offset_range[1] = np.ceil(edge_offset_range[1] * min(ysize, xsize))
            
            # Sample actual edge offset
            edge_offset = random.randint(int(edge_offset_range[0]), int(edge_offset_range[1]))
            
            # Check for alpha layer
            has_alpha = 0
            if len(image.shape) > 2:
                if image.shape[2] == 4:
                    has_alpha = 1
                    image, image_alpha = image[:, :, :3], image[:, :, 3]
            
            # Create foreground or use provided one
            foreground_working = foreground_copy  # Start with the copy
            
            # If user input image path
            if isinstance(foreground_working, str) and os.path.isfile(foreground_working):
                foreground_working = cv2.imread(foreground_working)
                ob = OverlayBuilder(
                    overlay_types,
                    foreground_working,
                    image,
                    ntimes,
                    nscales,
                    edge,
                    edge_offset,
                    1,
                )
            # If user input image
            elif isinstance(foreground_working, np.ndarray):
                ob = OverlayBuilder(
                    overlay_types,
                    foreground_working,
                    image,
                    ntimes,
                    nscales,
                    edge,
                    edge_offset,
                    1,
                )
            else:
                # User didn't input foreground or not readable file
                if use_figshare_library:
                    try:
                        self.retrieve_foreground()
                        ob = OverlayBuilder(
                            overlay_types,
                            self.foreground,
                            image,
                            ntimes,
                            nscales,
                            edge,
                            edge_offset,
                            1,
                        )
                    # If failed to download from Figshare, set to create own foreground
                    except Exception:
                        use_figshare_library = 0
                
                if not use_figshare_library:
                    self.create_foreground(image, edge, edge_offset)
                    ob = OverlayBuilder(
                        overlay_types,
                        self.foreground,
                        image,
                        ntimes,
                        nscales,
                        edge,
                        edge_offset,
                        1,
                    )
            
            # Build the overlay and apply it
            image_output = ob.build_overlay()
            
            # Restore alpha channel if needed
            if has_alpha:
                ysize_out, xsize_out = image_output.shape[:2]
                if ysize_out != image_alpha.shape[0] or xsize_out != image_alpha.shape[1]:
                    image_alpha = cv2.resize(image_alpha, (xsize_out, ysize_out), interpolation=cv2.INTER_AREA)
                image_output = np.dstack((image_output, image_alpha))
            
            # Update the layer's image
            layer.image = image_output
        
        return meta