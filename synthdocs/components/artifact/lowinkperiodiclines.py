import random

from synthdocs.components.artifact.lowinkline import LowInkLine


class LowInkPeriodicLines(LowInkLine):
    """Creates a set of lines that repeat in a periodic fashion throughout the
    image.

    :param count_range: Pair of ints determining the range from which to sample
           the number of lines to apply.
    :type count_range: tuple, optional
    :param period_range: Pair of ints determining the range from which to sample
           the distance between lines.
    :type period_range: tuple, optional
    :param use_consistent_lines: Whether or not to vary the width and alpha of
           generated low ink lines.
    :type use_consistent_lines: bool, optional
    :param noise_probability: The probability to add noise into the generated lines.
    :type noise_probability: float, optional
    """

    def __init__(
        self,
        count_range=(2, 5),
        period_range=(10, 30),
        use_consistent_lines=True,
        noise_probability=0.1,
    ):
        """Constructor method"""
        super().__init__(
            use_consistent_lines=use_consistent_lines,
            noise_probability=noise_probability,
        )
        self.count_range = count_range
        self.period_range = period_range

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"LowInkPeriodicLines(count_range={self.count_range}, period_range={self.period_range}, use_consistent_lines={self.use_consistent_lines})"

    def add_periodic_transparency_lines(self, mask, line_count, period):
        """Creates horizontal lines of some opacity over the input image, at y-positions determined by the period.

        :param mask: The image to apply the line to.
        :type mask: numpy.array
        :param line_count: The number of lines to generate.
        :type line_count: int
        :param period: The distance between lines.
        :type period: int
        """
        for i in range(line_count):
            y_position = i * period
            if y_position < mask.shape[0]:
                alpha = random.randint(16, 255)
                mask = self.add_transparency_line(mask, y_position, alpha)
        return mask

    def sample(self, meta=None):
        if meta is None:
            meta = {}
        
        # Sample count and period
        count = random.randint(self.count_range[0], self.count_range[1])
        period = random.randint(self.period_range[0], self.period_range[1])
        
        # Update metadata
        meta["count"] = count
        meta["period"] = period
        
        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        
        count = meta["count"]
        period = meta["period"]

        for layer in layers:
            image = layer.image.copy()

            for i in range(count):
                if image.shape[0] - 1 >= 1:  # 이미지 크기 체크 추가
                    image = self.add_periodic_transparency_lines(
                        image,
                        random.randint(1, image.shape[0] - 1),
                        period,
                    )
            
            # Update layer's image
            layer.image = image

        return meta