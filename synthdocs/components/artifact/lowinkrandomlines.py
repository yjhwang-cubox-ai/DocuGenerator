import random
import numpy as np

from synthdocs.components.artifact.lowinkline import LowInkLine


class LowInkRandomLines(LowInkLine):
    """Adds low ink lines randomly throughout the image.

    :param count_range: Pair of ints determining the range from which the number
           of lines is sampled.
    :type count_range: tuple, optional
    :param use_consistent_lines: Whether or not to vary the width and alpha of
           generated low ink lines.
    :type use_consistent_lines: bool, optional
    :param noise_probability: The probability to add noise into the generated lines.
    :type noise_probability: float, optional
    """

    def __init__(
        self,
        count_range=(5, 10),
        use_consistent_lines=True,
        noise_probability=0.1,
    ):
        """Constructor method"""
        super().__init__(
            use_consistent_lines=use_consistent_lines if isinstance(use_consistent_lines, bool) else random.choice([k for k in use_consistent_lines]), 
            noise_probability=noise_probability
        )
        self.count_range = count_range

    def sample(self, meta=None):
        if meta is None:
            meta = {}
        
        # Get parent's meta first
        meta = super().sample(meta)
        
        # Sample count
        count = meta.get("count", random.randint(
            self.count_range[0], 
            self.count_range[1]
        ))
        
        # Update metadata
        meta["count"] = count
        
        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        
        count = meta["count"]
        
        for layer in layers:
            image = layer.image.copy()
            
            for i in range(count):
                if image.shape[0] - 1 >= 1:
                    image = self.add_transparency_line(
                        image,
                        random.randint(1, image.shape[0] - 1),
                    )
            
            # Update layer's image
            layer.image = image
            
        return meta