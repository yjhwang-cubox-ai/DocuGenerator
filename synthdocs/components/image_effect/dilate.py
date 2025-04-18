"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import numpy as np

from synthdocs import utils
from synthdocs.components.component import Component


class Dilate(Component):
    def __init__(self, k=(1, 3)):
        super().__init__()
        self.k = k

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        k = meta.get("k", np.random.randint(self.k[0], self.k[1] + 1))

        meta = {
            "k": k,
        }

        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        k = meta["k"]

        for layer in layers:
            image = utils.dilate_image(layer.image, k)
            layer.image = image

        return meta
