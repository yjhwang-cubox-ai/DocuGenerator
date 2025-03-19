import json
import os
import re
from typing import Any, List

import numpy as np
from synthdocs import elements
from PIL import Image
from synthdocs import components, layers, templates

class BusinessRegistration(templates.Template):
    def __init__(self, config=None):
        super().__init__(config)
        if config is None:
            config = {}
        
        self.quality = config.get("quality", [50, 95])
        self.landscape = config.get("landscape", 0.5)
        self.short_size = config.get("short_size", [720, 1024])
        self.aspect_ratio = config.get("aspect_ratio", [1, 2])    
        self.background = elements.Background(config.get("background", {}))
        self.document = elements.Document(config.get("document", {}))
        self.effect = components.Iterator(
            [
                components.Switch(components.RGB()),
                components.Switch(components.Shadow()),
                components.Switch(components.Contrast()),
                components.Switch(components.Brightness()),
                components.Switch(components.MotionBlur()),
                components.Switch(components.GaussianBlur()),
            ],
            **config.get("effect", {}),
        )
        
    def generate(self):
        # 기존에 설정한 이미지 사이즈: width=1478, height=2074,
        short_size = np.random.randint(self.short_size[0], self.short_size[1] + 1)
        aspect_ratio = 1.141 # A4 용지 규격
        long_size = int(short_size * aspect_ratio)
        size = (short_size, long_size)
        
        bg_layer = self.background.generate(size)
        # bg layer, text layer, texts 생성
        # merge -> 이미지 생성
        # data 구성        
        pass
    
    def init_save(self, root):
        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)

    def save(self, root, data, idx):
        pass

    def end_save(self, root):
        pass

    def format_metadata(self, image_filename: str, keys: List[str], values: List[Any]):
        pass