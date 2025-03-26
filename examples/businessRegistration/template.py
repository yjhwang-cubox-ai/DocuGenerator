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
        self.business_registration = elements.BusinessRegistrationTemplate(config.get("business_registration", {}))
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
        # short_size = np.random.randint(self.short_size[0], self.short_size[1] + 1)
        short_size = 1478
        aspect_ratio = 1.141 # A4 용지 규격
        long_size = int(short_size * aspect_ratio)
        size = (short_size, long_size)
        
        # bg_layer = self.background.generate(size)

        bg_img = Image.open("resources/business_registration/background_image/bg1_cleanup.png")
        bg_layer = layers.Layer(bg_img)
        paper_layer = layers.Layer(Image.open("resources/paper/paper_1.jpg").convert("RGBA"))

        image_group = layers.Group([bg_layer, paper_layer])
        image = image_group.merge()
        image = image.output(bbox=[0, 0, *size])
        print(",,,")


        # paper_layer, text_layers, texts = self.business_registration.generate(size)
        # document_group = layers.Group([*text_layers, paper_layer])

        # layer = document_group.merge()
        # image = layer.output(bbox=[0, 0, *size])

        # quality = np.random.randint(self.quality[0], self.quality[1] + 1)

        # data ={
        #     "image": image,
        #     "quality": quality
        # }
        # return data
        # bg layer, text layer, texts 생성
        # merge -> 이미지 생성
        # data 구성        
        # pass
    
    def init_save(self, root):
        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)

    def save(self, root, data, idx):
        pass
        # image = data["image"]
        # quality = data["quality"]

        # # output_dirpath = os.path.join(root, self.splits[split_idx])
        # # save image
        # image_filename = f"image_{idx}.jpg"
        # image_filepath = os.path.join(root, image_filename)
        # os.makedirs(os.path.dirname(image_filepath), exist_ok=True)
        # image = Image.fromarray(image[..., :3].astype(np.uint8))
        # image.save(image_filepath, quality=quality)


    def end_save(self, root):
        pass

    def format_metadata(self, image_filename: str, keys: List[str], values: List[Any]):
        pass