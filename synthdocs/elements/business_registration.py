"""
사업자 등록증 템플릿을 생성하는 클래스
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from synthdocs import layers, components
from synthdocs.elements.content import BusinessContent
from synthdocs.elements.paper import Paper

class BusinessRegistrationTemplate:
    def __init__(self, config):
        # self.fullscreen = config.get("fullscreen", 1)
        # self.short_size = config.get("short_size", [480, 1024])
        # self.aspect_ratio = config.get("aspect_ratio", [1, 2])
        # self.content = Content(config.get("content", {}))

        # 콘텐츠 및 종이 설정
        self.content = BusinessContent(config.get("content", {}))
        self.paper = Paper(config.get("paper", {}))
        
        self.effect = components.Iterator(
            [
                components.Switch(components.ElasticDistortion()),
                components.Switch(components.AdditiveGaussianNoise()),
                components.Switch(
                    components.Selector(
                        [
                            components.Perspective(),
                            components.Perspective(),
                            components.Perspective(),
                            components.Perspective(),
                            components.Perspective(),
                            components.Perspective(),
                            components.Perspective(),
                            components.Perspective(),
                        ]
                    )
                ),
            ],
            **config.get("effect", {}),
        )
    def generate(self, size):
        width, height = size
        aspect_ratio = 1.414 # A4 용지 규격
        long_size = int(width * aspect_ratio)
        size = (width, long_size)
        
        text_layers, texts = self.content.generate(size)
        paper_layer = self.paper.generate(size)
        # self.effect.apply([*text_layers, paper_layer])

        return paper_layer, text_layers, texts