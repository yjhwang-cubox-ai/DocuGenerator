"""
사업자 등록증 이미지 생성기
항목별 다른 폰트 크기 적용이 가능한 사업자 등록증 이미지 생성 모듈
"""
import os
import re
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from synthdocs import components, layers, templates, utils
from synthdocs.elements import Background, Paper
from PIL import Image, ImageFont

from config import (
    DEFAULT_WIDTH, 
    DEFAULT_HEIGHT, 
    DEFAULT_QUALITY, 
    DEFAULT_APPLY_TEXTURE_PROB,
    DEFAULT_SPLIT_RATIO,
    DEFAULT_SPLITS,
    DEFAULT_FONT_PATH,
    DEFAULT_BOLD_FONT_PATH,
    DEFAULT_TEMPLATE_PATHS
)
from generators.business_info_generator import BusinessInfoGenerator
from generators.layout import DocumentLayout
from effects.ink_effects import InkEffects, create_ink_effects
from effects.paper_effects import PaperEffects
from effects.post_effects import PostEffects
from utils.image_utils import ImageUtils
from utils.text_utils import TextUtils


class BusinessRegistration(templates.Template):
    """사업자 등록증 템플릿 클래스"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, split_ratio: List[float] = DEFAULT_SPLIT_RATIO):
        """
        사업자 등록증 템플릿 초기화
        
        Args:
            config: 템플릿 설정
            split_ratio: 데이터셋 분할 비율 (학습/검증/테스트)
        """
        super().__init__(config)
        if config is None:
            config = {}

        # 기본 설정
        self.quality = config.get("quality", DEFAULT_QUALITY)
        self.landscape = 1.0  # 사업자 등록증은 가로 방향으로 고정
        self.apply_texture = config.get("apply_texture", DEFAULT_APPLY_TEXTURE_PROB)
        self.width = config.get("width", DEFAULT_WIDTH)
        self.height = config.get("height", DEFAULT_HEIGHT)
        
        # 배경 및 종이 설정
        self.background = Background(config.get("background", {}))
        self.paper = Paper(config.get("paper", {}))
        
        # 효과 컴포넌트 설정
        self.ink_effects = InkEffects(config)
        self.paper_effects = PaperEffects(config)
        self.post_effects = PostEffects(config)
        self.general_effects = self._create_general_effects(config)
        
        # 사업자 등록증 배경이미지 경로
        self.registration_template_path = config.get("business_registration", {}).get(
            "template_image", {}).get("paths", DEFAULT_TEMPLATE_PATHS
        )
        
        # 글꼴 설정
        self.font_path = config.get("font_path", DEFAULT_FONT_PATH)
        self.bold_font_path = config.get("bold_font_path", DEFAULT_BOLD_FONT_PATH)
        
        # 레이아웃 설정
        self.layout = DocumentLayout(self.width, self.height)
        
        # 데이터셋 분할 설정
        self.splits = DEFAULT_SPLITS
        self.split_ratio = split_ratio
        self.split_indexes = np.random.choice(3, size=10000, p=split_ratio)
        
        # 사업자 정보 생성기
        self.business_info_generator = BusinessInfoGenerator(
            dataset_path=config.get("business_registration", {}).get("business_info", {}).get("path", [])
        )

    def _create_general_effects(self, config: Dict[str, Any]) -> components.Iterator:
        """
        일반 효과 컴포넌트 생성
        
        Args:
            config: 효과 설정
            
        Returns:
            components.Iterator: 일반 효과 컴포넌트
        """
        return components.Iterator(
            [
                components.Switch(components.RGB()),
                components.Switch(components.Shadow()),
                components.Switch(components.Contrast()),
                components.Switch(components.RGBShiftBrightness()),
                components.Switch(components.MotionBlur()),
                components.Switch(components.GaussianBlur()),
            ],
            **config.get("effect", {}),
        )

    def generate(self, business_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        사업자 등록증 이미지 생성
        
        Args:
            business_info: 사업자 정보 (없으면 랜덤 생성)
            
        Returns:
            Dict[str, Any]: 생성된 이미지와 메타데이터
        """
        # 사업자 등록증 크기 설정
        size = (self.width, self.height)

        # 배경 레이어 생성
        bg_layer = layers.RectLayer(size, (255, 255, 255, 255))
        
        # 사업자 등록증 양식 이미지 생성
        template_layer = ImageUtils.create_registration_template(self.registration_template_path, size)
        
        # 무작위 사업자 정보 생성 또는 전달받은 정보 사용
        if business_info is None:
            business_info = self.business_info_generator.generate_random_info()
        
        # 텍스트 레이어 생성
        text_layers, texts, combined_info = self._create_text_layers(business_info)
        
        # 텍스트 레이어 결합 & 텍스트 효과 적용
        text_group = layers.Group([*text_layers, bg_layer])
        text_layer = text_group.merge()
        self.ink_effects.apply([text_layer])
        text_layer.image = ImageUtils.make_white_transparent(text_layer)

        # 배경 texture 적용
        texture_layer = self.paper.generate(size)
        if np.random.rand() < self.apply_texture:
            ImageUtils.apply_texture(template_layer, texture_layer)
            
        # 배경 레이어에 종이 효과 적용
        self.paper_effects.apply([template_layer])
        
        # 모든 레이어 결합
        document_group = layers.Group([text_layer, template_layer])
        layer = document_group.merge()
        
        # 후처리 효과 적용
        self.post_effects.apply([layer])
        
        # 최종 이미지 출력
        image = layer.output(bbox=[0, 0, *size])
        
        # 텍스트 순서 정렬 및 레이블 생성
        ordered_texts = TextUtils.sort_texts_by_position(texts)
        label = TextUtils.combine_texts(ordered_texts)
        
        # 이미지 품질 설정
        quality = np.random.randint(self.quality[0], self.quality[1] + 1)
        
        # ROI (관심 영역) 생성
        roi = ImageUtils.generate_roi(size)
        
        # 결과 데이터 구성
        data = {
            "image": image,
            "label": label,
            "quality": quality,
            "roi": roi,
            "business_info": business_info,
            "combined_info": combined_info
        }
        
        return data

    def _create_text_layers(self, business_info: Dict[str, Any]) -> Tuple[List[layers.Layer], List[Dict[str, Any]], Dict[str, Any]]:
        """
        텍스트 레이어 생성
        
        Args:
            business_info: 사업자 정보
            
        Returns:
            Tuple: 텍스트 레이어 목록, 텍스트 정보 목록, 통합 정보
        """
        text_layers = []
        texts = []
        combined_info = {}  # key와 value를 모두 저장할 딕셔너리

        # 1. 키 레이어 생성
        for field, config in self.layout.get_all_keys().items():
            key_layer = ImageUtils.create_text_layer(
                f"key_{field}",
                config["text"],
                config["position"],
                config["font_size"],
                self.font_path,
                self.bold_font_path,
                config["bold"]
            )
            text_layers.append(key_layer)
            texts.append({
                "text": TextUtils.format_text_for_display(config["text"]), 
                "position": config["position"]
            })
            
            # combined_info에 키 추가
            combined_info[f"key_{field}"] = field

        # 2. 값 레이어 생성
        for field, value in business_info.items():
            if field in self.layout.get_all_fields():
                field_config = self.layout.get_field_config(field)
                
                if isinstance(value, list):
                    # 리스트 형태의 값(업태, 종목 등) 처리
                    positions = self.layout.calculate_list_positions(
                        field, len(value), y_offset_step=50
                    )

                    for i, (item, position) in enumerate(zip(value, positions)):
                        text_layer = ImageUtils.create_text_layer(
                            f"{field}_{i}", 
                            item, 
                            position, 
                            field_config["font_size"],
                            self.font_path,
                            self.bold_font_path,
                            field_config["bold"]
                        )
                        text_layers.append(text_layer)
                        texts.append({
                            "text": TextUtils.format_text_for_display(item), 
                            "position": position
                        })

                        # combined_info에 값 추가
                        combined_info[f"value_{field}_{i}"] = item
                else:
                    # 주소 필드 처리 (줄바꿈)
                    if field in ["사업장주소", "본점주소"]:
                        font = ImageFont.truetype(
                            self.bold_font_path if field_config["bold"] else self.font_path, 
                            field_config["font_size"]
                        )
                        
                        # 주소 텍스트를 여러 줄로 분할
                        max_width = int(self.width * 0.6)
                        text_lines = ImageUtils.split_text_to_fit_width(str(value), font, max_width)
                        
                        # 각 줄 별로 레이어 생성
                        positions = self.layout.calculate_multiline_positions(
                            field, len(text_lines), y_offset_step=40
                        )
                        
                        for i, (line, position) in enumerate(zip(text_lines, positions)):
                            text_layer = ImageUtils.create_text_layer(
                                f"{field}_line{i}", 
                                line, 
                                position,
                                field_config["font_size"],
                                self.font_path,
                                self.bold_font_path,
                                field_config["bold"]
                            )
                            text_layers.append(text_layer)
                            texts.append({
                                "text": TextUtils.format_text_for_display(line), 
                                "position": position
                            })
                            
                            # combined_info에 값 추가
                            combined_info[f"value_{field}_line{i}"] = line
                    else:
                        # 일반 필드 처리
                        text_layer = ImageUtils.create_text_layer(
                            field, 
                            str(value), 
                            field_config["position"],
                            field_config["font_size"],
                            self.font_path,
                            self.bold_font_path,
                            field_config["bold"]
                        )
                        text_layers.append(text_layer)
                        texts.append({
                            "text": TextUtils.format_text_for_display(value), 
                            "position": field_config["position"]
                        })

                        # combined_info에 값 추가
                        combined_info[f"value_{field}"] = value
                        
        return text_layers, texts, combined_info

    def init_save(self, root: str) -> None:
        """
        저장 초기화
        
        Args:
            root: 저장 경로
        """
        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)

    def save(self, root: str, data: Dict[str, Any], idx: int) -> None:
        """
        데이터 저장
        
        Args:
            root: 저장 경로
            data: 저장할 데이터
            idx: 인덱스
        """
        image = data["image"]
        label = data["label"]
        quality = data["quality"]
        business_info = data["business_info"]
        combined_info = data["combined_info"]

        # 데이터셋 분할
        split_idx = self.split_indexes[idx % len(self.split_indexes)]
        output_dirpath_images = os.path.join(root, self.splits[split_idx], 'images')
        output_dirpath_annotations = os.path.join(root, self.splits[split_idx], 'annotations')

        # 이미지 저장
        image_filename = f"business_reg_{idx}.jpg"
        image_filepath = os.path.join(output_dirpath_images, image_filename)
        ImageUtils.save_image(image, image_filepath, quality)
        
        # 주석 데이터 저장
        annotation_filename = f"{image_filename.split('.')[0]}.json"
        annotation_filepath = os.path.join(output_dirpath_annotations, annotation_filename)
        os.makedirs(os.path.dirname(annotation_filepath), exist_ok=True)

        annotation_data = TextUtils.format_annotation(
            image_filename=image_filename, 
            business_info=business_info, 
            text_sequence=label, 
            combined_info=combined_info
        )
        TextUtils.append_annotation(annotation_filepath, annotation_data)

    def end_save(self, root: str) -> None:
        """
        저장 종료
        
        Args:
            root: 저장 경로
        """
        pass


if __name__ == "__main__":
    # 테스트 코드
    generator = BusinessRegistration()
    data = generator.generate()
    
    # 결과 저장
    os.makedirs("output", exist_ok=True)
    ImageUtils.save_image(data["image"], "output/test_business_reg.jpg")
    
    print(f"생성된 이미지 크기: {data['image'].shape}")
    print(f"텍스트 길이: {len(data['label'])}")
    print("이미지 생성 완료!")