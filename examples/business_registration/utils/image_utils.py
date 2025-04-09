"""
이미지 처리 유틸리티 모듈
사업자 등록증 이미지 생성 과정에서 필요한 이미지 처리 기능을 제공합니다.
"""
import os
import cv2
import numpy as np
from typing import Tuple, List, Any, Dict, Optional

from PIL import Image, ImageDraw, ImageFont
from synthdocs import layers, utils
from blend_modes import normal, multiply, darken_only, lighten_only


class ImageUtils:
    """이미지 처리 유틸리티 클래스"""
    
    @staticmethod
    def make_white_transparent(layer: layers.Layer) -> np.ndarray:
        """
        이미지에서 흰 부분을 투명하게 변환
        
        Args:
            layer: 처리할 레이어
            
        Returns:
            np.ndarray: 투명 처리된 이미지
        """
        img_bgra = cv2.cvtColor(layer.image, cv2.COLOR_BGRA2RGBA)

        # 그레이스케일로 변환
        if len(layer.image.shape) > 2 and layer.image.shape[2] > 1:
            img_alpha = cv2.cvtColor(layer.image.astype(np.single), cv2.COLOR_BGR2GRAY)
        else:
            img_alpha = layer.image
        
        # 이진화할 임계값 설정
        threshold_value = 200

        # 이진화 수행
        _, img_binary = cv2.threshold(img_alpha, threshold_value, 255, cv2.THRESH_BINARY)

        # 투명도 마스크 적용
        img_bgra[:, :, 3] = ~(img_binary[:, :].astype(np.int64))
        return img_bgra
    
    @staticmethod
    def create_text_layer(
        field: str, 
        text: str, 
        position: Tuple[int, int], 
        font_size: int, 
        font_path: str,
        bold_font_path: str,
        bold: bool = False
    ) -> layers.Layer:
        """
        텍스트 레이어 생성
        
        Args:
            field: 필드 이름
            text: 표시할 텍스트
            position: 위치 좌표 (x, y)
            font_size: 폰트 크기
            font_path: 일반 폰트 경로
            bold_font_path: 볼드 폰트 경로
            bold: 볼드체 적용 여부
            
        Returns:
            layers.Layer: 생성된 텍스트 레이어
        """
        if text is None:
            text = ""
        
        text = str(text)
        
        # 볼드 여부에 따라 폰트 경로 선택
        selected_font_path = bold_font_path if bold else font_path
        font = ImageFont.truetype(selected_font_path, font_size)
        
        # 임시 이미지로 텍스트 크기 계산
        temp_img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        temp_draw = ImageDraw.Draw(temp_img)
        bbox = temp_draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # 텍스트를 위한 충분한 크기의 레이어 생성
        text_layer = layers.RectLayer((text_width + 50, text_height + 20), (0, 0, 0, 0))
        
        # 텍스트 그리기
        if text_layer.image.dtype != np.uint8:
            text_layer.image = text_layer.image.astype(np.uint8)
        img = Image.fromarray(text_layer.image)
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), text, font=font, fill=(0, 0, 0, 255))
        
        # 이미지를 다시 레이어로 변환
        text_layer.image = np.array(img)
        
        # 위치 설정
        text_layer.left = position[0]
        text_layer.top = position[1]
        
        return text_layer
    
    @staticmethod
    def split_text_to_fit_width(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
        """
        텍스트를 최대 너비에 맞게 여러 줄로 분할
        
        Args:
            text: 분할할 텍스트
            font: 폰트 객체
            max_width: 최대 너비
            
        Returns:
            List[str]: 분할된 텍스트 줄 목록
        """
        if not text:
            return []
            
        temp_img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        temp_draw = ImageDraw.Draw(temp_img)
        
        words = text.split()
        lines = []
        current_line = words[0]
        
        for word in words[1:]:
            # 현재 라인과 새 단어를 합쳤을 때 너비 계산
            test_line = current_line + " " + word
            bbox = temp_draw.textbbox((0, 0), test_line, font=font)
            text_width = bbox[2] - bbox[0]
            
            if text_width <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        
        lines.append(current_line)  # 마지막 라인 추가
        return lines
    
    @staticmethod
    def create_registration_template(registration_template_path: List[str], size: Tuple[int, int]) -> layers.Layer:
        """
        사업자 등록증 템플릿 레이어 생성
        
        Args:
            registration_template_path: 템플릿 이미지 경로 목록
            size: 원하는 크기 (너비, 높이)
            
        Returns:
            layers.Layer: 생성된 템플릿 레이어
        """
        template_img_paths = utils.search_files(registration_template_path, exts=[".jpg", ".jpeg", ".png", ".bmp"])
        selected_template_path = np.random.choice(template_img_paths)
        template_img = Image.open(selected_template_path).convert("RGBA").resize(size, Image.LANCZOS)

        # 이미지를 레이어로 변환
        template = layers.RectLayer(size, (0, 0, 0, 0))
        template.image = np.array(template_img).astype(np.float32)

        return template
    
    @staticmethod
    def apply_texture(template_layer: layers.Layer, texture_layer: layers.Layer, opacity: float = 0.7) -> None:
        """
        템플릿 레이어에 텍스처 적용
        
        Args:
            template_layer: 템플릿 레이어
            texture_layer: 텍스처 레이어
            opacity: 텍스처 적용 투명도
        """
        blended_img = multiply(template_layer.image, texture_layer.image, opacity)
        template_layer.image = blended_img
    
    @staticmethod
    def save_image(
        image: np.ndarray, 
        filepath: str, 
        quality: int = 90
    ) -> None:
        """
        이미지 저장
        
        Args:
            image: 저장할 이미지 데이터
            filepath: 저장 경로
            quality: 이미지 품질 (JPEG 압축률)
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        pil_image = Image.fromarray(image[..., :3].astype(np.uint8))
        pil_image.save(filepath, quality=quality)
    
    @staticmethod
    def generate_roi(size: Tuple[int, int]) -> np.ndarray:
        """
        관심 영역(ROI) 생성
        
        Args:
            size: 이미지 크기 (너비, 높이)
            
        Returns:
            np.ndarray: ROI 좌표 배열
        """
        return np.array([[0, 0], [size[0], 0], [size[0], size[1]], [0, size[1]]], dtype=int)