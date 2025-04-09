"""
텍스트 처리 유틸리티 모듈
사업자 등록증 텍스트 처리 및 포맷팅 기능을 제공합니다.
"""
import re
import json
from typing import Dict, List, Any, Tuple

from PIL import Image, ImageFont, ImageDraw


class TextUtils:
    """텍스트 처리 유틸리티 클래스"""
    
    @staticmethod
    def format_text_for_display(text: str) -> str:
        """
        텍스트를 표시용으로 포맷팅
        중복 공백 제거 및 공백 정리
        
        Args:
            text: 원본 텍스트
            
        Returns:
            str: 포맷팅된 텍스트
        """
        if not text:
            return ""
        
        # 연속된 공백을 하나로 치환
        formatted_text = re.sub(r'\s{2,}', ' ', text)
        # 좌우 공백 제거
        formatted_text = formatted_text.strip()
        return formatted_text
    
    @staticmethod
    def sort_texts_by_position(texts: List[Dict[str, Any]]) -> List[str]:
        """
        텍스트를 위치에 따라 정렬하고 텍스트 값만 추출
        
        Args:
            texts: 텍스트와 위치 정보가 담긴 딕셔너리 목록
            
        Returns:
            List[str]: 정렬된 텍스트 목록
        """
        # 텍스트를 위치(위에서 아래로, 왼쪽에서 오른쪽으로)에 따라 정렬
        sorted_texts = sorted(texts, key=lambda item: (item['position'][1], item['position'][0]))
        
        # 텍스트 값만 추출
        return [item['text'] for item in sorted_texts]
    
    @staticmethod
    def combine_texts(texts: List[str], separator: str = " ") -> str:
        """
        텍스트 목록을 하나의 문자열로 결합
        
        Args:
            texts: 결합할 텍스트 목록
            separator: 텍스트 사이에 넣을 구분자
            
        Returns:
            str: 결합된 텍스트
        """
        combined = separator.join(texts)
        # 연속된 공백 정리
        return re.sub(r"\s+", " ", combined).strip()
    
    @staticmethod
    def calculate_text_dimensions(text: str, font: ImageFont.FreeTypeFont) -> Tuple[int, int]:
        """
        텍스트의 픽셀 단위 크기 계산
        
        Args:
            text: 크기를 계산할 텍스트
            font: 사용할 폰트
            
        Returns:
            Tuple[int, int]: 너비와 높이
        """
        temp_img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        temp_draw = ImageDraw.Draw(temp_img)
        bbox = temp_draw.textbbox((0, 0), text, font=font)
        
        # 바운딩 박스에서 너비와 높이 계산
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        return width, height
    
    @staticmethod
    def format_annotation(image_filename: str, business_info: Dict[str, Any], text_sequence: str, combined_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        주석 데이터 포맷팅
        
        Args:
            image_filename: 이미지 파일명
            business_info: 사업자 정보
            text_sequence: 텍스트 시퀀스
            combined_info: 결합된 정보
            
        Returns:
            Dict[str, Any]: 포맷팅된 주석 데이터
        """
        annotation_data = {
            "image": image_filename,
            "gt": f"<HEAD>사업자등록증</HEAD>" +
                f"<사업자종류>{business_info['사업자종류']}</사업자종류>" +
                f"<사업자등록번호>{business_info['등록번호']}</사업자등록번호>" +
                f"<상호>{business_info['상호']}</상호>" +
                f"<대표자>{business_info['대표자']}</대표자>" +
                f"<개업연월일>{business_info['개업일']}</개업연월일>" +
                f"<법인등록번호>{business_info['법인등록번호']}</법인등록번호>" +
                f"<사업장소재지>{business_info['사업장주소']}</사업장소재지>" +
                f"<본점소재지>{business_info['본점주소']}</본점소재지>" +
                f"<업태>{', '.join(business_info['업태'])}</업태>" +
                f"<종목>{', '.join(business_info['종목'])}</종목>" +
                f"<발급사유>{business_info['발급사유']}</발급사유>" +
                f"<발급일자>{business_info['발급일']}</발급일자>" +
                f"<세무서명>{business_info['세무서']}</세무서명>",
            "texts": text_sequence
        }
        return annotation_data
    
    @staticmethod
    def save_annotation(filepath: str, annotation_data: Dict[str, Any]) -> None:
        """
        주석 데이터를 JSON 파일로 저장
        
        Args:
            filepath: 저장 경로
            annotation_data: 저장할 주석 데이터
        """
        with open(filepath, "w", encoding="utf-8") as annotation_file:
            json.dump(annotation_data, annotation_file, ensure_ascii=False, indent=4)
    
    @staticmethod
    def append_annotation(filepath: str, annotation_data: Dict[str, Any]) -> None:
        """
        기존 JSON 파일에 주석 데이터 추가
        
        Args:
            filepath: 파일 경로
            annotation_data: 추가할 주석 데이터
        """
        with open(filepath, "a", encoding="utf-8") as annotation_file:
            json.dump(annotation_data, annotation_file, ensure_ascii=False, indent=4)
    
    @staticmethod
    def load_annotation(filepath: str) -> Dict[str, Any]:
        """
        JSON 파일에서 주석 데이터 로드
        
        Args:
            filepath: 파일 경로
            
        Returns:
            Dict[str, Any]: 로드된 주석 데이터
        """
        with open(filepath, "r", encoding="utf-8") as annotation_file:
            return json.load(annotation_file)