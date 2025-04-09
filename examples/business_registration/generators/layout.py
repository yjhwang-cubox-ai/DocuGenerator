"""
사업자 등록증 레이아웃 관리 모듈
필드 위치, 폰트 크기 등의 레이아웃 정보를 관리합니다.
"""
from typing import Dict, Any, Tuple

from config import (
    DEFAULT_WIDTH,
    DEFAULT_HEIGHT,
    get_field_positions,
    get_key_positions
)

class DocumentLayout:
    """문서 레이아웃 관리 클래스"""
    
    def __init__(self, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT):
        """
        레이아웃 관리자 초기화
        
        Args:
            width: 문서 너비
            height: 문서 높이
        """
        self.width = width
        self.height = height
        self.fields = get_field_positions(width, height)
        self.keys = get_key_positions(width, height)
        
    def get_field_config(self, field_name: str) -> Dict[str, Any]:
        """
        필드 구성 정보 반환
        
        Args:
            field_name: 필드 이름
            
        Returns:
            Dict[str, Any]: 위치, 폰트 크기, 볼드 여부 등의 정보
        """
        if field_name not in self.fields:
            raise ValueError(f"Unknown field: {field_name}")
        return self.fields[field_name]
    
    def get_key_config(self, key_name: str) -> Dict[str, Any]:
        """
        키 레이블 구성 정보 반환
        
        Args:
            key_name: 키 이름
            
        Returns:
            Dict[str, Any]: 위치, 텍스트, 폰트 크기, 볼드 여부 등의 정보
        """
        if key_name not in self.keys:
            raise ValueError(f"Unknown key: {key_name}")
        return self.keys[key_name]
    
    def get_field_position(self, field_name: str) -> Tuple[int, int]:
        """
        필드 위치 반환
        
        Args:
            field_name: 필드 이름
            
        Returns:
            Tuple[int, int]: (x, y) 좌표
        """
        return self.get_field_config(field_name)["position"]
    
    def get_field_font_size(self, field_name: str) -> int:
        """
        필드 폰트 크기 반환
        
        Args:
            field_name: 필드 이름
            
        Returns:
            int: 폰트 크기
        """
        return self.get_field_config(field_name)["font_size"]
    
    def is_field_bold(self, field_name: str) -> bool:
        """
        필드 볼드 여부 반환
        
        Args:
            field_name: 필드 이름
            
        Returns:
            bool: 볼드 여부
        """
        return self.get_field_config(field_name)["bold"]
    
    def get_key_position(self, key_name: str) -> Tuple[int, int]:
        """
        키 레이블 위치 반환
        
        Args:
            key_name: 키 이름
            
        Returns:
            Tuple[int, int]: (x, y) 좌표
        """
        return self.get_key_config(key_name)["position"]
    
    def get_key_text(self, key_name: str) -> str:
        """
        키 레이블 텍스트 반환
        
        Args:
            key_name: 키 이름
            
        Returns:
            str: 레이블 텍스트
        """
        return self.get_key_config(key_name)["text"]
    
    def get_key_font_size(self, key_name: str) -> int:
        """
        키 레이블 폰트 크기 반환
        
        Args:
            key_name: 키 이름
            
        Returns:
            int: 폰트 크기
        """
        return self.get_key_config(key_name)["font_size"]
    
    def is_key_bold(self, key_name: str) -> bool:
        """
        키 레이블 볼드 여부 반환
        
        Args:
            key_name: 키 이름
            
        Returns:
            bool: 볼드 여부
        """
        return self.get_key_config(key_name)["bold"]
    
    def get_all_fields(self) -> Dict[str, Dict[str, Any]]:
        """
        모든 필드 구성 정보 반환
        
        Returns:
            Dict[str, Dict[str, Any]]: 모든 필드 정보
        """
        return self.fields
    
    def get_all_keys(self) -> Dict[str, Dict[str, Any]]:
        """
        모든 키 레이블 구성 정보 반환
        
        Returns:
            Dict[str, Dict[str, Any]]: 모든 키 레이블 정보
        """
        return self.keys
    
    def calculate_list_positions(self, field_name: str, item_count: int, y_offset_step: int = 50) -> list:
        """
        리스트 형태 필드의 각 항목 위치 계산
        
        Args:
            field_name: 필드 이름
            item_count: 항목 수
            y_offset_step: 항목 간 y축 간격
            
        Returns:
            list: 각 항목의 위치 리스트
        """
        base_position = self.get_field_position(field_name)
        positions = []
        
        for i in range(item_count):
            positions.append((base_position[0], base_position[1] + i * y_offset_step))
            
        return positions

    def calculate_multiline_positions(self, field_name: str, line_count: int, y_offset_step: int = 40) -> list:
        """
        여러 줄로 나눠진 필드의 각 줄 위치 계산
        
        Args:
            field_name: 필드 이름
            line_count: 줄 수
            y_offset_step: 줄 간 y축 간격
            
        Returns:
            list: 각 줄의 위치 리스트
        """
        base_position = self.get_field_position(field_name)
        positions = []
        
        for i in range(line_count):
            positions.append((base_position[0], base_position[1] + i * y_offset_step))
            
        return positions