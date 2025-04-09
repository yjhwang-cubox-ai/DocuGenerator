"""
종이 효과 관련 컴포넌트 모듈
사업자 등록증 배경에 적용할 종이 질감 및 효과를 정의합니다.
"""
from typing import Dict, List, Any

from synthdocs import components

class PaperEffects:
    """종이 효과 관리 클래스"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        종이 효과 초기화
        
        Args:
            config: 효과 설정
        """
        if config is None:
            config = {}
            
        self.effects = self._create_paper_effects(config)
    
    def _create_paper_effects(self, config: Dict[str, Any]) -> components.Iterator:
        """
        종이 효과 컴포넌트 생성
        
        Args:
            config: 효과 설정
            
        Returns:
            components.Iterator: 종이 효과 컴포넌트
        """
        return components.Iterator(
            [
                components.Switch(components.ColorPaper()),
                components.Switch(
                    components.Selector(
                        [
                            components.DelaunayTessellation(),
                            components.PatternGenerator(),
                            components.VoronoiTessellation(),                            
                        ]
                    ),
                ),
                components.Switch(components.WaterMark()),
                components.Switch(
                    components.Selector(
                        [
                            components.Iterator(
                                [
                                    components.NoiseTexturize(),
                                    components.BrightnessTexturize(),
                                ]
                            ),
                            components.Iterator(
                                [
                                    components.BrightnessTexturize(),
                                    components.NoiseTexturize(),
                                ]
                            )
                        ]
                    ),
                ),
            ],
            **config.get("paper_effect", {}),
        )
    
    def apply(self, layers: List) -> None:
        """
        레이어에 종이 효과 적용
        
        Args:
            layers: 효과를 적용할 레이어 목록
        """
        self.effects.apply(layers)
        
    def get_component(self) -> components.Iterator:
        """
        종이 효과 컴포넌트 반환
        
        Returns:
            components.Iterator: 종이 효과 컴포넌트
        """
        return self.effects