"""
잉크 효과 관련 컴포넌트 모듈
사업자 등록증의 텍스트에 적용할 잉크 관련 효과를 정의합니다.
"""
from typing import Dict, List, Any, Optional

from synthdocs import components

def create_ink_effects(config: Optional[Dict[str, Any]] = None) -> components.Iterator:
    """
    잉크 효과 컴포넌트 생성
    
    Args:
        config: 효과 설정
        
    Returns:
        components.Iterator: 잉크 효과 컴포넌트
    """
    if config is None:
        config = {}
        
    return components.Iterator(
        [
            components.Switch(components.InkColorSwap()),
            components.Switch(components.LinesDegradation()),
            components.Switch(
                components.Selector(
                    [
                        components.Dithering(),
                        components.InkBleed(),
                    ]
                ),
            ),
            components.Switch(
                components.Selector(
                    [
                        components.InkShifter(),
                        components.BleedThrough(),
                    ]
                ),
            ),
            components.Switch(
                components.Selector(
                    [
                        components.Hollow(),
                        components.Letterpress(),
                    ]
                ),
            ),
            components.Switch(
                components.Selector(
                    [
                        components.LowInkRandomLines(),
                        components.LowInkPeriodicLines(),
                    ]
                ),
            ),
        ],            
        **config.get("ink_effect", {}),
    )


class InkEffects:
    """잉크 효과 관리 클래스"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        잉크 효과 초기화
        
        Args:
            config: 효과 설정
        """
        self.config = config or {}
        self.effects = create_ink_effects(self.config)
    
    def apply(self, layers: List) -> None:
        """
        레이어에 잉크 효과 적용
        
        Args:
            layers: 효과를 적용할 레이어 목록
        """
        self.effects.apply(layers)