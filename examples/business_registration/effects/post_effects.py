"""
후처리 효과 관련 컴포넌트 모듈
사업자 등록증 이미지에 적용할 후처리 효과를 정의합니다.
"""
from typing import Dict, List, Any

from synthdocs import components

class PostEffects:
    """후처리 효과 관리 클래스"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        후처리 효과 초기화
        
        Args:
            config: 효과 설정
        """
        if config is None:
            config = {}
            
        self.effects = self._create_post_effects(config)
    
    def _create_post_effects(self, config: Dict[str, Any]) -> components.Iterator:
        """
        후처리 효과 컴포넌트 생성
        
        Args:
            config: 효과 설정
            
        Returns:
            components.Iterator: 후처리 효과 컴포넌트
        """
        return components.Iterator(
            [
                components.Switch(
                    components.Selector(
                        [
                            components.GlitchEffect(),
                            components.ColorShift(),
                        ]
                    ),
                ),
                components.Switch(
                    components.Selector(
                        [
                            components.DirtyDrum(),
                            components.DirtyRollers(),
                        ]
                    ),
                ),
                components.Switch(
                    components.Selector(
                        [
                            components.LightingGradient(),
                            components.Brightness(),
                            components.Gamma(),
                        ]
                    ),
                ),
                components.Switch(
                    components.Selector(
                        [
                            components.SubtleNoise(),
                            components.Jpeg()
                        ]
                    ),
                ),
                components.Switch(
                    components.Selector(
                        [
                            components.Markup(),
                            components.Scribbles()
                        ]
                    ),
                ),
                components.Switch(
                    components.Selector(
                        [
                            components.BadPhotoCopy(),
                            components.ShadowCast(),
                            components.LowLightNoise()
                        ]
                    ),
                ),
                components.Switch(
                    components.Selector(
                        [
                            components.NoisyLines(),
                            components.BindingsAndFasteners()
                        ]
                    ),
                ),
                components.Switch(
                    components.Selector(
                        [
                            components.Squish(),
                            components.Geometric()
                        ]
                    ),
                ),
                components.Switch(
                    components.Selector(
                        [
                            components.DotMatrix(),
                            components.Faxify()
                        ]
                    ),
                ),
                components.Switch(
                    components.Selector(
                        [
                            components.InkMottling(),
                            components.ReflectedLight()
                        ]
                    ),
                ),
            ],
            **config.get("post_effect", {}),
        )
    
    def apply(self, layers: List) -> None:
        """
        레이어에 후처리 효과 적용
        
        Args:
            layers: 효과를 적용할 레이어 목록
        """
        self.effects.apply(layers)
        
    def get_component(self) -> components.Iterator:
        """
        후처리 효과 컴포넌트 반환
        
        Returns:
            components.Iterator: 후처리 효과 컴포넌트
        """
        return self.effects