import streamlit as st
import os
import numpy as np
from PIL import Image
import tempfile
import importlib
import inspect
import sys
from synthdocs import components
from examples.businessRegistration.business_registration_generator import BusinessRegistration
from ui.docu_generator import BusinessNormalRegistration
from yaml import safe_load
import json
import random

# 타이틀 설정
st.set_page_config(page_title="문서 이미지 생성기", layout="wide")
st.title("문서 이미지 생성기")

# synthdocs/components/artifact 디렉토리에서 적용 가능한 효과 목록 가져오기
def get_available_effects():
    effects = []
    # synthdocs/components/artifact 디렉토리 내의 모든 Component 클래스 찾기
    artifact_dir = components.artifact.__path__[0]
    for filename in os.listdir(artifact_dir):
        if filename.endswith('.py') and not filename.startswith('__'):
            module_name = f"synthdocs.components.artifact.{filename[:-3]}"
            try:
                module = importlib.import_module(module_name)
                # 모듈에서 Component 클래스 찾기
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and issubclass(obj, components.Component) and obj != components.Component:
                        effects.append((name, obj))
            except Exception as e:
                st.error(f"모듈 {module_name} 로드 중 오류 발생: {e}")
    
    return effects

# 클래스 생성자 매개변수 분석 함수
def get_class_params(cls):
    """클래스 생성자의 매개변수를 분석하고 기본값을 반환"""
    try:
        signature = inspect.signature(cls.__init__)
        params = {}
        for name, param in signature.parameters.items():
            if name not in ['self', 'args', 'kwargs']:
                default = param.default
                if default is not inspect.Parameter.empty:
                    params[name] = default
        return params
    except Exception as e:
        st.warning(f"매개변수 분석 중 오류 발생: {e}")
        return {}

# 효과 매개변수 UI 생성 함수
def create_param_ui(params):
    """효과 매개변수 설정을 위한 UI 위젯 생성"""
    ui_params = {}
    for name, default in params.items():
        if isinstance(default, bool):
            ui_params[name] = st.checkbox(name, default)
        elif isinstance(default, int):
            ui_params[name] = st.slider(name, min_value=0, max_value=100, value=default)
        elif isinstance(default, float):
            ui_params[name] = st.slider(name, min_value=0.0, max_value=1.0, value=default, step=0.01)
        elif isinstance(default, (list, tuple)) and len(default) == 2:
            # 범위 설정으로 가정
            if all(isinstance(x, int) for x in default):
                min_val, max_val = default
                ui_params[name] = st.slider(name, min_value=min_val, max_value=max_val, value=default)
            elif all(isinstance(x, float) for x in default):
                min_val, max_val = default
                ui_params[name] = st.slider(name, min_value=min_val, max_value=max_val, value=default, step=0.01)
            else:
                ui_params[name] = st.text_input(name, str(default))
        else:
            ui_params[name] = default
    
    return ui_params

# 효과 적용 함수
def apply_effect(image, effect_class, params=None):
    """선택한 효과를 이미지에 적용하는 함수"""
    from synthdocs import layers
    
    # 매개변수 처리
    if params is None:
        params = {}
    
    # 문자열 매개변수를 적절한 형태로 변환
    processed_params = {}
    for key, value in params.items():
        if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
            # 리스트로 변환 시도
            try:
                processed_params[key] = json.loads(value)
            except:
                processed_params[key] = value
        else:
            processed_params[key] = value
    
    # PIL 이미지를 Layer로 변환
    if isinstance(image, Image.Image):
        image_array = np.array(image)
        layer = layers.Layer(image_array)
    else:
        layer = layers.Layer(image)
    
    # 효과 인스턴스 생성 및 적용
    try:
        effect_instance = effect_class(**processed_params)
        meta = effect_instance.apply([layer])
        
        # 결과 이미지 반환
        result = layer.output()
        if isinstance(result, np.ndarray):
            return Image.fromarray(result.astype(np.uint8))
        return result
    except Exception as e:
        st.error(f"효과 적용 중 오류 발생: {e}")
        return image

# 사이드바에 문서 종류 선택 옵션 추가
st.sidebar.header("설정")
document_type = st.sidebar.selectbox(
    "문서 종류 선택",
    ["사업자등록증"]
)

# 설정 파일 로드
config_file = "examples/businessRegistration/config.yaml"
with open(config_file, 'r', encoding='utf-8') as f:
    config = safe_load(f)

# 사용 가능한 효과 목록 가져오기
available_effects = get_available_effects()
effect_names = [name for name, _ in available_effects]
effect_dict = {name: cls for name, cls in available_effects}

# 메인 영역
if document_type == "사업자등록증":
    # 탭 설정
    tab1, tab2 = st.tabs(["문서 생성", "효과 적용"])
    
    with tab1:
        st.header("사업자등록증 생성")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            # 데이터 수 입력
            num_copies = st.number_input("생성할 이미지 수", min_value=1, max_value=10, value=1)
            
            # 랜덤 시드 설정
            random_seed = st.number_input("랜덤 시드", min_value=0, value=77)
            
            # 생성 버튼
            generate_button = st.button("사업자등록증 생성")
        
        # 생성 결과 표시 영역
        with col2:
            if generate_button:
                with st.spinner("사업자등록증 생성 중..."):
                    # 랜덤 시드 설정
                    random.seed(random_seed)
                    np.random.seed(random_seed)
                    
                    # 여러 이미지 생성 시 리스트 초기화
                    st.session_state["generated_images"] = []
                    st.session_state["business_infos"] = []
                    
                    for i in range(num_copies):
                        # 생성기 초기화
                        generator = BusinessNormalRegistration(config)
                        
                        # 이미지 생성
                        data = generator.generate()
                        
                        # 생성된 이미지와 정보 저장
                        if "image" in data:
                            image = Image.fromarray(data["image"].astype(np.uint8))
                            st.session_state["generated_images"].append(image)
                            st.session_state["business_infos"].append(data["business_info"])
                
                # 첫 번째 이미지 기본으로 선택
                if "generated_images" in st.session_state and len(st.session_state["generated_images"]) > 0:
                    st.session_state["current_image_index"] = 0
                    st.session_state["original_image"] = st.session_state["generated_images"][0]
            
            # 생성된 이미지가 있을 경우 표시
            if "generated_images" in st.session_state and len(st.session_state["generated_images"]) > 0:
                # 여러 이미지가 있을 경우 선택기 표시
                if len(st.session_state["generated_images"]) > 1:
                    selected_idx = st.selectbox(
                        "생성된 이미지 선택", 
                        range(len(st.session_state["generated_images"])),
                        index=st.session_state.get("current_image_index", 0),
                        format_func=lambda i: f"이미지 #{i+1}"
                    )
                    
                    # 선택된 이미지 인덱스 저장
                    st.session_state["current_image_index"] = selected_idx
                    st.session_state["original_image"] = st.session_state["generated_images"][selected_idx]
                
                # 현재 선택된 이미지 표시
                st.image(
                    st.session_state["original_image"], 
                    caption=f"생성된 사업자등록증", 
                    width = 700
                )

                with st.expander("사업자 정보 보기"):
                    current_idx = st.session_state.get("current_image_index", 0)
                    st.json(st.session_state["business_infos"][current_idx])                                
                
                # 이미지 저장 버튼
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                st.session_state["original_image"].save(temp_file.name)
                with open(temp_file.name, "rb") as file:
                    btn = st.download_button(
                        label="이미지 다운로드",
                        data=file,
                        file_name="business_registration.png",
                        mime="image/jpeg"
                    )
    
    with tab2:
        st.header("효과 적용")
        
        if "original_image" not in st.session_state:
            st.warning("먼저 '문서 생성' 탭에서 사업자등록증을 생성해주세요.")
        else:
            col1, col2 = st.columns([1, 4])
            
            with col1:
                # 효과 선택
                selected_effect = st.selectbox("적용할 효과 선택", effect_names)
                
                # 선택한 효과의 클래스 가져오기
                effect_class = effect_dict[selected_effect]
                
                # 클래스 생성자 매개변수 분석
                default_params = get_class_params(effect_class)
                
                # 매개변수 설정 UI 생성
                st.subheader("효과 설정")
                # effect_params = create_param_ui(default_params)
                
                # 효과 적용 버튼
                apply_effect_button = st.button("효과 적용")
            
            with col2:
                # 원본과 처리된 이미지 비교 표시
                if apply_effect_button:
                    with st.spinner("효과 적용 중..."):
                        # 효과 적용
                        processed_image = apply_effect(st.session_state["original_image"], effect_class, default_params)
                        
                        # 결과 저장
                        st.session_state["processed_image"] = processed_image
                
                # 이미지 표시 영역
                img_col1, img_col2 = st.columns(2)
                
                with img_col1:
                    st.subheader("원본 이미지")
                    st.image(st.session_state["original_image"], use_container_width=True)
                
                with img_col2:
                    st.subheader("처리된 이미지")
                    if "processed_image" in st.session_state:
                        st.image(st.session_state["processed_image"], use_container_width=True)
                        
                        # 처리된 이미지 저장 버튼
                        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                        st.session_state["processed_image"].save(temp_file.name)
                        with open(temp_file.name, "rb") as file:
                            btn = st.download_button(
                                label="처리된 이미지 다운로드",
                                data=file,
                                file_name="processed_business_registration.png",
                                mime="image/jpeg"
                            )
                    else:
                        st.info("효과를 적용하면 여기에 결과가 표시됩니다.")

else:
    st.warning("현재 사업자등록증만 지원합니다.")

# 푸터
st.sidebar.markdown("---")
st.sidebar.info("본 어플리케이션은 문서 이미지 생성 및 변형을 위한 도구입니다.")