"""
사업자 등록증 이미지 생성기 설정 파일
기본 설정값과 경로 정보 등을 정의합니다.
"""
from typing import Dict, List, Any, Tuple

# 문서 크기 설정
DEFAULT_WIDTH = 1478  # 사업자 등록증 너비
DEFAULT_HEIGHT = 2074  # 사업자 등록증 높이

# 이미지 품질 설정
DEFAULT_QUALITY = [80, 95]  # JPEG 압축 품질 범위
DEFAULT_APPLY_TEXTURE_PROB = 0.3  # 텍스처 적용 확률

# 데이터셋 분할 설정
DEFAULT_SPLIT_RATIO = [0.8, 0.1, 0.1]  # 학습/검증/테스트 비율
DEFAULT_SPLITS = ["train", "validation", "test"]

# 폰트 경로 설정
DEFAULT_FONT_PATH = "fonts/gulim.ttc"
DEFAULT_BOLD_FONT_PATH = "fonts/gulim.ttc"

# 데이터 경로 설정
DEFAULT_BIZ_INFO_PATH = "resources/business_registration/biz_info.csv"
DEFAULT_TAX_OFFICE_LIST_PATH = "dataset/tax_office_list.txt"
DEFAULT_TEMPLATE_PATHS = []  # 사업자 등록증 배경 이미지 경로

# 레이아웃: 사업자 등록증 필드 정의 (필드명, 좌표(x, y), 폰트 크기, 볼드 여부)
def get_field_positions(width: int, height: int) -> Dict[str, Dict[str, Any]]:
    """문서 크기에 따른 필드 위치 정보 반환"""
    return {
        "TITLE": {"position": (width//2 - 280, 280), "font_size": 70, "bold": True},
        "사업자종류": {"position": (width//2 - 150, 360), "font_size": 45, "bold": False},
        "등록번호": {"position": (730, 430), "font_size": 45, "bold": False},
        "상호": {"position": (420, 510), "font_size": 32, "bold": False},
        "대표자": {"position": (420, 560), "font_size": 32, "bold": False},
        "개업일": {"position": (420, 640), "font_size": 32, "bold": False},
        "법인등록번호": {"position": (1020, 640), "font_size": 32, "bold": False},
        "사업장주소": {"position": (420, 690), "font_size": 32, "bold": False},
        "본점주소": {"position": (420, 770), "font_size": 32, "bold": False},
        "업태": {"position": (500, 850), "font_size": 28, "bold": False},
        "종목": {"position": (950, 850), "font_size": 28, "bold": False},
        "발급사유": {"position": (420, 1100), "font_size": 32, "bold": False},
        "과세여부": {"position": (700, 1400), "font_size": 32, "bold": False},
        "발급일": {"position": (width//2 - 190, 1680), "font_size": 40, "bold": False},
        "세무서": {"position": (width//2 - 230, 1760), "font_size": 60, "bold": True},
    }

# 레이블 키 필드 정의 (필드명, 좌표(x, y), 텍스트, 폰트 크기, 볼드 여부)
def get_key_positions(width: int, height: int) -> Dict[str, Dict[str, Any]]:
    """문서 크기에 따른 레이블 키 위치 정보 반환"""
    return {
        "등록번호": {"position": (500, 430), "text": "등록번호 : ", "font_size": 45, "bold": False},
        "상호": {"position": (150, 510), "text": "법인명(단체명)  :", "font_size": 32, "bold": False},
        "대표자": {"position": (150, 560), "text": "대      표      자 :", "font_size": 32, "bold": False},
        "개업일": {"position": (150, 640), "text": "개 업 연 월 일   :", "font_size": 32, "bold": False},
        "법인등록번호": {"position": (780, 640), "text": "법인등록번호  :", "font_size": 32, "bold": False},
        "사업장주소": {"position": (150, 690), "text": "사업장  소재지  :", "font_size": 32, "bold": False},
        "본점주소": {"position": (150, 770), "text": "본 점 소 재 지  :", "font_size": 32, "bold": False},
        "사업종류": {"position": (150, 850), "text": "사 업 의  종 류  :", "font_size": 32, "bold": False},
        "업태": {"position": (420, 850), "text": "업태", "font_size": 32, "bold": False},
        "종목": {"position": (860, 850), "text": "종목", "font_size": 32, "bold": False},
        "발급사유": {"position": (150, 1100), "text": "발  급  사  유  :", "font_size": 32, "bold": False},
        "과세여부": {"position": (150, 1400), "text": "사업자 단위 과세 적용사업자 여부  :", "font_size": 32, "bold": False},
        "전자우편주소": {"position": (150, 1450), "text": "자세금계산서 전용 전자우편주소 :", "font_size": 32, "bold": False},
    }

# 비즈니스 유형 데이터
BUSINESS_TYPES = {
    "main_types": [
        "제조", "건설", "도소매", "운수", "숙박", "정보통신", "금융", "부동산", 
        "과학기술 서비스", "교육 서비스", "전문, 과학 및 기술서비스업"
    ],
    "sub_types": [
        "금속 가공제품 제조업", "전자부품 제조업", "소프트웨어 개발", "유통업", 
        "도매 및 상품 중개업", "소매업", "일반 건설업", "전문직별 공사업", "전기 전자공학 연구개발업"
    ]
}

# 발급 사유 목록
ISSUE_REASONS = [
    '신규 개업 신고', '사업장 이전', '대표자 변경', '업종 변경', 
    '휴업 및 재개업', '분실 또는 훼손', '기타 변경 등록'
]

# 회사 접두사
COMPANY_PREFIXES = ["주식회사", "유한회사", "(주)"]