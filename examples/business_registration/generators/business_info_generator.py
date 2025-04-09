"""
사업자 정보 생성 모듈
사업자 등록증에 들어갈 랜덤 정보를 생성합니다.
"""
import random
import datetime
from typing import Dict, List, Tuple, Any

import pandas as pd

from config import (
    DEFAULT_BIZ_INFO_PATH, 
    DEFAULT_TAX_OFFICE_LIST_PATH,
    BUSINESS_TYPES,
    ISSUE_REASONS,
    COMPANY_PREFIXES
)

class BusinessInfoGenerator:
    """사업자 정보 생성 클래스"""
    
    def __init__(self, dataset_path: str = DEFAULT_BIZ_INFO_PATH, tax_office_path: str = DEFAULT_TAX_OFFICE_LIST_PATH):
        """
        사업자 정보 생성기 초기화
        
        Args:
            dataset_path: 사업자 정보 데이터셋 경로
            tax_office_path: 세무서 목록 파일 경로
        """
        self.dataset_path = dataset_path
        self.tax_office_path = tax_office_path
        self.information, self.tax_office = self._read_information()

    def generate_random_info(self) -> Dict[str, Any]:
        """
        무작위 사업자 정보 생성
        
        Returns:
            Dict[str, Any]: 생성된 사업자 정보
        """
        biz_registration_data = self.information.sample(n=1)

        # 법인등록번호
        corporate_registration_number = self._generate_business_number()

        # 상호
        biz_name = self._generate_biz_name(biz_registration_data['상호'].values[0])

        # 개업일자
        open_date = self._generate_open_date(biz_registration_data['신고일자'].values[0])

        # 업태, 종목
        biz_types, biz_items = self._generate_business_type(random.randint(1, 5))

        # 발급사유
        issue_reason = self._generate_issue_reason()

        # 과세여부
        is_taxable = "여(   ) 부(∨)" if random.random() < 0.8 else "여(∨) 부(   )"

        # 발급일
        issue_date = self._generate_issue_date()

        # 세무서명
        issuer_name = self._generate_issuer_name()

        return {
            "TITLE": "사 업 자 등 록 증",
            "사업자종류": "( 법인사업자 )" if biz_registration_data['법인여부'].values[0] == "법인" else "( 일반과세자 )",
            "등록번호": biz_registration_data['사업자등록번호'].values[0],
            "상호": biz_name,
            "대표자": biz_registration_data['대표자명'].values[0],
            "개업일": open_date,
            "법인등록번호": corporate_registration_number,
            "사업장주소": biz_registration_data['사업장소재지(도로명)'].values[0],
            "본점주소": biz_registration_data['사업장소재지(도로명)'].values[0],
            "업태": biz_types,
            "종목": biz_items,
            "발급사유": issue_reason,
            "과세여부": is_taxable,
            "발급일": issue_date,
            "세무서": issuer_name
        }
    
    def _read_information(self) -> Tuple[pd.DataFrame, List[str]]:
        """
        사업자 정보와 세무서 목록 읽기
        
        Returns:
            Tuple[pd.DataFrame, List[str]]: 사업자 정보 데이터프레임과 세무서 목록
        """
        information = pd.read_csv(self.dataset_path, dtype={'column_name': str}, low_memory=False)
        with open(self.tax_office_path, 'r', encoding='utf-8') as f:
            tax_offices = [line.strip() for line in f.readlines()]
        return information, tax_offices

    def _generate_business_number(self) -> str:
        """
        법인등록번호 생성
        
        Returns:
            str: 형식에 맞는 법인등록번호
        """
        return f"{random.randint(100000, 999999)}-{random.randint(1000000, 9999999)}"
    
    def _generate_biz_name(self, biz_name: str) -> str:
        """
        상호 생성/변환
        
        Args:
            biz_name: 기본 상호명
            
        Returns:
            str: 처리된 상호명
        """
        if random.random() < 0.5:
            if "주식회사" not in biz_name and "(주)" not in biz_name:
                biz_name = f"{random.choice(COMPANY_PREFIXES)} {biz_name}"
        return biz_name

    def _generate_open_date(self, date: int) -> str:
        """
        개업일자 형식 변환
        
        Args:
            date: 신고일자 (YYYYMMDD 형식의 정수)
            
        Returns:
            str: 형식화된 개업일자 문자열
        """
        return f"{str(date)[:4]} 년 {str(date)[4:6]} 월 {str(date)[6:]} 일"
    
    def _generate_business_type(self, num: int) -> Tuple[List[str], List[str]]:
        """
        업태와 종목 생성
        
        Args:
            num: 생성할 업태/종목 수
            
        Returns:
            Tuple[List[str], List[str]]: 업태 목록과 종목 목록
        """
        main_types = BUSINESS_TYPES["main_types"]
        sub_types = BUSINESS_TYPES["sub_types"]
        
        return random.sample(main_types, min(num, len(main_types))), random.sample(sub_types, min(num, len(sub_types)))
    
    def _generate_issue_reason(self) -> str:
        """
        발급사유 생성
        
        Returns:
            str: 발급사유 (확률적으로 빈 문자열일 수 있음)
        """
        if random.random() < 0.3:
            return random.choice(ISSUE_REASONS)
        else:
            return ""
    
    def _generate_issue_date(self, start_year: int = 2022, end_year: int = 2025) -> str:
        """
        발급일자 생성
        
        Args:
            start_year: 시작 연도
            end_year: 종료 연도
            
        Returns:
            str: 형식화된 발급일자 문자열
        """
        start_date = datetime.date(start_year, 1, 1)
        end_date = datetime.date(end_year, 12, 31)
        delta = end_date - start_date
        random_days = random.randrange(delta.days)
        date = start_date + datetime.timedelta(days=random_days)
        return date.strftime("%Y 년 %m 월 %d 일")
    
    def _generate_issuer_name(self) -> str:
        """
        세무서장 이름 생성
        
        Returns:
            str: 세무서장 이름
        """
        issuer_name = f"{random.choice(self.tax_office)}장"
        return ' '.join(issuer_name)