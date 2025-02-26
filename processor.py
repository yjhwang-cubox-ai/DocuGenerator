import random
import string
import datetime
from PIL import Image, ImageDraw, ImageFont
import os
import glob
import pandas as pd

class BusinessRegistrationGenerator:
    def __init__(self, 
                 template_path=None,
                 dataset_path: str = 'dataset/total.csv',
                 font_path="fonts/gulim.ttc",
                 width=210,
                 height=297,
                 background_color=(255, 255, 255),
                 output_dir="output"):
        """
        생성에 필요한 기본 설정을 초기화.
        - template_path: 배경 템플릿 이미지 경로 (None이면 단색 배경 사용)
        - font_path: 한글 폰트 파일 경로
        - width, height: 이미지 사이즈 (템플릿을 사용하지 않을 때만 유효)
        - background_color: 배경 색상 (템플릿 미사용 시)
        - output_dir: 생성된 이미지가 저장될 폴더
        """
        self.template_path = template_path
        self.dataset_path = dataset_path
        self.font_path = font_path
        self.width = width
        self.height = height
        self.background_color = background_color
        self.output_dir = output_dir
        
        # 폴더가 없으면 생성
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 폰트 미리 로드
        self.title_font = ImageFont.truetype(self.font_path, 40)
        self.normal_font = ImageFont.truetype(self.font_path, 28)
        self.small_font = ImageFont.truetype(self.font_path, 24)

        self.information, self.tax_office = self._read_information()

    def _read_information(self):
        information = pd.read_csv(self.dataset_path)
        with open('dataset/tax_office_list.txt', 'r', encoding='utf-8') as f:
            tax_offices = [line.strip() for line in f.readlines()]
        return information, tax_offices
    
    def generate_business_number():
        return f"{random.randint(100000, 999999)}-{random.randint(1000000, 9999999)}"

    def generate_random_date(self, start_year=2024, end_year=2025):
        start_date = datetime.date(start_year, 1, 1)
        end_date = datetime.date(end_year, 12, 31)
        delta = end_date - start_date
        random_days = random.randrange(delta.days)
        date = start_date + datetime.timedelta(days=random_days)
        return date.strftime("%Y년 %m월 %d일")

    def generate_random_text(self):
        items = ["도소매", "서비스", "제조업", "정보통신업", "부동산업", "컨설팅"]
        return random.choice(items)

    def create_single_image(self, index=0):
        """단일 사업자등록증 이미지를 생성하여 파일로 저장"""
        # 1) 템플릿 또는 단색 배경 이미지 준비
        if self.template_path:
            image = Image.open(self.template_path).convert("RGB")
        else:
            image = Image.new("RGB", (self.width, self.height), self.background_color)

        draw = ImageDraw.Draw(image)

        # 2) 데이터 생성
        data = self.information.sample(n=1)
        information = {
            "등록번호": data['사업자등록번호'].values[0],
            "상호": data['상호'].values[0],
            "대표자명": data['대표자명'].values[0],
            "사업장 소재지": data['사업장소재지(도로명)'].values[0],
            "개업일자": data['신고일자'].values[0],
            # "업태/종목": self.generate_random_text(),
            # "발급일자": self.generate_random_date(),
            # "세무서명": "북부세무서장"
        }

        # 1) 랜덤으로 상호명에 주식회사 붙이기
        if random.random() < 0.3:
            information['상호'] = f"주식회사 {information['상호']}"

        # 2. 개업일자 데이터 형태 수정
        information['개업일자'] = f"{str(information['개업일자'])[:4]} 년 {str(information['개업일자'])[4:6]} 월 {str(information['개업일자'])[6:]} 일"
        # 3. 업태
        
        # 4. 종목
        # 5. 세무서명 래덤 선택

        # 3) 텍스트 배치
        # 실제 사업자등록증 레이아웃에 맞게 좌표 조절
        draw.text((self.width/2 - 150, 50), "사업자등록증", font=self.title_font, fill=(0, 0, 0))

        y_offset = 150
        line_spacing = 60

        draw.text((100, y_offset), f"등록번호: {data['등록번호']}", font=self.normal_font, fill=(0, 0, 0))
        y_offset += line_spacing
        draw.text((100, y_offset), f"상호: {data['상호']}", font=self.normal_font, fill=(0, 0, 0))
        y_offset += line_spacing
        draw.text((100, y_offset), f"대표자명: {data['대표자명']}", font=self.normal_font, fill=(0, 0, 0))
        y_offset += line_spacing
        draw.text((100, y_offset), f"사업장 소재지: {data['사업장 소재지']}", font=self.normal_font, fill=(0, 0, 0))
        y_offset += line_spacing
        draw.text((100, y_offset), f"개업일자: {data['개업일자']}", font=self.normal_font, fill=(0, 0, 0))
        y_offset += line_spacing
        draw.text((100, y_offset), f"업태/종목: {data['업태/종목']}", font=self.normal_font, fill=(0, 0, 0))
        y_offset += line_spacing

        # 하단 발급 정보
        draw.text((100, y_offset + 100), f"발급일자: {data['발급일자']}", font=self.small_font, fill=(0, 0, 0))
        draw.text((100, y_offset + 140), f"{data['세무서명']}", font=self.small_font, fill=(0, 0, 0))

        # 4) 저장
        filename = os.path.join(self.output_dir, f"business_reg_{index:05d}.png")
        image.save(filename)

    def create_bulk_images(self, n=100000):
        """n장의 이미지를 순차적으로 생성"""
        for i in range(n):
            self.create_single_image(index=i)
            if i % 1000 == 0 and i != 0:
                print(f"{i}장 생성 완료")

if __name__ == "__main__":
    # 예시 사용
    generator = BusinessRegistrationGenerator(
        template_path=None,      # 템플릿 이미지 경로가 있다면 여기 넣기
        dataset_path='dataset/total.csv',
        font_path="fonts/gulim.ttc",
        width=1000,
        height=1400,
        background_color=(255, 255, 255),
        output_dir="output_business_reg"
    )
    generator.create_bulk_images(n=10000)  # 예: 1만 장 생성
