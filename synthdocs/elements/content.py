"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
from collections import OrderedDict

import random
import datetime
import pandas as pd
import numpy as np
from synthdocs import components

from synthdocs.elements.textbox import TextBox
from synthdocs.layouts import GridStack
from synthdocs import layers


class TextReader:
    def __init__(self, path, cache_size=2 ** 28, block_size=2 ** 20):
        self.fp = open(path, "r", encoding="utf-8")
        self.length = 0
        self.offsets = [0]
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self.block_size = block_size
        self.bucket_size = cache_size // block_size
        self.idx = 0

        while True:
            text = self.fp.read(self.block_size)
            if not text:
                break
            self.length += len(text)
            self.offsets.append(self.fp.tell())

    def __len__(self):
        return self.length

    def __iter__(self):
        return self

    def __next__(self):
        char = self.get()
        self.next()
        return char

    def move(self, idx):
        self.idx = idx

    def next(self):
        self.idx = (self.idx + 1) % self.length

    def prev(self):
        self.idx = (self.idx - 1) % self.length

    def get(self):
        key = self.idx // self.block_size

        if key in self.cache:
            text = self.cache[key]
        else:
            if len(self.cache) >= self.bucket_size:
                self.cache.popitem(last=False)

            offset = self.offsets[key]
            self.fp.seek(offset, 0)
            text = self.fp.read(self.block_size)
            self.cache[key] = text

        self.cache.move_to_end(key)
        char = text[self.idx % self.block_size]
        return char


class Content:
    def __init__(self, config):
        self.margin = config.get("margin", [0, 0.1])
        self.reader = TextReader(**config.get("text", {}))
        self.font = components.BaseFont(**config.get("font", {}))
        self.layout = GridStack(config.get("layout", {}))
        self.textbox = TextBox(config.get("textbox", {}))
        self.textbox_color = components.Switch(components.Gray(), **config.get("textbox_color", {}))
        self.content_color = components.Switch(components.Gray(), **config.get("content_color", {}))

    def generate(self, size):
        width, height = size

        layout_left = width * np.random.uniform(self.margin[0], self.margin[1])
        layout_top = height * np.random.uniform(self.margin[0], self.margin[1])
        layout_width = max(width - layout_left * 2, 0)
        layout_height = max(height - layout_top * 2, 0)
        layout_bbox = [layout_left, layout_top, layout_width, layout_height]

        text_layers, texts = [], []
        layouts = self.layout.generate(layout_bbox)
        self.reader.move(np.random.randint(len(self.reader)))

        for layout in layouts:
            font = self.font.sample()

            for bbox, align in layout:
                x, y, w, h = bbox
                text_layer, text = self.textbox.generate((w, h), self.reader, font)
                self.reader.prev()

                if text_layer is None:
                    continue

                text_layer.center = (x + w / 2, y + h / 2)
                if align == "left":
                    text_layer.left = x
                if align == "right":
                    text_layer.right = x + w

                self.textbox_color.apply([text_layer])
                text_layers.append(text_layer)
                texts.append(text)

        self.content_color.apply(text_layers)

        return text_layers, texts
    
class BusinessContent:
    def __init__(self, config):
        # self.margin = config.get("margin", [0, 0.1])
        # self.reader = TextReader(**config.get("text", {}))
        self.font = components.BaseFont(**config.get("font", {}))
        self.textbox = TextBox(config.get("textbox", {}))
        # self.textbox_color = components.Switch(components.Gray(), **config.get("textbox_color", {}))
        # self.content_color = components.Switch(components.Gray(), **config.get("content_color", {}))

        self.information, self.tax_office = self._read_information()

    def generate(self, size):
        width, height = size
        information = self._create_business_registration_info()

        text_layers, texts = [], []
        font = self.font.sample()

        # 레이아웃을 고정하고 텍스트만 설정
        for field_name, field_info in information.items():
            # bbox 좌표 필요
            x, y, w, h = field_info["position"]["x"], field_info["position"]["y"], field_info["position"]["width"], field_info["position"]["height"]
            # text 필요
            text = field_info["text"]
            if text == "":
                continue
            # 폰트필요            
            text_layer, text = self.textbox.generate((w, h), text, font)
            
            text_layers.append(text_layer)
            texts.append(text)

        return text_layers, texts
    
    def _read_information(self):
        information = pd.read_csv("resources/business_registration/total.csv")
        with open('resources/business_registration/tax_office_list.txt', 'r', encoding='utf-8') as f:
            tax_offices = [line.strip() for line in f.readlines()]
        return information, tax_offices

    def _create_business_registration_info(self):
        """Generate default business document information"""

        data = self.information.sample(n=1)

        information = {
            "HEAD": {
                "text": "사 업 자 등 록 증",
                "position": {"x": 459, "y": 280, "width": 537, "height": 58},  # 상대적 위치 (0~1 사이 값)
                "font_size": 70,
                "font_weight": "bold"
            },            
            "사업자종류": {
                "text": "( 법인사업자 )" if data['법인여부'].values[0] == "법인" else "( 일반과세자 )",
                "position": {"x": 589, "y": 360, "width": 289, "height": 42},
                "font_size": 45,
                "font_weight": "normal"
            },
            "등록번호": {
                "text": data['사업자등록번호'].values[0],
                "position": {"x": 730, "y": 430, "width": 315, "height": 33},
                "font_size": 12,
                "font_weight": "normal"
            },
            # "상호": {
            #     "text": data['상호'].values[0],
            #     "position": {"x": 459, "y": 280, "width": 537, "height": 58},                "font_size": 14,
            #     "font_weight": "normal"
            # },
            # "대표자명": {
            #     "text": data['대표자명'].values[0],
            #     "position": {"x": 459, "y": 280, "width": 537, "height": 58},
            #     "font_size": 14,
            #     "font_weight": "normal"
            # },
            # "개업연월일": {
            #     "text": data['신고일자'].values[0],
            #     "position": {"x": 459, "y": 280, "width": 537, "height": 58},                "font_size": 12,
            #     "font_weight": "normal"
            # },
            # "법인등록번호": {
            #     "text": f"{random.randint(100000, 999999)}-{random.randint(1000000, 9999999)}",
            #     "position": {"x": 459, "y": 280, "width": 537, "height": 58},
            #     "font_size": 12,
            #     "font_weight": "normal"
            # },
            # "사업장 소재지": {
            #     "text": data['사업장소재지(도로명)'].values[0][:30],
            #     "position": {"x": 459, "y": 280, "width": 537, "height": 58},                "font_size": 12,
            #     "font_weight": "normal"
            # },
            # "본점 소재지": {
            #     "text": data['사업장소재지(도로명)'].values[0][:30],
            #     "position": {"x": 459, "y": 280, "width": 537, "height": 58},
            #     "font_size": 12,
            #     "font_weight": "normal"
            # },
            # "업태": {
            #     "text": self._gen_biz_main_type(random.randint(1, 5)),
            #     "position": {"x": 459, "y": 280, "width": 537, "height": 58},            #     "font_size": 12,
            #     "font_weight": "normal"
            # },
            # "종목": {
            #     "text": self._gen_biz_sub_type(random.randint(1, 5)),
            #     "position": {"x": 459, "y": 280, "width": 537, "height": 58},
            #     "font_size": 12,
            #     "font_weight": "normal"
            # },
            # "발급사유": {
            #     "text": self._set_issuance_reason(),
            #     "position": {"x": 459, "y": 280, "width": 537, "height": 58},                "font_size": 12,
            #     "font_weight": "normal"
            # },
            # "발급일자": {
            #     "text": self._generate_issuance_date(start_year=2020),
            #     "position": {"x": 459, "y": 280, "width": 537, "height": 58},                "font_size": 12,
            #     "font_weight": "normal"
            # },
            # "세무서명": {
            #     "text": f"{random.choice(self.tax_office)}장",
            #     "position": {"x": 459, "y": 280, "width": 537, "height": 58},
            #     "font_size": 14,
            #     "font_weight": "bold"
            # }
        }

        return information
    
    def _gen_biz_main_type(self, num):
        main_types = ["제조", "건설", "도소매", "운수", "숙박", "정보통신", "금융", "부동산", "과학기술 서비스", "교육 서비스", "전문, 과학 및 기술서비스업"]
    
        return random.sample(main_types, num)
    
    def _gen_biz_sub_type(self, num):
        sub_types = ["금속 가공제품 제조업", "전자부품 제조업", "소프트웨어 개발", "유통업", "도매 및 상품 중개업", "소매업", "일반 건설업", "전문직별 공사업", "전기 전자공학 연구개발업"]
    
        return random.sample(sub_types, num)
    
    def _set_issuance_reason(self):
        return random.choice(['신규 개업 신고', '사업장 이전', '대표자 변경', '업종 변경', '휴업 및 재개업', '분실 또는 훼손', '기타 변경 등록']) if random.random() < 0.3 else ""
    
    def _generate_issuance_date(self, start_year=2020):
        start_date = datetime.date(start_year, 1, 1)
        end_date = datetime.date.today()  # 오늘 날짜로 설정
        delta = end_date - start_date
        random_days = random.randrange(delta.days)
        date = start_date + datetime.timedelta(days=random_days)

        return date.strftime("%Y 년 %m 월 %d 일")
