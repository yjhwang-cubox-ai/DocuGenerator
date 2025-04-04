"""
사업자 등록증 이미지 생성기 - 항목별 다른 폰트 크기 적용
"""
import json
import os
import re
import random
import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from synthdocs import elements, components, layers, templates, utils
from synthdocs.elements import Background, Document, Paper
from PIL import Image, ImageDraw, ImageFont
from blend_modes import normal, darken_only, lighten_only, multiply

# 문서 사이즈 고정 -> 데이터 수정후 다양하게 변형하는 걸로

class BusinessRegistration(templates.Template):
    def __init__(self, config=None, split_ratio: List[float] = [0.8, 0.1, 0.1]):
        super().__init__(config)
        if config is None:
            config = {}

        self.quality = config.get("quality", [80, 95])
        # 사업자 등록증은 가로 방향으로 고정
        self.landscape = 1.0
        # 사업자 등록증 크기 설정
        self.width = config.get("width", 1478)
        self.height = config.get("height", 2074)
        self.background = Background(config.get("background", {}))
        # self.document = Document(config.get("document", {}))
        self.paper = Paper(config.get("document", {}).get("paper", {}))
        self.effect = components.Iterator(
            [
                components.Switch(components.RGB()),
                components.Switch(components.Shadow()),
                components.Switch(components.Contrast()),
                components.Switch(components.RGBShiftBrightness()),
                components.Switch(components.MotionBlur()),
                components.Switch(components.GaussianBlur()),
            ],
            **config.get("effect", {}),
        )

        self.custom_effect = components.Iterator(
            [
                components.Switch(components.LowInkRandomLines()),
            ],
            **config.get("custom_effect", {}),
        )

        self.paper_effect = components.Iterator(
            [
                components.Switch(components.BrightnessTexturize()),
            ],
            **config.get("paper_effect", {}),
        )

        self.post_effect = components.Iterator(
            [
                components.Switch(components.ReflectedLight()),
            ],
            **config.get("post_effect", {}),
        )

        # 사업자 등록증 배경이미지
        self.registration_template_path = config.get("business_registration", {}).get("template_image", {}).get("paths", [])
        # 글꼴 설정
        self.font_path = config.get("font_path", "fonts/gulim.ttc")
        self.bold_font_path = config.get("bold_font_path", "fonts/gulim.ttc")
        
        # 사업자 등록증 필드 정의 (필드명, 좌표(x, y), 폰트 크기, 볼드 여부)
        self.fields = {
            "TITLE": {"position": (self.width//2 - 280, 280), "font_size": 70, "bold": True},
            "사업자종류": {"position": (self.width//2 - 150, 360), "font_size": 45, "bold": False},
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
            "발급일": {"position": (self.width//2 - 190, 1680), "font_size": 40, "bold": False},
            "세무서": {"position": (self.width//2 - 230, 1760), "font_size": 60, "bold": True},
        }

        self.keys = {
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

        # config for splits
        self.splits = ["train", "validation", "test"]
        self.split_ratio = split_ratio
        self.split_indexes = np.random.choice(3, size=10000, p=split_ratio)
        
        # 랜덤 사업자 정보 생성 기능
        self.business_info_generator = BusinessInfoGenerator(dataset_path=config.get("business_registration", {}).get("business_info", {}).get("path", []))

    def generate(self, business_info=None):
        # 사업자 등록증 크기 설정
        size = (self.width, self.height)

        # 배경 레이어 생성 (흰색 종이)
        # bg_layer = layers.RectLayer(size, (255, 255, 255, 255))
        bg_layer = self.paper.generate(size)
        
        # 사업자 등록증 양식 이미지 생성
        template_layer = self.create_registration_template(size)

        # Blend images - texture
        # opacity = 0.9
        # blended_img = multiply(template_layer.image, bg_layer.image, opacity)
        # template_layer.image = blended_img
        # Blend images - 종이 질감 + 사업자등록증 배경
        opacity = 0.3
        blended_img = normal(template_layer.image, bg_layer.image, opacity)
        template_layer.image = blended_img

        # Paper Effect 적용
        # self.paper_effect.apply([template_layer])
        
        # 무작위 사업자 정보 생성 또는 전달받은 정보 사용
        if business_info is None:
            business_info = self.business_info_generator.generate_random_info()
        
        # 텍스트 레이어 생성
        text_layers = []
        texts = []
        combined_info = {}  # key와 value를 모두 저장할 딕셔너리

        # 1. 키 레이어 생성
        for field, config in self.keys.items():
            key_layer = self.create_text_layer(
                f"key_{field}",
                config["text"],
                config["position"],
                config["font_size"],
                config["bold"]
            )
            text_layers.append(key_layer)
            texts.append({"text": re.sub(r'\s{2,}', ' ', config["text"]), "position": config["position"]})
            
            # combined_info에 키 추가
            combined_info[f"key_{field}"] = field

        # 2. 값 레이어 생성
        for field, value in business_info.items():            
            if field in self.fields:
                if isinstance(value, list):
                    base_position = self.fields[field]["position"]
                    y_offset_step = 50

                    for i, item in enumerate(value):
                        position = (base_position[0], base_position[1] + i * y_offset_step)
                        text_layer = self.create_text_layer(
                            f"{field}_{i}", 
                            item, 
                            position, 
                            self.fields[field]["font_size"], 
                            self.fields[field]["bold"])
                        text_layers.append(text_layer)
                        # texts.append(item)
                        texts.append({"text": re.sub(r'\s{2,}', ' ', item), "position": position})

                        # combined_info에 값 추가
                        combined_info[f"value_{field}_{i}"] = item
                else:
                    # 주소 필드인 경우 줄바꿈 처리
                    if field in ["사업장주소", "본점주소"]:
                        font_size = self.fields[field]["font_size"]
                        font_path = self.bold_font_path if self.fields[field]["bold"] else self.font_path
                        font = ImageFont.truetype(font_path, font_size)
                        
                        # 주소 텍스트를 여러 줄로 분할 (종이 가로 크기의 약 60%를 최대 너비로 설정)
                        max_width = int(self.width * 0.6)
                        text_lines = self.split_text_to_fit_width(str(value), font, max_width)
                        
                        # 각 줄을 별도의 텍스트 레이어로 생성
                        base_position = self.fields[field]["position"]
                        y_offset_step = 40  # 줄 간격
                        
                        for i, line in enumerate(text_lines):
                            position = (base_position[0], base_position[1] + i * y_offset_step)
                            text_layer = self.create_text_layer(
                                f"{field}_line{i}", 
                                line, 
                                position,
                                font_size,
                                self.fields[field]["bold"]
                            )
                            text_layers.append(text_layer)
                            # texts.append(line)
                            texts.append({"text": re.sub(r'\s{2,}', ' ', line), "position": position})
                            
                            # combined_info에 값 추가
                            combined_info[f"value_{field}_line{i}"] = line
                    else:
                        # 일반 필드는 그대로 처리
                        text_layer = self.create_text_layer(
                            field, 
                            str(value), 
                            self.fields[field]["position"],
                            self.fields[field]["font_size"],
                            self.fields[field]["bold"]
                        )
                        text_layers.append(text_layer)
                        # texts.append(value)
                        texts.append({"text": re.sub(r'\s{2,}', ' ', value), "position": self.fields[field]["position"]})

                        # combined_info에 값 추가
                        combined_info[f"value_{field}"] = value
        
        # 모든 레이어 결합
        # document_group = layers.Group([*text_layers, template_layer, bg_layer])
        document_group = layers.Group([*text_layers, template_layer])
        layer = document_group.merge()
        
        # 효과 적용
        # self.effect.apply([layer])
        # self.custom_effect.apply([layer])
        # self.paper_effect.apply([layer])
        self.post_effect.apply([layer])
        # 최종 이미지 출력
        image = layer.output(bbox=[0, 0, *size])
        
        # 레이블 생성 (모든 텍스트를 공백으로 연결)
        # 텍스트의 순서를 left top -> right bottom 순으로 정렬        
        new_ordered_texts = []
        sorted_texts = sorted(texts, key=lambda item: (item['position'][1], item['position'][0]))
        # 정렬된 결과 출력
        for item in sorted_texts:
            new_ordered_texts.append(item['text'])

        label = " ".join(new_ordered_texts)
        label = label.strip()
        label = re.sub(r"\s+", " ", label)
        
        quality = np.random.randint(self.quality[0], self.quality[1] + 1)
        
        # ROI (관심 영역) - 문서 전체 영역 지정
        roi = np.array([[0, 0], [size[0], 0], [size[0], size[1]], [0, size[1]]], dtype=int)
        
        data = {
            "image": image,
            "label": label,
            "quality": quality,
            "roi": roi,
            "business_info": business_info,
            "combined_info": combined_info
        }
        
        return data
    
    def create_registration_template(self, size):
        temlate_img_paths = utils.search_files(self.registration_template_path, exts=[".jpg", ".jpeg", ".png", ".bmp"])
        selected_temlate_path = np.random.choice(temlate_img_paths)
        template_img = Image.open(selected_temlate_path).convert("RGBA").resize(size, Image.LANCZOS)

        # 이미지를 다시 레이어로 변환
        template = layers.RectLayer(size, (0,0,0,0))
        template.image = np.array(template_img).astype(np.float32)

        return template
    
    def create_text_layer(self, field, text, position, font_size, bold=False):
        """특정 위치에 텍스트 레이어 생성 (폰트 크기와 볼드 여부 조정 가능)"""
        if text is None:
            text = ""
        
        text = str(text)        
        
        # 볼드 여부에 따라 폰트 경로 선택
        font_path = self.bold_font_path if bold else self.font_path
        font = ImageFont.truetype(font_path, font_size)
        
        # 임시 이미지로 텍스트 크기 계산
        temp_img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        temp_draw = ImageDraw.Draw(temp_img)
        # text_width, text_height = temp_draw.textsize(text, font=font)
        bbox = temp_draw.textbbox((0, 0), text, font=font)  # (x, y) 좌표와 텍스트, 폰트를 인자로 전달
        text_width = bbox[2] - bbox[0]  # 오른쪽 - 왼쪽
        text_height = bbox[3] - bbox[1]  # 아래 - 위
        
        # 텍스트를 위한 충분한 크기의 레이어 생성
        text_layer = layers.RectLayer((text_width + 50, text_height + 20), (0, 0, 0, 0))
        
        # 텍스트 그리기
        if text_layer.image.dtype != np.uint8:
            text_layer.image = text_layer.image.astype(np.uint8)
        img = Image.fromarray(text_layer.image)
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), text, font=font, fill=(0, 0, 0, 255))
        
        # 이미지를 다시 레이어로 변환
        text_layer.image = np.array(img)
        
        # 위치 설정
        text_layer.left = position[0]
        text_layer.top = position[1]
        
        return text_layer

    def split_text_to_fit_width(self, text, font, max_width):
        """텍스트를 최대 너비에 맞게 여러 줄로 분할"""
        if not text:
            return []
            
        temp_img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        temp_draw = ImageDraw.Draw(temp_img)
        
        words = text.split()
        lines = []
        current_line = words[0]
        
        for word in words[1:]:
            # 현재 라인과 새 단어를 합쳤을 때 너비 계산
            test_line = current_line + " " + word
            bbox = temp_draw.textbbox((0, 0), test_line, font=font)
            text_width = bbox[2] - bbox[0]
            
            if text_width <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        
        lines.append(current_line)  # 마지막 라인 추가
        return lines

    def init_save(self, root):
        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)

    def save(self, root, data, idx):
        image = data["image"]
        label = data["label"]
        quality = data["quality"]
        business_info = data["business_info"]
        combined_info = data["combined_info"]

        # split
        split_idx = self.split_indexes[idx % len(self.split_indexes)]
        output_dirpath_images = os.path.join(root, self.splits[split_idx], 'images')
        output_dirpath_annotations = os.path.join(root, self.splits[split_idx], 'annotations')

        # save image
        image_filename = f"business_reg_{idx}.jpg"
        image_filepath = os.path.join(output_dirpath_images, image_filename)
        os.makedirs(os.path.dirname(image_filepath), exist_ok=True)
        image = Image.fromarray(image[..., :3].astype(np.uint8))
        image.save(image_filepath, quality=quality)

        # save metadata (gt_json)
        # metadata_filename = "metadata.json"
        annotation_filename = f"{image_filename.split('.')[0]}.json"
        annotation_filepath = os.path.join(output_dirpath_annotations, annotation_filename)
        os.makedirs(os.path.dirname(annotation_filepath), exist_ok=True)

        annotation_data = self.format_annotation(
            image_filename=image_filename, 
            keys=["text_sequence", "business_info", "combined_info"], 
            values=[label, business_info, combined_info]
        )
        with open(annotation_filepath, "a") as annotation_file:
            json.dump(annotation_data, annotation_file, ensure_ascii=False, indent=4)
        

    
    def end_save(self, root):
        pass

    def format_annotation(self, image_filename: str, keys: List[str], values: List[Any]):
        """
        Fit gt_parse contents to huggingface dataset's format
        keys and values, whose lengths are equal, are used to constrcut 'gt_parse' field in 'ground_truth' field
        Args:
            keys: List of task_name
            values: List of actual gt data corresponding to each task_name
        """
        assert len(keys) == len(values), "Length does not match: keys({}), values({})".format(len(keys), len(values))

        _gt_parse_v = dict()
        for k, v in zip(keys, values):
            _gt_parse_v[k] = v

        # 레이블 포맷팅
        annotation_data={
            "image": image_filename,
            "gt": f"<HEAD>사업자등록증</HEAD>" +
                f"<사업자종류>{_gt_parse_v['business_info']['사업자종류']}</사업자종류>" +
                f"<사업자등록번호>{_gt_parse_v['business_info']['등록번호']}</사업자등록번호>" +
                f"<상호>{_gt_parse_v['business_info']['상호']}</상호>" +
                f"<대표자>{_gt_parse_v['business_info']['대표자']}</대표자>" +
                f"<개업연월일>{_gt_parse_v['business_info']['개업일']}</개업연월일>" +
                f"<법인등록번호>{_gt_parse_v['business_info']['법인등록번호']}</법인등록번호>" +
                f"<사업장소재지>{_gt_parse_v['business_info']['사업장주소']}</사업장소재지>" +
                f"<본점소재지>{_gt_parse_v['business_info']['본점주소']}</본점소재지>" +
                f"<업태>{', '.join(_gt_parse_v['business_info']['업태'])}</업태>" +
                f"<종목>{', '.join(_gt_parse_v['business_info']['종목'])}</종목>" +
                f"<발급사유>{_gt_parse_v['business_info']['발급사유']}</발급사유>" +
                f"<발급일자>{_gt_parse_v['business_info']['발급일']}</발급일자>" +
                f"<세무서명>{_gt_parse_v['business_info']['세무서']}</세무서명>",
            "texts": _gt_parse_v['text_sequence']
        }
        
        # gt_parse = {"gt_parse": _gt_parse_v}
        # gt_parse_str = json.dumps(annotation_data, ensure_ascii=False, indent=4)
        # metadata = {"file_name": image_filename, "ground_truth": gt_parse_str}
        return annotation_data

class BusinessInfoGenerator:
    def __init__(self, dataset_path: str = 'resources/business_registration/biz_info.csv'):
        self.dataset_path = dataset_path
        self.information, self.tax_office = self._read_information()

    def generate_random_info(self) -> Dict[str, str]:
        """사업자 정보 생성"""
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
    
    def _read_information(self):
        information = pd.read_csv(self.dataset_path, dtype={'column_name': str}, low_memory=False)
        with open('dataset/tax_office_list.txt', 'r', encoding='utf-8') as f:
            tax_offices = [line.strip() for line in f.readlines()]
        return information, tax_offices

    def _generate_business_number(self):
        return f"{random.randint(100000, 999999)}-{random.randint(1000000, 9999999)}"
    
    def _generate_biz_name(self, biz_name):        
        if random.random() < 0.5:
            prefixes = ["주식회사", "유한회사", "(주)"]
            if "주식회사" not in biz_name or "(주)" not in biz_name:
                biz_name = f"{random.choice(prefixes)} {biz_name}"

        return biz_name

    def _generate_open_date(self, date):
        return f"{str(date)[:4]} 년 {str(date)[4:6]} 월 {str(date)[6:]} 일"
    
    def _generate_business_type(self, num):
        main_types = ["제조", "건설", "도소매", "운수", "숙박", "정보통신", "금융", "부동산", "과학기술 서비스", "교육 서비스", "전문, 과학 및 기술서비스업"]
        sub_types = ["금속 가공제품 제조업", "전자부품 제조업", "소프트웨어 개발", "유통업", "도매 및 상품 중개업", "소매업", "일반 건설업", "전문직별 공사업", "전기 전자공학 연구개발업"]
    
        return random.sample(main_types, num), random.sample(sub_types, num)
    
    def _generate_issue_reason(self):
        if random.random() < 0.3:
            reasons = ['신규 개업 신고', '사업장 이전', '대표자 변경', '업종 변경', '휴업 및 재개업', '분실 또는 훼손', '기타 변경 등록']
            return random.choice(reasons)
        else:
            return ""
    
    def _generate_issue_date(self, start_year=2022, end_year=2025):
        start_date = datetime.date(start_year, 1, 1)
        end_date = datetime.date(end_year, 12, 31)
        delta = end_date - start_date
        random_days = random.randrange(delta.days)
        date = start_date + datetime.timedelta(days=random_days)
        return date.strftime("%Y 년 %m 월 %d 일")
    
    def _generate_issuer_name(self):
        issuer_name = f"{random.choice(self.tax_office)}장"
        return ' '.join(issuer_name)
