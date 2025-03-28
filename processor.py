import random
import datetime
from PIL import Image, ImageDraw, ImageFont
import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from augmentation import DocumentAugmentor
from augraphy import *


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
        os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)
        
        # Donut 데이터셋을 위한 JSON 파일 경로
        self.donut_json_path = os.path.join(self.output_dir, "donut_dataset.json")
        # Donut 데이터셋 구조 초기화
        self.donut_dataset = {
            "version": "v1.0",
            "data": []
        }
        
        # 기존 annotations
        self.annotations_path = os.path.join(self.output_dir, "annotations.json")
        self.annotations = {"images": [], "annotations": []}
        
        # 폰트 미리 로드
        self.title_font = ImageFont.truetype(self.font_path, 70)
        self.normal_font = ImageFont.truetype(self.font_path, 45)
        self.small_font = ImageFont.truetype(self.font_path, 32)
        self.smallest_font = ImageFont.truetype(self.font_path, 28)
        self.reg_date_font = ImageFont.truetype(self.font_path, 40)
        self.tax_office_font = ImageFont.truetype(self.font_path, 60)

        self.information, self.tax_office = self._read_information()
        #artifact pipeline
        self.augraphy_pipeline = default_augraphy_pipeline()

    def _read_information(self):
        information = pd.read_csv(self.dataset_path)
        with open('dataset/tax_office_list.txt', 'r', encoding='utf-8') as f:
            tax_offices = [line.strip() for line in f.readlines()]
        return information, tax_offices
    
    def generate_business_number(self):
        return f"{random.randint(100000, 999999)}-{random.randint(1000000, 9999999)}"

    def generate_random_date(self, start_year=2022, end_year=2025):
        start_date = datetime.date(start_year, 1, 1)
        end_date = datetime.date(end_year, 12, 31)
        delta = end_date - start_date
        random_days = random.randrange(delta.days)
        date = start_date + datetime.timedelta(days=random_days)
        return date.strftime("%Y 년 %m 월 %d 일")

    def generate_business_type(self, num):
        main_types = ["제조", "건설", "도소매", "운수", "숙박", "정보통신", "금융", "부동산", "과학기술 서비스", "교육 서비스", "전문, 과학 및 기술서비스업"]
        sub_types = ["금속 가공제품 제조업", "전자부품 제조업", "소프트웨어 개발", "유통업", "도매 및 상품 중개업", "소매업", "일반 건설업", "전문직별 공사업", "전기 전자공학 연구개발업"]
    
        return random.sample(main_types, num), random.sample(sub_types, num)

    def get_text_bbox(self, draw, text, x, y, font):
        """텍스트의 bbox 좌표를 계산하여 반환"""
        # 텍스트 크기 계산
        bbox = font.getbbox(text)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        # 실제 화면상 좌표 계산 (left, top, right, bottom)
        bbox_coords = (x, y, x + width, y + height)
        return bbox_coords

    def draw_text_with_bbox(self, draw, text, x, y, font, fill, image_id, idx):
        """텍스트를 그리고 bbox 정보를 저장"""
        # 텍스트 그리기
        draw.text((x, y), text, font=font, fill=fill)
        
        # bbox 계산
        bbox = self.get_text_bbox(draw, text, x, y, font)
        
        # annotations에 추가
        annotation = {
            "id": idx,
            "image_id": image_id,
            "bbox": [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],  # [x, y, width, height] 형식
            "text": text,
            "area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        }
        self.annotations["annotations"].append(annotation)
        
        return bbox

    def create_single_image(self, index=0):
        """단일 사업자등록증 이미지를 생성하여 파일로 저장하고 Donut 모델용 JSON 데이터 생성"""
        # 1) 템플릿 또는 단색 배경 이미지 준비
        if self.template_path:
            background_images = [f for f in os.listdir(self.template_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            background_image_path = os.path.join(self.template_path, background_images[index % len(background_images)])
            image = Image.open(background_image_path).convert("RGB").resize((self.width, self.height))
        else:
            image = Image.new("RGB", (self.width, self.height), self.background_color)

        draw = ImageDraw.Draw(image)

        # 2) 데이터 생성
        data = self.information.sample(n=1)
        information = {
            "HEAD": "사업자등록증",            
            "사업자종류": None,
            "등록번호": data['사업자등록번호'].values[0],
            "상호": data['상호'].values[0],
            "대표자명": data['대표자명'].values[0],
            "개업연월일": data['신고일자'].values[0],
            "법인등록번호": None,
            "사업장 소재지": data['사업장소재지(도로명)'].values[0][:30],
            "본점 소재지": None,
            "업태": None,
            "종목": None,
            "발급사유": None,
            "발급일자": None,
            "세무서명": None            
        }

        # 2-1.사업자 종류
        information["사업자종류"] = "법인사업자" if data['법인여부'].values[0] == "법인" else "일반과세자"

        # 2-2.랜덤으로 상호명에 주식회사 붙이기
        if random.random() < 0.5:
            prefixes = ["주식회사", "유한회사", "(주)"]
            if "주식회사" not in information['상호']:
                information['상호'] = f"{random.choice(prefixes)} {information['상호']}"

        # 2-3.개업일자 데이터 형태 수정
        information['개업연월일'] = f"{str(information['개업연월일'])[:4]} 년 {str(information['개업연월일'])[4:6]} 월 {str(information['개업연월일'])[6:]} 일"
        
        # 2-4.법인등록번호 생성
        information['법인등록번호'] = self.generate_business_number()
        
        # 2-5.업태, 종목
        업태_list, 종목_list = self.generate_business_type(random.randint(1, 5))
        information["업태"] = 업태_list
        information["종목"] = 종목_list
        
        # 2-6.발급일자 랜덤 선택
        information['발급일자'] = self.generate_random_date()

        # 2-7.세무서명 랜덤 선택
        information['세무서명'] = f"{random.choice(self.tax_office)}장"
        
        # 2-8.발급사유
        if random.random() < 0.3:
            reasons = ['신규 개업 신고', '사업장 이전', '대표자 변경', '업종 변경', '휴업 및 재개업', '분실 또는 훼손', '기타 변경 등록']
            information['발급사유'] = random.choice(reasons)
        else:
            information['발급사유'] = ""
        
        # 2-9.본점 소재지
        information['본점 소재지'] = information['사업장 소재지']

        # 3) 텍스트 배치 및 bbox 정보 저장
        filename = f"business_reg_{index:05d}.png"
        image_path = os.path.join("images", filename)
        full_image_path = os.path.join(self.output_dir, image_path)
        
        image_info = {
            "id": index,
            "file_name": filename,
            "width": self.width,
            "height": self.height
        }
        self.annotations["images"].append(image_info)
        
        annotation_idx = len(self.annotations["annotations"])
        
        # 제목 텍스트 배치
        head_text = ' '.join(information['HEAD'])
        head_x, head_y = self.width//2 - 280, 280
        self.draw_text_with_bbox(draw, head_text, head_x, head_y, self.title_font, (0, 0, 0), index, annotation_idx)
        annotation_idx += 1
        
        # 사업자 종류 배치
        kind_x, kind_y = self.width//2 - 150, 360
        self.draw_text_with_bbox(draw, f"( {information['사업자종류']} )", kind_x, kind_y, self.normal_font, (0, 0, 0), index, annotation_idx)
        annotation_idx += 1
        
        # 등록번호 배치
        reg_y = 430
        reg_label_x = 500
        reg_value_x = 730
        self.draw_text_with_bbox(draw, "등록번호 : ", reg_label_x, reg_y, self.normal_font, (0, 0, 0), index, annotation_idx)
        annotation_idx += 1
        self.draw_text_with_bbox(draw, information['등록번호'], reg_value_x, reg_y, self.normal_font, (0, 0, 0), index, annotation_idx)
        annotation_idx += 1
        
        anchor_x = 150
        # 상호 배치
        company_y = reg_y + 80
        company_label_x = anchor_x
        company_value_x = 420
        self.draw_text_with_bbox(draw, "법인명(단체명)  :", company_label_x, company_y, self.small_font, (0, 0, 0), index, annotation_idx)
        annotation_idx += 1
        self.draw_text_with_bbox(draw, information['상호'], company_value_x, company_y, self.small_font, (0, 0, 0), index, annotation_idx)
        annotation_idx += 1
        
        # 대표자명 배치
        owner_y = company_y + 50
        owner_label_x = anchor_x
        owner_value_x = 420
        self.draw_text_with_bbox(draw, "대      표      자 :", owner_label_x, owner_y, self.small_font, (0, 0, 0), index, annotation_idx)
        annotation_idx += 1
        self.draw_text_with_bbox(draw, information['대표자명'], owner_value_x, owner_y, self.small_font, (0, 0, 0), index, annotation_idx)
        annotation_idx += 1
        
        # 개업연월일 배치
        open_y = owner_y + 80
        open_label_x = anchor_x
        open_value_x = 420
        self.draw_text_with_bbox(draw, "개 업 연 월 일   :", open_label_x, open_y, self.small_font, (0, 0, 0), index, annotation_idx)
        annotation_idx += 1
        self.draw_text_with_bbox(draw, information['개업연월일'], open_value_x, open_y, self.small_font, (0, 0, 0), index, annotation_idx)
        annotation_idx += 1
        
        # 법인등록번호 배치
        self.draw_text_with_bbox(draw, "법인등록번호  :", open_value_x + 360, open_y, self.small_font, (0, 0, 0), index, annotation_idx)
        annotation_idx += 1
        self.draw_text_with_bbox(draw, information['법인등록번호'], open_value_x + 600, open_y, self.small_font, (0, 0, 0), index, annotation_idx)
        annotation_idx += 1
        
        # 사업장 소재지 배치
        addr_y = open_y + 50
        addr_label_x = anchor_x
        addr_value_x = 420
        self.draw_text_with_bbox(draw, "사업장  소재지  :", addr_label_x, addr_y, self.small_font, (0, 0, 0), index, annotation_idx)
        annotation_idx += 1
        self.draw_text_with_bbox(draw, information['사업장 소재지'], addr_value_x, addr_y, self.small_font, (0, 0, 0), index, annotation_idx)
        annotation_idx += 1

        # 본점 소재지 배치
        ho_y = addr_y + 80
        ho_label_x = anchor_x
        ho_value_x = 420
        self.draw_text_with_bbox(draw, "본 점 소 재 지  :", ho_label_x, ho_y, self.small_font, (0, 0, 0), index, annotation_idx)
        annotation_idx += 1
        self.draw_text_with_bbox(draw, information['본점 소재지'], ho_value_x, ho_y, self.small_font, (0, 0, 0), index, annotation_idx)
        annotation_idx += 1
        
        # 업태 배치
        business_y = ho_y + 80
        business_label_x = anchor_x
        business_value_x = 420
        self.draw_text_with_bbox(draw, "사 업 의  종 류  :", business_label_x, business_y, self.small_font, (0, 0, 0), index, annotation_idx)
        annotation_idx += 1
        self.draw_text_with_bbox(draw, "업태", business_value_x, business_y, self.small_font, (0, 0, 0), index, annotation_idx)
        annotation_idx += 1        
        for i, btype in enumerate(information['업태']):
            y_offset = business_y + i * 50
            self.draw_text_with_bbox(draw, btype, business_value_x + 80, y_offset, self.smallest_font, (0, 0, 0), index, annotation_idx)
            annotation_idx += 1
        
        # # 종목 배치
        self.draw_text_with_bbox(draw, "종목", business_value_x + 440, business_y, self.small_font, (0, 0, 0), index, annotation_idx)
        annotation_idx += 1
        for i, item in enumerate(information['종목']):
            y_offset = business_y + i * 50
            self.draw_text_with_bbox(draw, item, business_value_x + 530, y_offset, self.smallest_font, (0, 0, 0), index, annotation_idx)
            annotation_idx += 1

        # 기타 텍스트 배치
        text_y = 1100
        text_label_x = anchor_x
        text_value_x = 420
        self.draw_text_with_bbox(draw, "발  급  사  유  :", text_label_x, text_y, self.small_font, (0, 0, 0), index, annotation_idx)
        annotation_idx += 1
        self.draw_text_with_bbox(draw, information['발급사유'], text_value_x, text_y, self.small_font, (0, 0, 0), index, annotation_idx)
        annotation_idx += 1

        text_y = 1400
        text_label_x = anchor_x
        text_value_x = 420
        self.draw_text_with_bbox(draw, "사업자 단위 과세 적용사업자 여부  :", text_label_x, text_y, self.small_font, (0, 0, 0), index, annotation_idx)
        annotation_idx += 1
        taxation_text = "여(   ) 부(∨)" if random.random() < 0.8 else "여(∨) 부(   )"
        self.draw_text_with_bbox(draw, taxation_text, text_label_x + 550, text_y, self.small_font, (0, 0, 0), index, annotation_idx)
        annotation_idx += 1
        self.draw_text_with_bbox(draw, "전자세금계산서 전용 전자우편주소 :", text_label_x, text_y+50, self.small_font, (0, 0, 0), index, annotation_idx)
        annotation_idx += 1
        
        # 발급일자 배치
        date_x, date_y = self.width//2 - 190, 1680
        self.draw_text_with_bbox(draw, information['발급일자'], date_x, date_y, self.reg_date_font, (0, 0, 0), index, annotation_idx)
        annotation_idx += 1
        
        # 세무서명 배치
        tax_x, tax_y = self.width//2 - 230, 1760
        self.draw_text_with_bbox(draw, ' '.join(information['세무서명']), tax_x, tax_y, self.tax_office_font, (0, 0, 0), index, annotation_idx)
        
        # 4) augment 적용 + 저장
        # augmented_img = self.random_augmentations(image)
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        augmented_img = self.augraphy_pipeline(opencv_image)
        # augmented_img.save(full_image_path)
        cv2.imwrite(full_image_path, augmented_img)
        
        # 5) Donut 모델용 ground truth JSON 생성 (업태와 종목은 리스트를 문자열로 변환)
        업태_str = ", ".join(information['업태'])
        종목_str = ", ".join(information['종목'])

        # Donut 모델용 ground truth
        ground_truth = {
            "HEAD": information['HEAD'],
            "사업자종류": information['사업자종류'],
            "사업자등록번호": information['등록번호'],            
            "상호": information['상호'],
            "대표자": information['대표자명'],
            "개업연월일": information['개업연월일'],
            "법인등록번호": information['법인등록번호'],
            "사업장소재지": information['사업장 소재지'],
            "본점소재지": information['본점 소재지'],
            "업태": 업태_str,
            "종목": 종목_str,
            "발급사유": information['발급사유'],
            "발급일자": information['발급일자'],            
            "세무서명": information['세무서명']
        }
        
        # OCR 텍스트 생성 (실제 OCR 결과를 모방)
        ocr_text = f"사업자등록증\n({information['사업자종류']})\n"
        ocr_text += f"등록번호 : {information['등록번호']}\n"
        ocr_text += f"법인명(단체명) : {information['상호']}\n"
        ocr_text += f"대표자 : {information['대표자명']}\n"
        ocr_text += f"개업연월일 : {information['개업연월일']}\n"
        ocr_text += f"법인등록번호 : {information['법인등록번호']}\n"
        ocr_text += f"사업장소재지 : {information['사업장 소재지']}\n"
        ocr_text += f"본점소재지 : {information['본점 소재지']}\n"
        ocr_text += f"사업의 종류 : 업태 {업태_str} 종목 {종목_str}\n"
        ocr_text += f"발급사유 : {information['발급사유']}\n"
        ocr_text += f"발급일자 : {information['발급일자']}\n"
        ocr_text += f"{information['세무서명']}"
        
        # Donut 데이터셋에 추가
        donut_item = {
            "uid": f"business_reg_{index:05d}",
            "image_path": image_path,
            "ground_truth": ground_truth,
            "ocr_text": ocr_text
        }
        
        self.donut_dataset["data"].append(donut_item)
        
        return full_image_path

    def create_bulk_images(self, n=10000):
        """n장의 이미지를 순차적으로 생성"""
        for i in tqdm(range(n)):
            self.create_single_image(index=i)
            if i % 1000 == 0 and i != 0:
                print(f"{i}장 생성 완료")
        
        # 모든 이미지 생성 후 annotations.json 저장
        with open(self.annotations_path, 'w', encoding='utf-8') as f:
            json.dump(self.annotations, f, ensure_ascii=False, indent=4)
        print(f"annotations 저장 완료: {self.annotations_path}")
        
        # Donut 모델용 JSON 저장
        with open(self.donut_json_path, 'w', encoding='utf-8') as f:
            json.dump(self.donut_dataset, f, ensure_ascii=False, indent=4)
        print(f"Donut 데이터셋 저장 완료: {self.donut_json_path}")
        
        # Donut 모델용 개별 JSON 파일 생성 (한국어 모델 학습을 위한 형식)
        donut_dir = os.path.join(self.output_dir, "donut_format")
        os.makedirs(donut_dir, exist_ok=True)
        
        for item in self.donut_dataset["data"]:
            uid = item["uid"]
            image_path = os.path.join(os.path.dirname(self.output_dir), self.output_dir, item["image_path"])
            
            # ground_truth를 모델 학습에 필요한 문자열로 변환
            task_format = ""
            for key, value in item["ground_truth"].items():
                task_format += f"<{key}>{value}</{key}>"
            task_format += "</s>"
            
            # 단일 아이템 JSON
            donut_item = {
                "image": image_path,
                "ground_truth": task_format
            }
            
            # 저장
            item_path = os.path.join(donut_dir, f"{uid}.json")
            with open(item_path, 'w', encoding='utf-8') as f:
                json.dump(donut_item, f, ensure_ascii=False, indent=4)
                
        print(f"Donut 개별 JSON 파일 저장 완료: {donut_dir}")
        
        # Train/Val/Test 분할
        self.create_train_val_test_split()
    
    def create_train_val_test_split(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        """데이터셋을 학습/검증/테스트 세트로 분할하는 함수"""
        # 전체 데이터 개수
        total_count = len(self.donut_dataset["data"])
        indices = list(range(total_count))
        random.shuffle(indices)
        
        # 분할 인덱스 계산
        train_count = int(total_count * train_ratio)
        val_count = int(total_count * val_ratio)
        
        train_indices = indices[:train_count]
        val_indices = indices[train_count:train_count + val_count]
        test_indices = indices[train_count + val_count:]
        
        # 분할된 데이터셋 생성
        splits = {
            "train": [self.donut_dataset["data"][i] for i in train_indices],
            "validation": [self.donut_dataset["data"][i] for i in val_indices],
            "test": [self.donut_dataset["data"][i] for i in test_indices]
        }
        
        # 분할된 데이터셋 저장
        for split_name, split_data in splits.items():
            split_path = os.path.join(self.output_dir, f"donut_{split_name}.json")
            with open(split_path, 'w', encoding='utf-8') as f:
                json.dump({"version": "v1.0", "data": split_data}, f, ensure_ascii=False, indent=4)
            print(f"{split_name} 데이터셋 저장 완료 ({len(split_data)} 건): {split_path}")
    
    def random_augmentations(self, img):
        augmentor = DocumentAugmentor(max_num_augmentations=3)
        augmented_img = augmentor.apply_random_augmentations(img)
        return augmented_img


if __name__ == "__main__":
    # 예시 사용
    generator = BusinessRegistrationGenerator(
        template_path='dataset/background_image',      # 템플릿 이미지 경로가 있다면 여기 넣기
        dataset_path='dataset/total.csv',
        font_path="fonts/gulim.ttc",
        width=1478,
        height=2074,
        background_color=(255, 255, 255),
        output_dir="BRCDataset_V6"
    )
    generator.create_bulk_images(n=10)  # 예: 10장 생성