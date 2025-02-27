import os
import json
import torch
from torch.utils.data import Dataset, random_split
from torch.nn.utils.rnn import pad_sequence
from transformers import DonutProcessor, VisionEncoderDecoderModel, Trainer, TrainingArguments
from PIL import Image

# Custom Trainer: compute_loss에서 num_items_in_batch 제거
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # 불필요한 인자 제거
        if "num_items_in_batch" in inputs:
            inputs.pop("num_items_in_batch")
        outputs = model(**inputs)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        return (loss, outputs) if return_outputs else loss

class DonutFinetuningDataset(Dataset):
    """
    donut_format 폴더 내 개별 JSON 파일들을 읽어와
    이미지와 ground truth (태스크 포맷 문자열)를 반환하는 데이터셋 클래스.
    """
    def __init__(self, json_dir, processor):
        self.processor = processor
        self.samples = []
        for file in os.listdir(json_dir):
            if file.endswith(".json"):
                with open(os.path.join(json_dir, file), "r", encoding="utf-8") as f:
                    sample = json.load(f)
                    self.samples.append(sample)
        print(f"총 {len(self.samples)}개의 샘플 로드 완료")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        # 이미지 경로가 저장되어 있으므로 이미지 로드 및 RGB 변환
        image = Image.open(sample["image"]).convert("RGB")
        # 최신 processor 호출 방식: legacy=False로 설정
        pixel_values = self.processor(image, return_tensors="pt", legacy=False).pixel_values.squeeze(0)
        
        # ground_truth는 이미 task 포맷 (<s_ko> 등)으로 구성된 문자열
        target_text = sample["ground_truth"]
        labels = self.processor.tokenizer(target_text, add_special_tokens=False).input_ids
        labels = torch.tensor(labels)
        
        return {"pixel_values": pixel_values, "labels": labels}

def collate_fn(batch):
    # 배치 내 이미지 텐서는 스택, 텍스트 시퀀스는 pad_sequence로 패딩
    pixel_values = torch.stack([x["pixel_values"] for x in batch])
    labels = [x["labels"] for x in batch]
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    return {"pixel_values": pixel_values, "labels": labels}

def main():
    # 1. 사전학습된 Donut 모델과 processor 로드
    model_name = "naver-clova-ix/donut-base"
    processor = DonutProcessor.from_pretrained(model_name, cache_dir='hub/model')
    model = VisionEncoderDecoderModel.from_pretrained(model_name, cache_dir='hub/model')

    # 2. donut_format 폴더에서 데이터셋 로드
    json_dir = "donut_dataset/donut_format"  # 개별 JSON 파일들이 위치한 폴더
    dataset = DonutFinetuningDataset(json_dir, processor)
    
    # 3. 학습/검증 데이터셋 분할 (예: 80% train, 20% validation)
    total_samples = len(dataset)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Train 샘플: {len(train_dataset)}, Validation 샘플: {len(val_dataset)}")
    
    # 4. TrainingArguments 설정
    training_args = TrainingArguments(
        output_dir="donut_finetuned",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        evaluation_strategy="steps",
        num_train_epochs=10,
        logging_steps=50,
        save_steps=500,
        eval_steps=500,
        learning_rate=5e-5,
        weight_decay=0.01,
        fp16=True,  # GPU fp16 사용 환경인 경우
        save_total_limit=3,
        report_to="none",  # 로그 플랫폼 사용하지 않을 경우
    )

    # 5. CustomTrainer 객체 생성 (deprecated warning 해결을 위해 processing_class 사용)
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        processing_class=processor,  # tokenizer 대신 사용하여 warning 해소
    )

    # 6. 파인튜닝 시작
    trainer.train()

if __name__ == "__main__":
    main()
