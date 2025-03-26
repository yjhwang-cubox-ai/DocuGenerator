import cv2
import numpy as np
import random
from PIL import Image

class DocumentAugmentor:
    def __init__(self, max_num_augmentations=3):
        self.num_augmentations = random.randint(0, max_num_augmentations)
        self.augmentations = [
            self.apply_perspective_transform,
            self.add_noise,
            self.random_rotation,
            self.random_blur,
            self.adjust_brightness_contrast,
            # self.random_shear
        ]

    def apply_perspective_transform(self, image):
        """원근 변환을 적용하여 문서가 기울어진 효과 부여"""
        rows, cols, _ = image.shape
        src_pts = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]])
        margin = 50
        dst_pts = np.float32([
            [np.random.randint(0, margin), np.random.randint(0, margin)],
            [cols - np.random.randint(0, margin), np.random.randint(0, margin)],
            [cols - np.random.randint(0, margin), rows - np.random.randint(0, margin)],
            [np.random.randint(0, margin), rows - np.random.randint(0, margin)]
        ])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, M, (cols, rows))
        return warped

    def add_noise(self, image):
        """가우시안 노이즈를 추가하여 스캔 시 발생할 수 있는 노이즈 효과 부여"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        noise = np.zeros_like(gray, dtype=np.int16)
        cv2.randn(noise, 0, 20)
        noisy = gray.astype(np.int16) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        noisy_bgr = cv2.cvtColor(noisy, cv2.COLOR_GRAY2BGR)
        return noisy_bgr

    def random_rotation(self, image, angle_range=(-5, 5)):
        """랜덤 각도로 이미지를 회전"""
        angle = np.random.uniform(angle_range[0], angle_range[1])
        rows, cols = image.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        rotated = cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
        return rotated

    def random_blur(self, image, kernel_size=(3, 3)):
        """가우시안 블러 적용"""
        return cv2.GaussianBlur(image, kernel_size, 0)

    def adjust_brightness_contrast(self, image, brightness=10, contrast=15):
        """
        밝기와 대비를 조정
        :param brightness: -100 ~ 100 범위 내에서 조정 가능
        :param contrast: -100 ~ 100 범위 내에서 조정 가능
        """
        beta = brightness
        alpha = 1 + (contrast / 100.0)
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return adjusted

    # def random_shear(self, image, shear_range=(-0.2, 0.2)):
    #     """시어 변환을 적용하여 이미지에 기울기를 추가"""
    #     rows, cols, _ = image.shape
    #     shear = np.random.uniform(shear_range[0], shear_range[1])
    #     M = np.float32([[1, shear, 0],
    #                     [0, 1, 0]])
    #     nW = cols + abs(shear * rows)
    #     sheared = cv2.warpAffine(image, M, (int(nW), rows), borderMode=cv2.BORDER_REPLICATE)
    #     sheared = cv2.resize(sheared, (cols, rows))
    #     return sheared

    def apply_random_augmentations(self, image):
        """
        여러 augmentation 함수 중에서 num_augmentations 개수를 랜덤 선택하여 순차적으로 적용
        :param image: 입력 이미지
        :return: augmentation이 적용된 이미지
        """
        num_augs = min(self.num_augmentations, len(self.augmentations))
        selected_augs = random.sample(self.augmentations, num_augs)
        
        image = np.array(image)
        aug_img = image.copy()
        for aug in selected_augs:
            aug_img = aug(aug_img)
        aug_img = Image.fromarray(aug_img)
        return aug_img