import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset

class MedicalFolderDataset(Dataset):
    """
    의료 내시경 영상과 라벨(마스크) 데이터를 Mask R-CNN 규격에 맞게 로드하는 클래스
    """
    def __init__(self, image_root, label_root, transforms=None):
        self.image_root = image_root
        self.label_root = label_root
        self.transforms = transforms
        
        # 숨김 파일(.)을 제외하고 파일 리스트를 정렬하여 이미지와 라벨의 순서를 맞춤
        self.imgs = sorted([f for f in os.listdir(image_root) if not f.startswith('.')])
        self.labels = sorted([f for f in os.listdir(label_root) if not f.startswith('.')])

    def __getitem__(self, idx):
        # 1. 이미지 및 마스크 경로 설정
        img_path = os.path.join(self.image_root, self.imgs[idx])
        mask_path = os.path.join(self.label_root, self.labels[idx])

        # 2. 이미지 로드 및 색상 공간 변환 (BGR -> RGB)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 3. 마스크 로드 (Grayscale 모드로 읽기)
        # 마스크 이미지의 픽셀 값: 0(배경), 1(용종), 2(암) 등 클래스 번호
        mask = cv2.imread(mask_path, 0)

        # 4. 객체(병변) 정보 추출
        # 마스크에서 사용된 고유한 값들을 찾음 (0번인 배경은 제외)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:] 

        # 각 클래스별로 이진 마스크(Binary Mask)를 생성
        # 예: 클래스가 2개라면 masks의 형태는 (2, 높이, 너비)
        masks = (mask == obj_ids[:, None, None]).astype(np.uint8)

        # 5. Bounding Box 추출 (마스크의 상하좌우 끝점을 찾음)
        boxes = []
        for i in range(len(obj_ids)):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            
            # 박스의 크기가 0인 경우를 방지하기 위한 안전장치
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])

        # 6. PyTorch 텐서로 변환
        num_objs = len(boxes)
        if num_objs == 0:
            # 만약 이미지에 병변이 없다면 빈 텐서를 생성하여 에러를 방지
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, mask.shape[0], mask.shape[1]), dtype=torch.uint8)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(obj_ids[:num_objs], dtype=torch.int64)
            masks = torch.as_tensor(masks[:num_objs], dtype=torch.uint8)

        image_id = torch.tensor([idx])
        
        # 박스의 넓이 계산 (Mask R-CNN 내부 연산에 필요)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if num_objs > 0 else torch.tensor([0])
        # 모든 객체는 iscrowd=0 (군집 객체 아님)으로 설정
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # 7. Mask R-CNN이 요구하는 최종 Target 딕셔너리
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        # 전처리(Resize, ToTensor 등) 적용
        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

def collate_fn(batch):
    """
    각 이미지마다 병변의 개수가 다르기 때문에, 
    기본 stack 방식이 아닌 튜플로 묶어주는 커스텀 함수가 반드시 필요함
    """
    return tuple(zip(*batch))