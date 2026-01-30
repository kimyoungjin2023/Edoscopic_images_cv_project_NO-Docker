# dataset

import os
import glob
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset

class MedicalFolderDataset(Dataset):
    """
    의료 내시경 영상과 라벨(마스크) 데이터를 Mask R-CNN 규격에 맞게 로드하는 클래스
    하위 폴더(위, 대장 등) 내의 모든 파일을 재귀적으로 탐색
    """
    def __init__(self, image_root, label_root, transforms=None):
        self.transforms = transforms
        
        # [핵심 수정] glob.glob과 recursive=True를 사용하여 모든 하위 폴더의 파일을 가져옴
        # 정렬(sorted)을 통해 이미지와 라벨의 순서를 일치시킴
        self.img_paths = sorted(glob.glob(os.path.join(image_root, "**/*.jpg"), recursive=True))
        self.label_paths = sorted(glob.glob(os.path.join(label_root, "**/*.png"), recursive=True))

        # 데이터 개수 확인용 출력
        print(f"✅ 데이터 로드 완료: 이미지 {len(self.img_paths)}장, 라벨 {len(self.label_paths)}장")

    def __getitem__(self, idx):
        # 1. 파일 경로 (이미 리스트에 전체 경로가 포함되어 있음)
        img_path = self.img_paths[idx]
        mask_path = self.label_paths[idx]

        # 2. 이미지 로드 및 색상 공간 변환
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 3. 마스크 로드 (Grayscale)
        mask = cv2.imread(mask_path, 0)

        # 4. 객체(병변) 정보 추출 (배경 0 제외)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:] 

        # 각 클래스별 이진 마스크 생성
        masks = (mask == obj_ids[:, None, None]).astype(np.uint8)

        # 5. Bounding Box 추출
        boxes = []
        for i in range(len(obj_ids)):
            pos = np.where(masks[i])
            xmin, xmax = np.min(pos[1]), np.max(pos[1])
            ymin, ymax = np.min(pos[0]), np.max(pos[0])
            
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])

        # 6. PyTorch 텐서 변환
        num_objs = len(boxes)
        if num_objs == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, mask.shape[0], mask.shape[1]), dtype=torch.uint8)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(obj_ids[:num_objs], dtype=torch.int64)
            masks = torch.as_tensor(masks[:num_objs], dtype=torch.uint8)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if num_objs > 0 else torch.tensor([0]),
            "iscrowd": torch.zeros((num_objs,), dtype=torch.int64)
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.img_paths)

def collate_fn(batch):
    return tuple(zip(*batch))