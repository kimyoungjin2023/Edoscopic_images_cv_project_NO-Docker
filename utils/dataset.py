"""
Endoscopic Image Dataset with COCO-style Annotation
- COCO format JSON 파일 파싱
- Classification, Detection, Segmentation 라벨 모두 제공
- PyTorch Dataset 형태로 구현
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
import cv2


class EndoscopicDataset(Dataset):
    """
    내시경 이미지 데이터셋
    
    데이터 구조 예시:
    data_root/
        images/
            train/
                image_001.jpg
                image_002.jpg
            val/
        annotations/
            train.json
            val.json
    
    COCO JSON 포맷:
    {
        "images": [{"id": 1, "file_name": "image_001.jpg", "height": 512, "width": 512}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [x, y, width, height],
                "segmentation": [[x1,y1,x2,y2,...]] or RLE,
                "area": float,
                "iscrowd": 0
            }
        ],
        "categories": [{"id": 1, "name": "polyp"}]
    }
    """
    
    def __init__(self, root, annotation_file, transforms=None, mode='train'):
        """
        Args:
            root (str): 이미지 루트 디렉토리
            annotation_file (str): COCO format annotation JSON 경로
            transforms: albumentations transforms
            mode (str): 'train' or 'val' or 'test'
        """
        self.root = root
        self.transforms = transforms
        self.mode = mode
        
        # COCO API로 annotation 로드
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
        
        # 카테고리 정보
        self.categories = {cat['id']: cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())}
        self.num_classes = len(self.categories) + 1  # +1 for background
        
        print(f"Loaded {len(self.image_ids)} images")
        print(f"Categories: {self.categories}")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        # 이미지 정보 가져오기
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info['file_name'])
        
        # 이미지 로드
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Annotation 가져오기
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Boxes, labels, masks 준비
        boxes = []
        labels = []
        masks = []
        areas = []
        iscrowd = []
        
        for ann in anns:
            # Bounding box: [x, y, width, height] -> [x1, y1, x2, y2]
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
            areas.append(ann['area'])
            iscrowd.append(ann.get('iscrowd', 0))
            
            # Mask 생성
            if isinstance(ann['segmentation'], list):
                # Polygon format
                rles = coco_mask.frPyObjects(
                    ann['segmentation'],
                    img_info['height'],
                    img_info['width']
                )
                mask = coco_mask.decode(rles)
                if len(mask.shape) < 3:
                    mask = mask[..., None]
                mask = mask.any(axis=2)
            else:
                # RLE format
                mask = coco_mask.decode(ann['segmentation'])
            
            masks.append(mask)
        
        # Transform 적용
        if self.transforms is not None:
            # Albumentations 사용
            transformed = self.transforms(
                image=image,
                masks=masks,
                bboxes=boxes,
                labels=labels
            )
            image = transformed['image']
            masks = transformed['masks']
            boxes = transformed['bboxes']
            labels = transformed['labels']
        
        # Tensor로 변환
        image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        num_objs = len(boxes)
        if num_objs > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        else:
            # 객체가 없는 경우
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, image.shape[1], image.shape[2]), dtype=torch.uint8)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        
        # Target dictionary 생성 (Mask R-CNN 형식)
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([img_id]),
            'area': areas,
            'iscrowd': iscrowd
        }
        
        return image, target
    
    def get_image_info(self, idx):
        """디버깅용: 이미지 정보 반환"""
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        return img_info


def collate_fn(batch):
    """
    Custom collate function for DataLoader
    Mask R-CNN은 배치마다 이미지 크기가 다를 수 있으므로 리스트로 반환
    """
    return tuple(zip(*batch))


class SimpleEndoscopicDataset(Dataset):
    """
    간단한 버전 - 이미지와 라벨만 있을 때 사용
    (COCO format 없이 직접 구현)
    """
    
    def __init__(self, image_dir, label_dir, transforms=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transforms = transforms
        
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # 이미지 로드
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 라벨 로드 (예: JSON 또는 텍스트 파일)
        label_name = img_name.replace('.jpg', '.json')
        label_path = os.path.join(self.label_dir, label_name)
        
        with open(label_path, 'r') as f:
            label_data = json.load(f)
        
        # Transform
        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed['image']
        
        # 여기서 label_data를 Mask R-CNN 형식으로 변환
        # (구현은 데이터 포맷에 따라 달라짐)
        
        return image, label_data


if __name__ == "__main__":
    # 데이터셋 테스트
    print("Testing EndoscopicDataset...")
    
    # 실제 사용 예시
    # dataset = EndoscopicDataset(
    #     root='data/images/train',
    #     annotation_file='data/annotations/train.json',
    #     transforms=None
    # )
    
    # from torch.utils.data import DataLoader
    # dataloader = DataLoader(
    #     dataset,
    #     batch_size=2,
    #     shuffle=True,
    #     num_workers=4,
    #     collate_fn=collate_fn
    # )
    
    # for images, targets in dataloader:
    #     print(f"Batch size: {len(images)}")
    #     print(f"Image shape: {images[0].shape}")
    #     print(f"Targets: {targets[0].keys()}")
    #     break
    
    print("Dataset module ready!")
