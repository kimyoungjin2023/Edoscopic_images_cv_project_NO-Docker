"""
AI Hub 내시경 이미지 합성 데이터셋 로더
================================================================
데이터셋: AI Hub 내시경 이미지 합성데이터 (위/대장 내시경)
- 총 40,000장 (위 20,000장, 대장 20,000장)
- 클래스: 궤양(Ulcer), 용종(Polyp), 암(Cancer)
- 형식: COCO-style annotation (bbox + segmentation mask)

데이터 분할:
- Train: 클래스별 1000장 (총 6,000장)
- Validation: 클래스별 150장 (총 900장)
- Test: 클래스별 250~500장 (총 1,500~3,000장)
================================================================
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import cv2
from collections import defaultdict
import random
from pathlib import Path


class AIHubEndoscopicDataset(Dataset):
    """
    AI Hub 내시경 이미지 데이터셋
    
    디렉토리 구조:
    data_root/
        01.원천데이터/
            1.위/
                1.궤양/  (5,000장)
                2.용종/  (5,000장)
                3.암/    (10,000장)
            2.대장/
                1.궤양/  (5,000장)
                2.용종/  (5,000장)
                3.암/    (10,000장)
        02.라벨링데이터/
            1.위/
                1.궤양/
                2.용종/
                3.암/
            2.대장/
                1.궤양/
                2.용종/
                3.암/
    
    JSON Annotation 구조:
    {
        "version": "4.2.7",
        "shapes": [
            {
                "label": "01_stomach_ulcer_generation",
                "organ": 0,      # 0: 위, 1: 대장
                "lesion": 0,     # 0: 궤양, 1: 용종, 2: 암
                "location": 1,
                "points": [[x1, y1], [x2, y2], ...],
                "shape_type": "polygon" or "rectangle"
            }
        ],
        "imagePath": "1_1_03827.png",
        "imageHeight": 2048,
        "imageWidth": 2048
    }
    """
    
    # 클래스 정의 (배경 포함)
    CLASSES = {
        0: 'background',
        1: 'stomach_ulcer',    # 위 궤양
        2: 'stomach_polyp',    # 위 용종
        3: 'stomach_cancer',   # 위 암
        4: 'colon_ulcer',      # 대장 궤양
        5: 'colon_polyp',      # 대장 용종
        6: 'colon_cancer'      # 대장 암
    }
    
    NUM_CLASSES = 7  # 배경 포함
    
    def __init__(
        self,
        root_dir,
        split='train',
        transforms=None,
        train_samples_per_class=1000,
        val_samples_per_class=150,
        test_samples_per_class=400,
        seed=42
    ):
        """
        Args:
            root_dir (str): 데이터 루트 디렉토리
            split (str): 'train', 'val', 'test'
            transforms: albumentations transforms
            train_samples_per_class (int): 클래스별 학습 샘플 수
            val_samples_per_class (int): 클래스별 검증 샘플 수
            test_samples_per_class (int): 클래스별 테스트 샘플 수
            seed (int): random seed for reproducibility
        """
        assert split in ['train', 'val', 'test'], f"Invalid split: {split}"
        
        self.root_dir = Path(root_dir)
        self.split = split
        self.transforms = transforms
        
        # 데이터 경로
        self.source_dir = self.root_dir / '01.원천데이터'
        self.label_dir = self.root_dir / '02.라벨링데이터'
        
        # 데이터 분할 파라미터
        self.train_samples_per_class = train_samples_per_class
        self.val_samples_per_class = val_samples_per_class
        self.test_samples_per_class = test_samples_per_class
        
        # Reproducibility
        random.seed(seed)
        np.random.seed(seed)
        
        # 데이터 로드 및 분할
        print(f"\n{'='*60}")
        print(f"Loading AI Hub Endoscopic Dataset - {split.upper()} split")
        print(f"{'='*60}")
        self.samples = self._load_and_split_data()
        
        print(f"\n{split.upper()} Dataset Summary:")
        print(f"  Total images: {len(self.samples)}")
        self._print_class_distribution()
        print(f"{'='*60}\n")
    
    def _load_and_split_data(self):
        """데이터 로드 및 train/val/test 분할"""
        all_samples_by_class = defaultdict(list)
        
        # 데이터 수집
        organs = [('1.위', 0), ('2.대장', 1)]
        lesions = [('1.궤양', 0), ('2.용종', 1), ('3.암', 2)]
        
        for organ_name, organ_id in organs:
            for lesion_name, lesion_id in lesions:
                # 클래스 ID 계산 (1-6)
                class_id = organ_id * 3 + lesion_id + 1
                
                print(f"Loading {organ_name}/{lesion_name} (class_id={class_id})...")
                samples = self._collect_samples(organ_name, lesion_name, organ_id, lesion_id)
                all_samples_by_class[class_id].extend(samples)
                print(f"  Found {len(samples)} samples")
        
        # 데이터 분할
        print(f"\nSplitting data for {self.split}...")
        split_samples = []
        
        for class_id in sorted(all_samples_by_class.keys()):
            class_samples = all_samples_by_class[class_id]
            
            # 셔플 (reproducibility 보장)
            random.shuffle(class_samples)
            
            # 분할 인덱스 계산
            train_end = self.train_samples_per_class
            val_end = train_end + self.val_samples_per_class
            test_end = val_end + self.test_samples_per_class
            
            # 분할
            if self.split == 'train':
                selected = class_samples[:train_end]
            elif self.split == 'val':
                selected = class_samples[train_end:val_end]
            elif self.split == 'test':
                selected = class_samples[val_end:test_end]
            
            split_samples.extend(selected)
            class_name = self.CLASSES[class_id]
            print(f"  {class_name}: {len(selected)} samples")
        
        # 최종 셔플
        random.shuffle(split_samples)
        
        return split_samples
    
    def _collect_samples(self, organ_name, lesion_name, organ_id, lesion_id):
        """특정 장기/병변 조합의 샘플 수집"""
        samples = []
        
        # 경로 설정
        img_dir = self.source_dir / organ_name / lesion_name
        label_dir = self.label_dir / organ_name / lesion_name
        
        if not img_dir.exists() or not label_dir.exists():
            print(f"  Warning: Directory not found - {img_dir}")
            return samples
        
        # JSON 파일 순회
        json_files = list(label_dir.glob('*.json'))
        
        for json_path in json_files:
            try:
                # JSON 로드
                with open(json_path, 'r', encoding='utf-8') as f:
                    ann_data = json.load(f)
                
                # 이미지 경로 확인
                img_name = ann_data['imagePath']
                img_path = img_dir / img_name
                
                if not img_path.exists():
                    continue
                
                # 유효한 annotation이 있는지 확인
                if len(ann_data['shapes']) == 0:
                    continue
                
                samples.append({
                    'image_path': str(img_path),
                    'annotation': ann_data,
                    'organ_id': organ_id,
                    'lesion_id': lesion_id,
                    'class_id': organ_id * 3 + lesion_id + 1
                })
                
            except Exception as e:
                print(f"  Error loading {json_path.name}: {e}")
                continue
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: (C, H, W) tensor
            target: dict with keys 'boxes', 'labels', 'masks', 'image_id', 'area', 'iscrowd'
        """
        sample = self.samples[idx]
        
        # 이미지 로드
        image = Image.open(sample['image_path']).convert('RGB')
        image = np.array(image)
        
        # Annotation 파싱
        ann_data = sample['annotation']
        img_h, img_w = ann_data['imageHeight'], ann_data['imageWidth']
        
        boxes = []
        labels = []
        masks = []
        
        for shape in ann_data['shapes']:
            try:
                organ_id = shape['organ']
                lesion_id = shape['lesion']
                class_id = organ_id * 3 + lesion_id + 1
                
                # 좌표 추출
                points = np.array(shape['points'], dtype=np.float32)
                
                if len(points) < 3:  # 최소 3개 점 필요
                    continue
                
                # Bounding Box 계산
                x_coords = points[:, 0]
                y_coords = points[:, 1]
                x1, y1 = float(x_coords.min()), float(y_coords.min())
                x2, y2 = float(x_coords.max()), float(y_coords.max())
                
                # 유효성 검사
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Mask 생성
                mask = np.zeros((img_h, img_w), dtype=np.uint8)
                
                if shape['shape_type'] == 'polygon':
                    pts = points.astype(np.int32)
                    cv2.fillPoly(mask, [pts], 1)
                elif shape['shape_type'] == 'rectangle':
                    cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 1, -1)
                
                boxes.append([x1, y1, x2, y2])
                labels.append(class_id)
                masks.append(mask)
                
            except Exception as e:
                print(f"Error parsing shape: {e}")
                continue
        
        # Transform 적용
        if self.transforms is not None and len(boxes) > 0:
            try:
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
            except Exception as e:
                print(f"Transform error: {e}")
        
        # Tensor 변환
        image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        num_objs = len(boxes)
        if num_objs > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
            areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, image.shape[1], image.shape[2]), dtype=torch.uint8)
            areas = torch.zeros((0,), dtype=torch.float32)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([idx]),
            'area': areas,
            'iscrowd': torch.zeros((num_objs,), dtype=torch.int64)
        }
        
        return image, target
    
    def _print_class_distribution(self):
        """클래스별 이미지/객체 분포 출력"""
        image_counts = defaultdict(int)
        object_counts = defaultdict(int)
        
        for sample in self.samples:
            class_id = sample['class_id']
            image_counts[class_id] += 1
            
            ann_data = sample['annotation']
            for shape in ann_data['shapes']:
                organ_id = shape['organ']
                lesion_id = shape['lesion']
                obj_class_id = organ_id * 3 + lesion_id + 1
                object_counts[obj_class_id] += 1
        
        print(f"\n  Class Distribution:")
        print(f"  {'Class':<20} {'Images':<10} {'Objects':<10}")
        print(f"  {'-'*40}")
        
        for class_id in sorted(image_counts.keys()):
            class_name = self.CLASSES[class_id]
            img_count = image_counts[class_id]
            obj_count = object_counts[class_id]
            print(f"  {class_name:<20} {img_count:<10} {obj_count:<10}")


def collate_fn(batch):
    """
    Custom collate function for DataLoader
    Mask R-CNN은 배치마다 이미지 크기가 다를 수 있으므로 리스트로 반환
    """
    return tuple(zip(*batch))


if __name__ == "__main__":
    print("="*60)
    print("AI Hub Endoscopic Dataset Module Test")
    print("="*60)
    
    # 사용 예시
    print("\nUsage Example:")
    print("""
    from datasets.dataset import AIHubEndoscopicDataset, collate_fn
    from torch.utils.data import DataLoader
    
    # 데이터셋 생성
    train_dataset = AIHubEndoscopicDataset(
        root_dir='./data',
        split='train',
        train_samples_per_class=1000,
        val_samples_per_class=150,
        test_samples_per_class=400
    )
    
    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    # 사용
    for images, targets in train_loader:
        # Training loop
        pass
    """)
    
    print(f"\nAvailable classes:")
    for class_id, class_name in AIHubEndoscopicDataset.CLASSES.items():
        print(f"  {class_id}: {class_name}")
    
    print(f"\nTotal number of classes: {AIHubEndoscopicDataset.NUM_CLASSES}")
