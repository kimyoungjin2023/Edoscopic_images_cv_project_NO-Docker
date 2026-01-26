# datasets/medical_folder_dataset.py
import os
import json
import torch
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

class MedicalFolderDataset(Dataset):
    """
    폴더 구조 기반 의료 영상 Dataset (LabelMe JSON)
    
    구조:
    1.원천데이터/
        └── 대장/
        ├── 궤양/
        ├── 암/
        └── 종양/
    
    2.라벨링데이터/
        └── 대장/
        ├── 궤양/
        │   ├── image1.json (LabelMe format)
        ├── 암/
        └── 종양/
    """
    
    # JSON의 lesion 코드 → 우리 클래스 ID
    LESION_CODE_TO_IDX = {
        0: 1,  # 궤양 → 1
        1: 3,  # 종양 → 3 (추정)
        2: 2,  # 암 → 2
        3: 3,  # 종양 → 3 (예비)
    }
    
    IDX_TO_CLASS = {
        1: '궤양 (ulcer)',
        2: '암 (cancer)',
        3: '종양 (tumor)'
    }
    
    def __init__(self, 
                image_root, 
                label_root, 
                organ_type='대장',
                transforms=None,
                min_area=100):  # 너무 작은 annotation 필터링
        """
        Args:
            image_root: 원천데이터 폴더 (1.원천데이터)
            label_root: 라벨링데이터 폴더 (2.라벨링데이터)
            organ_type: '대장' or '위'
            transforms: torchvision transforms
            min_area: 최소 annotation 면적 (픽셀)
        """
        self.image_root = image_root
        self.label_root = label_root
        self.organ_type = organ_type
        self.transforms = transforms
        self.min_area = min_area
        
        # 샘플 수집
        self.samples = []
        organ_img_path = os.path.join(image_root, organ_type)
        organ_label_path = os.path.join(label_root, organ_type)
        
        class_names = ['궤양', '암', '종양']
        
        for class_name in class_names:
            img_class_dir = os.path.join(organ_img_path, class_name)
            label_class_dir = os.path.join(organ_label_path, class_name)
            
            if not os.path.isdir(img_class_dir):
                continue
            
            # 이미지 파일 순회
            for img_name in os.listdir(img_class_dir):
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                
                img_path = os.path.join(img_class_dir, img_name)
                
                # 대응하는 JSON 파일 찾기
                json_name = os.path.splitext(img_name)[0] + '.json'
                json_path = os.path.join(label_class_dir, json_name)
                
                if os.path.exists(json_path):
                    self.samples.append({
                        'image_path': img_path,
                        'json_path': json_path,
                        'class_name': class_name
                    })
        
        print(f"\n📊 {organ_type} Dataset loaded:")
        print(f"  Total samples: {len(self.samples)}")
        
        # 클래스별 개수
        from collections import Counter
        class_counts = Counter(s['class_name'] for s in self.samples)
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name}: {count} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def _polygon_to_mask(self, points, img_width, img_height):
        """Polygon 좌표를 binary mask로 변환"""
        mask_img = Image.new('L', (img_width, img_height), 0)
        
        # points를 tuple list로 변환
        polygon = [tuple(p) for p in points]
        
        # Polygon 그리기
        ImageDraw.Draw(mask_img).polygon(polygon, outline=1, fill=1)
        
        return np.array(mask_img, dtype=np.uint8)
    
    def _parse_labelme_json(self, json_path, img_width, img_height):
        """
        LabelMe JSON을 Mask R-CNN 형식으로 변환
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        boxes = []
        labels = []
        masks = []
        areas = []
        
        shapes = data.get('shapes', [])
        
        for shape in shapes:
            points = shape['points']
            lesion_code = shape.get('lesion', 0)
            
            # Lesion code를 class index로 변환
            class_idx = self.LESION_CODE_TO_IDX.get(lesion_code, 1)
            
            # Points를 numpy array로
            points_array = np.array(points)
            
            # Bounding box 계산
            x_coords = points_array[:, 0]
            y_coords = points_array[:, 1]
            
            x1, y1 = x_coords.min(), y_coords.min()
            x2, y2 = x_coords.max(), y_coords.max()
            
            # 너무 작은 bbox 필터링
            area = (x2 - x1) * (y2 - y1)
            if area < self.min_area:
                continue
            
            boxes.append([x1, y1, x2, y2])
            labels.append(class_idx)
            areas.append(area)
            
            # Polygon → Mask
            mask = self._polygon_to_mask(points, img_width, img_height)
            masks.append(mask)
        
        # Tensor 변환
        if len(boxes) == 0:
            # 빈 annotation인 경우 (학습 시 문제가 될 수 있음)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, img_height, img_width), dtype=torch.uint8)
            areas = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
            areas = torch.as_tensor(areas, dtype=torch.float32)
        
        return boxes, labels, masks, areas
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 이미지 로드
        image = Image.open(sample['image_path']).convert('RGB')
        img_width, img_height = image.size
        
        # Annotation 파싱
        boxes, labels, masks, areas = self._parse_labelme_json(
            sample['json_path'],
            img_width,
            img_height
        )
        
        # Target 구성
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([idx]),
            'area': areas,
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }
        
        if self.transforms:
            image = self.transforms(image)
        
        return image, target


# 빈 annotation 필터링하는 collate function
def collate_fn_filter_empty(batch):
    """빈 annotation을 가진 샘플 제거"""
    batch = [(img, target) for img, target in batch 
            if len(target['boxes']) > 0]
    
    if len(batch) == 0:
        # 모든 샘플이 비어있으면 더미 반환
        return None
    
    return tuple(zip(*batch))