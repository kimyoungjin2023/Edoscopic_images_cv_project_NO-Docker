# medical_folder_dataset.py
import os
import json
import torch
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class MedicalFolderDataset(Dataset):
    """의료 영상 Dataset with 리사이징"""
    
    LESION_CODE_TO_IDX = {
        0: 1,  # 궤양
        1: 3,  # 종양
        2: 2,  # 암
        3: 3,  # 종양
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
                min_area=100,
                resize=None,  # ⭐ 추가: (height, width) 또는 None
                max_samples=None):  # ⭐ 추가: 최대 샘플 수
        self.image_root = image_root
        self.label_root = label_root
        self.organ_type = organ_type
        self.transforms = transforms
        self.min_area = min_area
        self.resize = resize
        
        self.samples = []
        organ_img_path = os.path.join(image_root, organ_type)
        organ_label_path = os.path.join(label_root, organ_type)
        
        # ⭐ 실제 존재하는 클래스 폴더만 확인
        available_classes = []
        if os.path.exists(organ_img_path):
            for item in os.listdir(organ_img_path):
                item_path = os.path.join(organ_img_path, item)
                if os.path.isdir(item_path):
                    available_classes.append(item)
        
        print(f"\n📁 Available class folders: {available_classes}")
        
        for class_name in available_classes:
            img_class_dir = os.path.join(organ_img_path, class_name)
            label_class_dir = os.path.join(organ_label_path, class_name)
            
            if not os.path.exists(label_class_dir):
                print(f"⚠️ Warning: Label folder not found for '{class_name}'")
                continue
            
            count = 0
            for img_name in os.listdir(img_class_dir):
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                
                img_path = os.path.join(img_class_dir, img_name)
                json_name = os.path.splitext(img_name)[0] + '.json'
                json_path = os.path.join(label_class_dir, json_name)
                
                if os.path.exists(json_path):
                    self.samples.append({
                        'image_path': img_path,
                        'json_path': json_path,
                        'class_name': class_name
                    })
                    count += 1
        
        # ⭐ 샘플 수 제한
        if max_samples is not None and len(self.samples) > max_samples:
            import random
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]
            print(f"📊 Sampled {max_samples} from total samples")
        
        print(f"\n📊 {organ_type} Dataset loaded:")
        print(f"  Total samples: {len(self.samples)}")
        
        from collections import Counter
        class_counts = Counter(s['class_name'] for s in self.samples)
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name}: {count} samples")
        
        if self.resize:
            print(f"  Resize to: {self.resize}")
    
    def __len__(self):
        return len(self.samples)
    
    def _polygon_to_mask(self, points, img_width, img_height):
        mask_img = Image.new('L', (img_width, img_height), 0)
        polygon = [tuple(p) for p in points]
        ImageDraw.Draw(mask_img).polygon(polygon, outline=1, fill=1)
        return np.array(mask_img, dtype=np.uint8)
    
    def _parse_labelme_json(self, json_path, img_width, img_height):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        boxes, labels, masks, areas = [], [], [], []
        
        for shape in data.get('shapes', []):
            points = shape['points']
            lesion_code = shape.get('lesion', 0)
            class_idx = self.LESION_CODE_TO_IDX.get(lesion_code, 1)
            
            points_array = np.array(points)
            x1, y1 = points_array[:, 0].min(), points_array[:, 1].min()
            x2, y2 = points_array[:, 0].max(), points_array[:, 1].max()
            
            area = (x2 - x1) * (y2 - y1)
            if area < self.min_area:
                continue
            
            boxes.append([x1, y1, x2, y2])
            labels.append(class_idx)
            areas.append(area)
            masks.append(self._polygon_to_mask(points, img_width, img_height))
        
        if len(boxes) == 0:
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
    
    def _resize_with_targets(self, image, target, new_size):
        """이미지와 타겟을 함께 리사이징"""
        old_width, old_height = image.size
        new_height, new_width = new_size
        
        # 이미지 리사이징
        image = image.resize((new_width, new_height), Image.BILINEAR)
        
        # 스케일 계산
        scale_x = new_width / old_width
        scale_y = new_height / old_height
        
        # Boxes 스케일링
        if len(target['boxes']) > 0:
            boxes = target['boxes'].clone()
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            target['boxes'] = boxes
            
            # Masks 리사이징
            masks = target['masks'].numpy()
            resized_masks = []
            for mask in masks:
                mask_img = Image.fromarray(mask)
                mask_img = mask_img.resize((new_width, new_height), Image.NEAREST)
                resized_masks.append(np.array(mask_img))
            target['masks'] = torch.as_tensor(np.stack(resized_masks), dtype=torch.uint8)
            
            # Area 재계산
            target['area'] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        return image, target
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('RGB')
        img_width, img_height = image.size
        
        boxes, labels, masks, areas = self._parse_labelme_json(
            sample['json_path'], img_width, img_height
        )
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([idx]),
            'area': areas,
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }
        
        # ⭐ 리사이징
        if self.resize:
            image, target = self._resize_with_targets(image, target, self.resize)
        
        if self.transforms:
            image = self.transforms(image)
        
        return image, target


def collate_fn_filter_empty(batch):
    """빈 annotation 필터링"""
    batch = [(img, target) for img, target in batch 
            if len(target['boxes']) > 0]
    
    if len(batch) == 0:
        return None
    
    return tuple(zip(*batch))