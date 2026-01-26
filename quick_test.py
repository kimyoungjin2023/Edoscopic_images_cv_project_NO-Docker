# quick_test.py
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# medical_folder_dataset.py의 코드를 여기에 직접 포함
import json
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

class MedicalFolderDataset(Dataset):
    """의료 영상 Dataset"""
    
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
    
    def __init__(self, image_root, label_root, organ_type='대장', transforms=None, min_area=100):
        self.image_root = image_root
        self.label_root = label_root
        self.organ_type = organ_type
        self.transforms = transforms
        self.min_area = min_area
        
        self.samples = []
        organ_img_path = os.path.join(image_root, organ_type)
        organ_label_path = os.path.join(label_root, organ_type)
        
        class_names = ['궤양', '암', '종양']
        
        for class_name in class_names:
            img_class_dir = os.path.join(organ_img_path, class_name)
            label_class_dir = os.path.join(organ_label_path, class_name)
            
            if not os.path.isdir(img_class_dir):
                continue
            
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
        
        print(f"\n📊 {organ_type} Dataset: {len(self.samples)} samples")
        from collections import Counter
        class_counts = Counter(s['class_name'] for s in self.samples)
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name}: {count}")
    
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
        
        if self.transforms:
            image = self.transforms(image)
        
        return image, target


# 테스트 실행
os.makedirs("outputs", exist_ok=True)
transform = transforms.ToTensor()

print("="*60)
print("QUICK TEST")
print("="*60)

dataset = MedicalFolderDataset(
    image_root='/Users/admin/Downloads/datasets/1.Training/1.원천데이터',
    label_root='/Users/admin/Downloads/datasets/1.Training/2.라벨링데이터',
    organ_type='대장',
    transforms=transform
)

# 첫 샘플 테스트
if len(dataset) > 0:
    image, target = dataset[0]
    print(f"\n✅ First sample loaded!")
    print(f"  Image: {image.shape}")
    print(f"  Boxes: {len(target['boxes'])}")
    print(f"  Labels: {target['labels'].tolist()}")
else:
    print("❌ No samples found!")