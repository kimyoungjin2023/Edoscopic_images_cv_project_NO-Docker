"""
Data Augmentation and Transforms
- Albumentations 라이브러리 사용
- Train/Val/Test용 Transform 분리
- Mask R-CNN에 적합한 bbox, mask transform 포함
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def get_train_transforms(img_size=512):
    """
    학습용 Transform
    - 다양한 augmentation 적용
    - bbox와 mask에도 동일하게 적용
    """
    return A.Compose([
        # Resize
        A.Resize(img_size, img_size),
        
        # Geometric transforms
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=30,
            p=0.5,
            border_mode=cv2.BORDER_CONSTANT
        ),
        
        # Color transforms
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0
            ),
            A.RandomGamma(p=1.0),
        ], p=0.5),
        
        # Noise and blur
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.3),
        
        # 의료 영상 특화 augmentation
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
        
        # Cutout (일부 영역 가리기)
        A.CoarseDropout(
            max_holes=8,
            max_height=32,
            max_width=32,
            min_holes=1,
            fill_value=0,
            p=0.3
        ),
        
        # Normalization은 모델에서 처리 (ImageNet 기준)
        # A.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225],
        # ),
    ], bbox_params=A.BboxParams(
        format='coco',  # [x, y, width, height]
        label_fields=['labels'],
        min_visibility=0.3,
        min_area=100
    ))


def get_val_transforms(img_size=512):
    """
    검증/테스트용 Transform
    - Augmentation 없이 기본 전처리만
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        # A.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225],
        # ),
    ], bbox_params=A.BboxParams(
        format='coco',
        label_fields=['labels'],
        min_visibility=0.0,
        min_area=0
    ))


def get_test_transforms(img_size=512):
    """
    추론용 Transform (라벨 불필요)
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        # A.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225],
        # ),
    ])


class Denormalize:
    """
    시각화를 위한 역정규화 클래스
    """
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        """
        Args:
            tensor: (C, H, W) 형태의 normalized tensor
        Returns:
            denormalized tensor
        """
        import torch
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.clone()
        
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        
        return tensor


# Custom transforms for Mask R-CNN specific preprocessing
class MaskRCNNTransform:
    """
    Mask R-CNN에 특화된 전처리
    """
    def __init__(self, min_size=800, max_size=1333, image_mean=None, image_std=None):
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        
        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std
    
    def __call__(self, image, target=None):
        import torch
        import torch.nn.functional as F
        
        # Normalize
        image = image.float()
        image = image / 255.0
        mean = torch.tensor(self.image_mean, dtype=torch.float32)
        std = torch.tensor(self.image_std, dtype=torch.float32)
        image = (image - mean[:, None, None]) / std[:, None, None]
        
        # Resize
        h, w = image.shape[-2:]
        size = min(h, w)
        scale_factor = self.min_size / size
        
        if h > w:
            newh, neww = self.min_size, int(w * scale_factor)
        else:
            newh, neww = int(h * scale_factor), self.min_size
        
        # Max size constraint
        if max(newh, neww) > self.max_size:
            scale_factor = self.max_size / max(newh, neww)
            newh = int(newh * scale_factor)
            neww = int(neww * scale_factor)
        
        image = F.interpolate(
            image.unsqueeze(0),
            size=(newh, neww),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        if target is not None:
            # Resize boxes
            target['boxes'][:, [0, 2]] *= neww / w
            target['boxes'][:, [1, 3]] *= newh / h
            
            # Resize masks
            if 'masks' in target:
                masks = target['masks']
                masks = F.interpolate(
                    masks.unsqueeze(0).float(),
                    size=(newh, neww),
                    mode='nearest'
                ).squeeze(0).byte()
                target['masks'] = masks
        
        return image, target


if __name__ == "__main__":
    # Transform 테스트
    import numpy as np
    from PIL import Image
    
    print("Testing transforms...")
    
    # Dummy data
    dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    dummy_boxes = [[100, 100, 200, 200], [300, 300, 400, 400]]
    dummy_labels = [1, 2]
    dummy_masks = [
        np.zeros((512, 512), dtype=np.uint8),
        np.zeros((512, 512), dtype=np.uint8)
    ]
    dummy_masks[0][100:200, 100:200] = 1
    dummy_masks[1][300:400, 300:400] = 1
    
    # Train transform
    train_transform = get_train_transforms(img_size=512)
    transformed = train_transform(
        image=dummy_image,
        masks=dummy_masks,
        bboxes=dummy_boxes,
        labels=dummy_labels
    )
    
    print(f"Original image shape: {dummy_image.shape}")
    print(f"Transformed image shape: {transformed['image'].shape}")
    print(f"Number of boxes: {len(transformed['bboxes'])}")
    print(f"Number of masks: {len(transformed['masks'])}")
    
    # Val transform
    val_transform = get_val_transforms(img_size=512)
    transformed_val = val_transform(
        image=dummy_image,
        masks=dummy_masks,
        bboxes=dummy_boxes,
        labels=dummy_labels
    )
    
    print(f"\nValidation transform applied successfully!")
    print("All transforms are ready!")
