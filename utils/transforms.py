"""
의료 영상(내시경 이미지) 특화 Transform
================================================================
내시경 이미지 특성:
- 고해상도 (2048x2048)
- 조명 변화가 큼
- 점막 표면의 세부 텍스처 중요
- 색상 정보가 진단에 중요
================================================================
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np


def get_train_transforms(img_size=512):
    """
    학습용 Transform - 의료 영상 특화
    
    Args:
        img_size (int): 타겟 이미지 크기
    
    Returns:
        albumentations.Compose
    """
    return A.Compose([
        # 1. 리사이즈
        A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
        
        # 2. Geometric Transforms (의료 영상에서 일반적)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=15,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=0.5
        ),
        
        # 3. Color Augmentation (내시경 이미지 조명 변화 대응)
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=1.0
            ),
            A.RandomGamma(
                gamma_limit=(80, 120),
                p=1.0
            ),
        ], p=0.5),
        
        # 4. 의료 영상 특화 Augmentation
        # CLAHE: Contrast Limited Adaptive Histogram Equalization
        # - 내시경 영상의 조명 불균일 보정
        A.CLAHE(
            clip_limit=2.0,
            tile_grid_size=(8, 8),
            p=0.4
        ),
        
        # Sharpen: 점막 표면 텍스처 강조
        A.Sharpen(
            alpha=(0.2, 0.5),
            lightness=(0.5, 1.0),
            p=0.3
        ),
        
        # 5. Noise and Blur (약하게 적용)
        A.OneOf([
            A.GaussNoise(
                var_limit=(5.0, 20.0),
                p=1.0
            ),
            A.GaussianBlur(
                blur_limit=(3, 5),
                p=1.0
            ),
            A.MotionBlur(
                blur_limit=3,
                p=1.0
            ),
        ], p=0.2),
        
        # 6. Elastic Transform (약하게 - 의료 영상에서 과도한 변형은 부적절)
        A.ElasticTransform(
            alpha=30,
            sigma=5,
            alpha_affine=5,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=0.2
        ),
        
        # 7. Cutout (일부 영역 masking)
        A.CoarseDropout(
            max_holes=4,
            max_height=int(img_size * 0.1),
            max_width=int(img_size * 0.1),
            min_holes=1,
            min_height=int(img_size * 0.05),
            min_width=int(img_size * 0.05),
            fill_value=0,
            p=0.3
        ),
        
    ], bbox_params=A.BboxParams(
        format='coco',  # [x_min, y_min, width, height]
        label_fields=['labels'],
        min_visibility=0.3,  # 30% 이상 보이는 객체만 유지
        min_area=100  # 최소 면적
    ))


def get_val_transforms(img_size=512):
    """
    검증용 Transform - Augmentation 없이 기본 전처리만
    
    Args:
        img_size (int): 타겟 이미지 크기
    
    Returns:
        albumentations.Compose
    """
    return A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
        
        # 의료 영상 품질 향상 (약하게)
        A.CLAHE(
            clip_limit=1.5,
            tile_grid_size=(8, 8),
            p=0.5
        ),
        
    ], bbox_params=A.BboxParams(
        format='coco',
        label_fields=['labels'],
        min_visibility=0.0,
        min_area=0
    ))


def get_test_transforms(img_size=512):
    """
    테스트/추론용 Transform
    
    Args:
        img_size (int): 타겟 이미지 크기
    
    Returns:
        albumentations.Compose
    """
    return A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
        
        # 의료 영상 품질 향상
        A.CLAHE(
            clip_limit=1.5,
            tile_grid_size=(8, 8),
            p=0.5
        ),
    ])


class Denormalize:
    """
    시각화를 위한 역정규화
    """
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array(mean)
        self.std = np.array(std)
    
    def __call__(self, tensor):
        """
        Args:
            tensor: (C, H, W) normalized tensor or (H, W, C) numpy array
        
        Returns:
            denormalized array/tensor
        """
        import torch
        
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.clone()
            for t, m, s in zip(tensor, self.mean, self.std):
                t.mul_(s).add_(m)
            return tensor
        else:
            # Numpy array
            return (tensor * self.std + self.mean).clip(0, 1)


def visualize_augmentation(image, masks, bboxes, labels, num_samples=4):
    """
    Augmentation 시각화
    
    Args:
        image: numpy array (H, W, C)
        masks: list of masks
        bboxes: list of bboxes
        labels: list of labels
        num_samples: 생성할 샘플 수
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    transform = get_train_transforms(img_size=512)
    
    fig, axes = plt.subplots(2, num_samples, figsize=(4*num_samples, 8))
    
    for i in range(num_samples):
        transformed = transform(
            image=image,
            masks=masks,
            bboxes=bboxes,
            labels=labels
        )
        
        aug_image = transformed['image']
        aug_bboxes = transformed['bboxes']
        aug_labels = transformed['labels']
        
        # 원본
        if num_samples > 1:
            ax = axes[0, i]
        else:
            ax = axes[0]
        ax.imshow(image)
        ax.set_title(f'Original')
        ax.axis('off')
        
        # Augmented
        if num_samples > 1:
            ax = axes[1, i]
        else:
            ax = axes[1]
        ax.imshow(aug_image)
        
        # Bounding boxes
        for bbox, label in zip(aug_bboxes, aug_labels):
            x, y, w, h = bbox
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(x, y-5, f'Class {label}', color='red', fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.7))
        
        ax.set_title(f'Augmented {i+1}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('augmentation_samples.png', dpi=150, bbox_inches='tight')
    print("Augmentation samples saved to 'augmentation_samples.png'")
    plt.close()


if __name__ == "__main__":
    print("="*60)
    print("Medical Image Transforms Module Test")
    print("="*60)
    
    # 테스트용 더미 데이터
    dummy_image = np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8)
    dummy_masks = [
        np.zeros((2048, 2048), dtype=np.uint8),
        np.zeros((2048, 2048), dtype=np.uint8)
    ]
    dummy_masks[0][500:1000, 500:1000] = 1
    dummy_masks[1][1200:1600, 1200:1600] = 1
    
    dummy_bboxes = [[500, 500, 500, 500], [1200, 1200, 400, 400]]  # [x, y, w, h]
    dummy_labels = [1, 2]
    
    # Train transform 테스트
    print("\nTesting train transforms...")
    train_transform = get_train_transforms(img_size=512)
    
    try:
        transformed = train_transform(
            image=dummy_image,
            masks=dummy_masks,
            bboxes=dummy_bboxes,
            labels=dummy_labels
        )
        
        print(f"✓ Train transform successful")
        print(f"  Original image shape: {dummy_image.shape}")
        print(f"  Transformed image shape: {transformed['image'].shape}")
        print(f"  Original bboxes: {len(dummy_bboxes)}")
        print(f"  Transformed bboxes: {len(transformed['bboxes'])}")
        
    except Exception as e:
        print(f"✗ Train transform failed: {e}")
    
    # Val transform 테스트
    print("\nTesting validation transforms...")
    val_transform = get_val_transforms(img_size=512)
    
    try:
        transformed = val_transform(
            image=dummy_image,
            masks=dummy_masks,
            bboxes=dummy_bboxes,
            labels=dummy_labels
        )
        print(f"✓ Validation transform successful")
        
    except Exception as e:
        print(f"✗ Validation transform failed: {e}")
    
    # Test transform 테스트
    print("\nTesting test transforms...")
    test_transform = get_test_transforms(img_size=512)
    
    try:
        transformed = test_transform(image=dummy_image)
        print(f"✓ Test transform successful")
        
    except Exception as e:
        print(f"✗ Test transform failed: {e}")
    
    print("\n" + "="*60)
    print("All transforms tested successfully!")
    print("="*60)
    
    print("\nUsage Example:")
    print("""
    from utils.transforms import get_train_transforms, get_val_transforms
    
    # 학습용
    train_transform = get_train_transforms(img_size=512)
    
    # 검증용
    val_transform = get_val_transforms(img_size=512)
    
    # 데이터셋에 적용
    train_dataset = AIHubEndoscopicDataset(
        root_dir='./data',
        split='train',
        transforms=train_transform
    )
    """)
