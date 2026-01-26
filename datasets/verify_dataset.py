# verify_dataset.py
import torch
from torchvision import transforms
from datasets.medical_folder_dataset import MedicalFolderDataset
import matplotlib.pyplot as plt
import numpy as np

def visualize_sample(dataset, idx):
    """샘플 시각화"""
    image, target = dataset[idx]
    
    # Tensor → numpy
    if isinstance(image, torch.Tensor):
        image_np = image.permute(1, 2, 0).numpy()
    else:
        image_np = np.array(image) / 255.0
    
    boxes = target['boxes'].numpy()
    masks = target['masks'].numpy()
    labels = target['labels'].numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 원본 이미지
    axes[0].imshow(image_np)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Bounding boxes
    axes[1].imshow(image_np)
    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = box
        rect = plt.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            fill=False, edgecolor='red', linewidth=2
        )
        axes[1].add_patch(rect)
        
        # 라벨 표시
        class_name = dataset.IDX_TO_CLASS[label]
        axes[1].text(
            x1, y1-5, class_name,
            color='red', fontsize=10, weight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
    axes[1].set_title(f"Boxes (n={len(boxes)})")
    axes[1].axis('off')
    
    # Masks
    axes[2].imshow(image_np)
    combined_mask = masks.sum(axis=0)
    axes[2].imshow(combined_mask, alpha=0.5, cmap='jet')
    axes[2].set_title("Masks")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"outputs/verify_sample_{idx}.png", dpi=150, bbox_inches='tight')
    print(f"✅ Saved: outputs/verify_sample_{idx}.png")
    plt.show()


def main():
    import os
    os.makedirs("outputs", exist_ok=True)
    
    transform = transforms.ToTensor()
    
    # 대장 데이터셋 로드
    print("="*60)
    print("Loading COLON dataset...")
    print("="*60)
    
    dataset = MedicalFolderDataset(
        image_root='/Users/admin/Downloads/datasets/1.Training/1.원천데이터',
        label_root='/Users/admin/Downloads/datasets/1.Training/2.라벨링데이터',
        organ_type='대장',
        transforms=transform
    )
    
    print(f"\n📊 Dataset statistics:")
    print(f"  Total samples: {len(dataset)}")
    
    # 처음 3개 샘플 확인
    print("\n" + "="*60)
    print("Checking first 3 samples...")
    print("="*60)
    
    for i in range(min(3, len(dataset))):
        print(f"\n--- Sample {i} ---")
        image, target = dataset[i]
        
        print(f"Image shape: {image.shape}")
        print(f"Boxes: {len(target['boxes'])}")
        print(f"Labels: {target['labels'].tolist()}")
        print(f"Label names: {[dataset.IDX_TO_CLASS[l.item()] for l in target['labels']]}")
        print(f"Masks shape: {target['masks'].shape}")
        
        # 시각화
        visualize_sample(dataset, i)
    
    # 위 데이터셋도 확인
    print("\n" + "="*60)
    print("Loading STOMACH dataset...")
    print("="*60)
    
    stomach_dataset = MedicalFolderDataset(
        image_root='/Users/admin/Downloads/datasets/1.Training/1.원천데이터',
        label_root='/Users/admin/Downloads/datasets/1.Training/2.라벨링데이터',
        organ_type='위',
        transforms=transform
    )
    
    print(f"\n📊 Stomach dataset: {len(stomach_dataset)} samples")
    
    # 샘플 하나 확인
    if len(stomach_dataset) > 0:
        visualize_sample(stomach_dataset, 0)
    
    print("\n✅ Dataset verification complete!")


if __name__ == "__main__":
    main()