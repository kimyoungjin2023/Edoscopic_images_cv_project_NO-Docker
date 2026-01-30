"""
Inference Script for Mask R-CNN
================================================================
단일 이미지 추론:
- 이미지 입력
- Detection (bbox + class label + confidence)
- Segmentation (mask)
- 시각화 및 저장
================================================================
"""

import os
import sys
import argparse
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

from models.mask_rcnn import get_maskrcnn_model
from datasets.dataset import AIHubEndoscopicDataset
from utils.transforms import get_test_transforms


# 시각화 색상 (클래스별)
COLORS = [
    [1.000, 0.000, 0.000],  # Red
    [0.000, 1.000, 0.000],  # Green
    [0.000, 0.000, 1.000],  # Blue
    [1.000, 1.000, 0.000],  # Yellow
    [1.000, 0.000, 1.000],  # Magenta
    [0.000, 1.000, 1.000],  # Cyan
    [0.500, 0.500, 0.500],  # Gray
]


def parse_args():
    """커맨드라인 인자 파싱"""
    parser = argparse.ArgumentParser(
        description='Inference with Mask R-CNN on Endoscopic Images'
    )
    
    # 입력
    parser.add_argument('--input', type=str, required=True,
                        help='입력 이미지 경로 또는 폴더')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='모델 체크포인트 경로')
    
    # 모델
    parser.add_argument('--num-classes', type=int, default=7,
                        help='클래스 개수')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--img-size', type=int, default=512,
                        help='입력 이미지 크기')
    parser.add_argument('--device', type=str, default='cuda',
                        help='디바이스')
    
    # 출력
    parser.add_argument('--output-dir', type=str, default='./inference_results',
                        help='결과 저장 디렉토리')
    parser.add_argument('--save-overlay', action='store_true', default=True,
                        help='오버레이 이미지 저장')
    parser.add_argument('--save-detailed', action='store_true', default=True,
                        help='상세 시각화 저장')
    
    return parser.parse_args()


def load_image(image_path, img_size=512):
    """
    이미지 로드 및 전처리
    
    Returns:
        image_tensor: (1, 3, H, W)
        original_image: (H, W, 3) numpy array
        original_size: (H, W)
    """
    # 원본 이미지
    original = Image.open(image_path).convert('RGB')
    original_size = original.size[::-1]  # (H, W)
    original = np.array(original)
    
    # Transform
    transform = get_test_transforms(img_size=img_size)
    transformed = transform(image=original)
    image = transformed['image']
    
    # Tensor 변환
    image_tensor = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    
    return image_tensor, original, original_size


@torch.no_grad()
def predict(model, image_tensor, device, conf_threshold=0.5):
    """
    추론 수행
    
    Returns:
        prediction: dict with 'boxes', 'labels', 'scores', 'masks'
    """
    model.eval()
    
    image_tensor = image_tensor.to(device)
    predictions = model(image_tensor)
    
    pred = predictions[0]
    
    # Confidence threshold 적용
    keep = pred['scores'] >= conf_threshold
    
    result = {
        'boxes': pred['boxes'][keep].cpu().numpy(),
        'labels': pred['labels'][keep].cpu().numpy(),
        'scores': pred['scores'][keep].cpu().numpy(),
        'masks': pred['masks'][keep].cpu().numpy()
    }
    
    return result


def visualize_detailed(image, prediction, class_names, save_path):
    """
    상세 시각화 (원본 + Detection + Segmentation)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. 원본 이미지
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # 2. Detection (BBox + Labels)
    axes[1].imshow(image)
    
    boxes = prediction['boxes']
    labels = prediction['labels']
    scores = prediction['scores']
    
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        x1, y1, x2, y2 = box
        color = COLORS[int(label) % len(COLORS)]
        
        # BBox
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2,
            edgecolor=color,
            facecolor='none'
        )
        axes[1].add_patch(rect)
        
        # Label
        class_name = class_names.get(int(label), f'Class {label}')
        label_text = f'{class_name}: {score:.2f}'
        
        axes[1].text(
            x1, y1 - 5,
            label_text,
            color='white',
            fontsize=10,
            fontweight='bold',
            bbox=dict(facecolor=color, alpha=0.8, edgecolor='none', pad=2)
        )
    
    axes[1].set_title(f'Detection ({len(boxes)} objects)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # 3. Segmentation (Masks)
    axes[2].imshow(image)
    
    masks = prediction['masks']
    
    for i, (mask, label) in enumerate(zip(masks, labels)):
        color = COLORS[int(label) % len(COLORS)]
        mask_binary = mask[0] > 0.5
        
        # 마스크를 컬러로 변환
        colored_mask = np.zeros((*mask_binary.shape, 3))
        colored_mask[mask_binary] = color
        
        # 반투명 오버레이
        axes[2].imshow(colored_mask, alpha=0.5)
    
    axes[2].set_title(f'Segmentation ({len(masks)} masks)', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Detailed visualization saved: {save_path}")
    plt.close()


def create_overlay(image, prediction, class_names, alpha=0.5):
    """
    오버레이 이미지 생성 (원본 + Detection + Segmentation)
    
    Returns:
        overlay: (H, W, 3) numpy array
    """
    overlay = image.copy()
    
    boxes = prediction['boxes']
    labels = prediction['labels']
    scores = prediction['scores']
    masks = prediction['masks']
    
    # 1. Masks 오버레이
    for mask, label in zip(masks, labels):
        color = np.array(COLORS[int(label) % len(COLORS)]) * 255
        mask_binary = mask[0] > 0.5
        overlay[mask_binary] = overlay[mask_binary] * (1 - alpha) + color * alpha
    
    # 2. BBoxes
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.astype(int)
        color = (np.array(COLORS[int(label) % len(COLORS)]) * 255).astype(int).tolist()
        
        # BBox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        
        # Label 배경
        class_name = class_names.get(int(label), f'Class {label}')
        text = f'{class_name}: {score:.2f}'
        
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        cv2.rectangle(
            overlay,
            (x1, y1 - text_height - 10),
            (x1 + text_width + 5, y1),
            color,
            -1
        )
        
        # Label 텍스트
        cv2.putText(
            overlay,
            text,
            (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
    
    return overlay.astype(np.uint8)


def print_prediction_summary(prediction, class_names):
    """예측 결과 요약 출력"""
    print(f"\n  Detection Summary:")
    print(f"    Total objects detected: {len(prediction['boxes'])}")
    
    if len(prediction['boxes']) > 0:
        print(f"\n    {'#':<4} {'Class':<20} {'Confidence':<12} {'BBox':<30}")
        print(f"    {'-'*66}")
        
        for i, (label, score, box) in enumerate(zip(
            prediction['labels'],
            prediction['scores'],
            prediction['boxes']
        )):
            class_name = class_names.get(int(label), f'Class {label}')
            bbox_str = f"[{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]"
            print(f"    {i+1:<4} {class_name:<20} {score:<12.3f} {bbox_str:<30}")


def main(args):
    """메인 추론 함수"""
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("Mask R-CNN Inference on Endoscopic Images")
    print("="*80)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Using device: {device}")
    
    # ==================== 모델 로드 ====================
    print("\n" + "="*80)
    print("Loading Model")
    print("="*80)
    
    model = get_maskrcnn_model(
        num_classes=args.num_classes,
        pretrained_backbone=False
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded from: {args.checkpoint}")
    
    # Class names
    class_names = AIHubEndoscopicDataset.CLASSES
    
    # ==================== 이미지 처리 ====================
    print("\n" + "="*80)
    print("Processing Images")
    print("="*80)
    
    # 입력이 폴더인지 파일인지 확인
    input_path = Path(args.input)
    
    if input_path.is_dir():
        image_files = list(input_path.glob('*.jpg')) + \
                     list(input_path.glob('*.jpeg')) + \
                     list(input_path.glob('*.png'))
    else:
        image_files = [input_path]
    
    print(f"\n✓ Found {len(image_files)} images")
    
    # 각 이미지 처리
    for idx, img_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] Processing: {img_path.name}")
        
        # 이미지 로드
        image_tensor, original_image, original_size = load_image(
            str(img_path),
            args.img_size
        )
        
        # 추론
        prediction = predict(model, image_tensor, device, args.conf_threshold)
        
        # 결과 출력
        print_prediction_summary(prediction, class_names)
        
        # 시각화
        img_name = img_path.stem
        
        if args.save_detailed and len(prediction['boxes']) > 0:
            detailed_path = output_dir / f'{img_name}_detailed.png'
            visualize_detailed(original_image, prediction, class_names, detailed_path)
        
        if args.save_overlay and len(prediction['boxes']) > 0:
            overlay = create_overlay(original_image, prediction, class_names)
            overlay_path = output_dir / f'{img_name}_overlay.jpg'
            cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            print(f"  ✓ Overlay image saved: {overlay_path}")
        
        if len(prediction['boxes']) == 0:
            print(f"  ⚠ No objects detected (confidence >= {args.conf_threshold})")
    
    print("\n" + "="*80)
    print("Inference Complete!")
    print("="*80)
    print(f"\n✓ Results saved in: {output_dir}")
    print(f"✓ Processed {len(image_files)} images")


if __name__ == "__main__":
    args = parse_args()
    
    print("\n" + "="*80)
    print("Inference Configuration")
    print("="*80)
    for key in sorted(vars(args).keys()):
        print(f"  {key}: {getattr(args, key)}")
    
    try:
        main(args)
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
