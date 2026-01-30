"""
Inference Script for Mask R-CNN
- 단일 이미지 또는 이미지 폴더에 대한 추론
- Detection + Segmentation 시각화
- 결과 이미지 저장
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
from matplotlib.patches import Polygon

# 프로젝트 모듈 import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.mask_rcnn import get_maskrcnn_resnet50
from utils.transforms import get_test_transforms


# COCO 카테고리 색상 (visualization용)
COLORS = [
    [0.000, 0.447, 0.741],
    [0.850, 0.325, 0.098],
    [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
    [0.301, 0.745, 0.933],
    [0.635, 0.078, 0.184],
    [0.300, 0.300, 0.300],
]


def parse_args():
    """커맨드라인 인자 파싱"""
    parser = argparse.ArgumentParser(description='Inference with Mask R-CNN')
    
    parser.add_argument('--input', type=str, required=True,
                        help='입력 이미지 경로 또는 폴더')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='모델 체크포인트 경로')
    parser.add_argument('--output-dir', type=str, default='./inference_results',
                        help='결과 저장 디렉토리')
    parser.add_argument('--num-classes', type=int, default=5,
                        help='클래스 개수')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--device', type=str, default='cuda',
                        help='디바이스')
    parser.add_argument('--img-size', type=int, default=512,
                        help='입력 이미지 크기')
    parser.add_argument('--class-names', type=str, nargs='+',
                        default=['background', 'polyp', 'adenoma', 'cancer', 'normal'],
                        help='클래스 이름 리스트')
    
    return parser.parse_args()


def load_image(image_path, img_size=512):
    """
    이미지 로드 및 전처리
    
    Args:
        image_path: 이미지 경로
        img_size: 리사이즈 크기
    
    Returns:
        image_tensor, original_image
    """
    # 원본 이미지
    original = Image.open(image_path).convert('RGB')
    original = np.array(original)
    
    # Transform
    transform = get_test_transforms(img_size=img_size)
    transformed = transform(image=original)
    image = transformed['image']
    
    # Tensor로 변환
    image_tensor = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
    
    return image_tensor, original


@torch.no_grad()
def predict(model, image_tensor, device, conf_threshold=0.5):
    """
    추론 수행
    
    Args:
        model: Mask R-CNN 모델
        image_tensor: 입력 이미지 텐서
        device: cuda/cpu
        conf_threshold: confidence threshold
    
    Returns:
        predictions: 예측 결과
    """
    model.eval()
    
    # GPU로 이동
    image_tensor = image_tensor.to(device)
    
    # 배치 차원 추가
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    
    # Forward
    predictions = model(image_tensor)
    
    # Threshold 적용
    pred = predictions[0]
    keep = pred['scores'] > conf_threshold
    
    result = {
        'boxes': pred['boxes'][keep].cpu().numpy(),
        'labels': pred['labels'][keep].cpu().numpy(),
        'scores': pred['scores'][keep].cpu().numpy(),
        'masks': pred['masks'][keep].cpu().numpy()
    }
    
    return result


def visualize_prediction(image, prediction, class_names, save_path=None, show=True):
    """
    예측 결과 시각화
    
    Args:
        image: 원본 이미지
        prediction: 예측 결과 딕셔너리
        class_names: 클래스 이름 리스트
        save_path: 저장 경로
        show: matplotlib show 여부
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 원본 이미지
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Detection 결과
    axes[1].imshow(image)
    boxes = prediction['boxes']
    labels = prediction['labels']
    scores = prediction['scores']
    
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        x1, y1, x2, y2 = box
        color = COLORS[label % len(COLORS)]
        
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2,
            edgecolor=color,
            facecolor='none'
        )
        axes[1].add_patch(rect)
        
        # 라벨 텍스트
        class_name = class_names[label] if label < len(class_names) else f'Class {label}'
        axes[1].text(
            x1, y1 - 5,
            f'{class_name}: {score:.2f}',
            color='white',
            fontsize=10,
            bbox=dict(facecolor=color, alpha=0.8, edgecolor='none', pad=1)
        )
    
    axes[1].set_title(f'Detection ({len(boxes)} objects)')
    axes[1].axis('off')
    
    # Segmentation 결과
    axes[2].imshow(image)
    masks = prediction['masks']
    
    # 마스크 오버레이
    for i, (mask, label) in enumerate(zip(masks, labels)):
        color = COLORS[label % len(COLORS)]
        mask_binary = mask[0] > 0.5
        
        # 마스크를 컬러로 변환
        colored_mask = np.zeros((*mask_binary.shape, 3))
        colored_mask[mask_binary] = color
        
        # 반투명 오버레이
        axes[2].imshow(colored_mask, alpha=0.5)
    
    axes[2].set_title(f'Segmentation ({len(masks)} masks)')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # 저장
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    # 표시
    if show:
        plt.show()
    else:
        plt.close()


def create_overlay_image(image, prediction, class_names, alpha=0.5):
    """
    Detection + Segmentation 결과를 원본 이미지에 오버레이
    
    Args:
        image: 원본 이미지
        prediction: 예측 결과
        class_names: 클래스 이름
        alpha: 투명도
    
    Returns:
        overlay_image: 오버레이된 이미지
    """
    overlay = image.copy()
    
    boxes = prediction['boxes']
    labels = prediction['labels']
    scores = prediction['scores']
    masks = prediction['masks']
    
    # 마스크 오버레이
    for mask, label in zip(masks, labels):
        color = np.array(COLORS[label % len(COLORS)]) * 255
        mask_binary = mask[0] > 0.5
        overlay[mask_binary] = overlay[mask_binary] * (1 - alpha) + color * alpha
    
    # 바운딩 박스
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.astype(int)
        color = (np.array(COLORS[label % len(COLORS)]) * 255).astype(int).tolist()
        
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        
        # 라벨 텍스트
        class_name = class_names[label] if label < len(class_names) else f'Class {label}'
        text = f'{class_name}: {score:.2f}'
        
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        cv2.rectangle(
            overlay,
            (x1, y1 - text_height - 5),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        cv2.putText(
            overlay,
            text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    return overlay.astype(np.uint8)


def main(args):
    """메인 추론 함수"""
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device 설정
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ==================== 모델 로드 ====================
    print("\n" + "="*50)
    print("Loading model...")
    print("="*50)
    
    model = get_maskrcnn_resnet50(
        num_classes=args.num_classes,
        pretrained_backbone=False
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    print(f"Model loaded from: {args.checkpoint}")
    
    # ==================== 이미지 처리 ====================
    # 입력이 폴더인지 파일인지 확인
    if os.path.isdir(args.input):
        image_files = [
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ]
    else:
        image_files = [args.input]
    
    print(f"\nProcessing {len(image_files)} images...")
    
    # 각 이미지에 대해 추론
    for img_path in image_files:
        print(f"\nProcessing: {img_path}")
        
        # 이미지 로드
        image_tensor, original_image = load_image(img_path, args.img_size)
        
        # 추론
        prediction = predict(model, image_tensor, device, args.conf_threshold)
        
        # 결과 출력
        num_objects = len(prediction['boxes'])
        print(f"Detected {num_objects} objects")
        
        if num_objects > 0:
            for i, (label, score) in enumerate(zip(prediction['labels'], prediction['scores'])):
                class_name = args.class_names[label] if label < len(args.class_names) else f'Class {label}'
                print(f"  {i+1}. {class_name} (confidence: {score:.3f})")
        
        # 시각화
        img_name = os.path.basename(img_path)
        save_path = os.path.join(args.output_dir, f'result_{img_name}')
        
        # Matplotlib 시각화
        visualize_prediction(
            original_image,
            prediction,
            args.class_names,
            save_path=save_path.replace('.jpg', '_detailed.png'),
            show=False
        )
        
        # 오버레이 이미지
        overlay = create_overlay_image(original_image, prediction, args.class_names)
        cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    print("\n" + "="*50)
    print("Inference completed!")
    print(f"Results saved in: {args.output_dir}")
    print("="*50)


if __name__ == "__main__":
    args = parse_args()
    
    print("\n" + "="*50)
    print("Inference Configuration")
    print("="*50)
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    
    main(args)
