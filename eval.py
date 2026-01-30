"""
Evaluation Script for Mask R-CNN
- COCO API를 사용한 정량적 평가
- mAP (mean Average Precision) 계산
- Detection과 Segmentation 모두 평가
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

# 프로젝트 모듈 import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.mask_rcnn import get_maskrcnn_resnet50
from datasets.dataset import EndoscopicDataset, collate_fn
from utils.transforms import get_val_transforms


def parse_args():
    """커맨드라인 인자 파싱"""
    parser = argparse.ArgumentParser(description='Evaluate Mask R-CNN')
    
    parser.add_argument('--data-root', type=str, default='./data/images/val',
                        help='검증 이미지 디렉토리')
    parser.add_argument('--ann-file', type=str, default='./data/annotations/val.json',
                        help='검증 annotation 파일')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='모델 체크포인트 경로')
    parser.add_argument('--num-classes', type=int, default=5,
                        help='클래스 개수')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='배치 크기')
    parser.add_argument('--img-size', type=int, default=512,
                        help='입력 이미지 크기')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='워커 수')
    parser.add_argument('--device', type=str, default='cuda',
                        help='디바이스')
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='결과 저장 디렉토리')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                        help='Confidence threshold')
    
    return parser.parse_args()


@torch.no_grad()
def evaluate_coco(model, data_loader, coco_gt, device, conf_threshold=0.5):
    """
    COCO 평가 메트릭 계산
    
    Args:
        model: Mask R-CNN 모델
        data_loader: 검증 데이터로더
        coco_gt: COCO ground truth
        device: cuda/cpu
        conf_threshold: confidence threshold
    
    Returns:
        results: 평가 결과
    """
    model.eval()
    
    # 예측 결과 저장
    coco_results_bbox = []
    coco_results_segm = []
    
    print("\nRunning inference...")
    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        
        # Forward
        outputs = model(images)
        
        # 결과 처리
        for target, output in zip(targets, outputs):
            image_id = int(target['image_id'][0])
            
            # Confidence threshold 적용
            keep = output['scores'] > conf_threshold
            boxes = output['boxes'][keep]
            scores = output['scores'][keep]
            labels = output['labels'][keep]
            masks = output['masks'][keep]
            
            # COCO format으로 변환
            boxes = boxes.cpu().numpy()
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()
            masks = masks.cpu().numpy()
            
            for i in range(len(boxes)):
                # Bounding box result
                bbox = boxes[i]
                bbox_coco = [
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[2] - bbox[0]),
                    float(bbox[3] - bbox[1])
                ]
                
                bbox_result = {
                    'image_id': image_id,
                    'category_id': int(labels[i]),
                    'bbox': bbox_coco,
                    'score': float(scores[i])
                }
                coco_results_bbox.append(bbox_result)
                
                # Segmentation result
                mask = masks[i, 0] > 0.5  # threshold mask
                from pycocotools import mask as mask_util
                
                # RLE 인코딩
                rle = mask_util.encode(
                    np.asfortranarray(mask.astype(np.uint8))
                )
                rle['counts'] = rle['counts'].decode('utf-8')
                
                segm_result = {
                    'image_id': image_id,
                    'category_id': int(labels[i]),
                    'segmentation': rle,
                    'score': float(scores[i])
                }
                coco_results_segm.append(segm_result)
    
    # COCO 평가
    print("\n" + "="*50)
    print("COCO Evaluation - Bounding Box")
    print("="*50)
    
    if len(coco_results_bbox) > 0:
        coco_dt_bbox = coco_gt.loadRes(coco_results_bbox)
        coco_eval_bbox = COCOeval(coco_gt, coco_dt_bbox, 'bbox')
        coco_eval_bbox.evaluate()
        coco_eval_bbox.accumulate()
        coco_eval_bbox.summarize()
        
        bbox_stats = coco_eval_bbox.stats
    else:
        print("No predictions found!")
        bbox_stats = None
    
    print("\n" + "="*50)
    print("COCO Evaluation - Segmentation")
    print("="*50)
    
    if len(coco_results_segm) > 0:
        coco_dt_segm = coco_gt.loadRes(coco_results_segm)
        coco_eval_segm = COCOeval(coco_gt, coco_dt_segm, 'segm')
        coco_eval_segm.evaluate()
        coco_eval_segm.accumulate()
        coco_eval_segm.summarize()
        
        segm_stats = coco_eval_segm.stats
    else:
        print("No segmentation predictions found!")
        segm_stats = None
    
    return {
        'bbox_results': coco_results_bbox,
        'segm_results': coco_results_segm,
        'bbox_stats': bbox_stats,
        'segm_stats': segm_stats
    }


def compute_iou(box1, box2):
    """IoU 계산"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def compute_dice(mask1, mask2):
    """Dice coefficient 계산"""
    intersection = np.logical_and(mask1, mask2).sum()
    return 2.0 * intersection / (mask1.sum() + mask2.sum() + 1e-8)


def main(args):
    """메인 평가 함수"""
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device 설정
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ==================== 데이터셋 로드 ====================
    print("\n" + "="*50)
    print("Loading dataset...")
    print("="*50)
    
    dataset = EndoscopicDataset(
        root=args.data_root,
        annotation_file=args.ann_file,
        transforms=get_val_transforms(img_size=args.img_size),
        mode='val'
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"Evaluation samples: {len(dataset)}")
    
    # COCO ground truth
    coco_gt = COCO(args.ann_file)
    
    # ==================== 모델 로드 ====================
    print("\n" + "="*50)
    print("Loading model...")
    print("="*50)
    
    model = get_maskrcnn_resnet50(
        num_classes=args.num_classes,
        pretrained_backbone=False
    )
    
    # 체크포인트 로드
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    print(f"Model loaded from: {args.checkpoint}")
    
    # ==================== 평가 실행 ====================
    results = evaluate_coco(
        model,
        data_loader,
        coco_gt,
        device,
        conf_threshold=args.conf_threshold
    )
    
    # ==================== 결과 저장 ====================
    print("\n" + "="*50)
    print("Saving results...")
    print("="*50)
    
    # JSON 저장
    bbox_path = os.path.join(args.output_dir, 'bbox_results.json')
    segm_path = os.path.join(args.output_dir, 'segm_results.json')
    
    with open(bbox_path, 'w') as f:
        json.dump(results['bbox_results'], f)
    
    with open(segm_path, 'w') as f:
        json.dump(results['segm_results'], f)
    
    print(f"Results saved to {args.output_dir}")
    
    # 요약 통계
    if results['bbox_stats'] is not None:
        print("\n" + "="*50)
        print("Summary")
        print("="*50)
        print(f"BBox mAP@0.5:0.95: {results['bbox_stats'][0]:.4f}")
        print(f"BBox mAP@0.5: {results['bbox_stats'][1]:.4f}")
        print(f"Segm mAP@0.5:0.95: {results['segm_stats'][0]:.4f}")
        print(f"Segm mAP@0.5: {results['segm_stats'][1]:.4f}")


if __name__ == "__main__":
    args = parse_args()
    
    print("\n" + "="*50)
    print("Evaluation Configuration")
    print("="*50)
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    
    main(args)
