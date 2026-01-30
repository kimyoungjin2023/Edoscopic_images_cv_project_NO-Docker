"""
Evaluation Script for Mask R-CNN
================================================================
Validation / Test 평가:
- Detection 성능 (mAP, Precision, Recall)
- Segmentation 성능 (Dice, IoU)
- 클래스별 성능 분석
================================================================
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from collections import defaultdict
import json

from models.mask_rcnn import get_maskrcnn_model
from datasets.dataset import AIHubEndoscopicDataset, collate_fn
from utils.transforms import get_val_transforms
from utils.engine import evaluate, compute_metrics, box_iou


def parse_args():
    """커맨드라인 인자 파싱"""
    parser = argparse.ArgumentParser(
        description='Evaluate Mask R-CNN on AI Hub Endoscopic Dataset'
    )
    
    # 데이터
    parser.add_argument('--data-root', type=str, required=True,
                        help='AI Hub 데이터 루트 디렉토리')
    parser.add_argument('--split', type=str, default='val',
                        choices=['val', 'test'],
                        help='평가 데이터셋 (val or test)')
    parser.add_argument('--train-samples-per-class', type=int, default=1000)
    parser.add_argument('--val-samples-per-class', type=int, default=150)
    parser.add_argument('--test-samples-per-class', type=int, default=400)
    
    # 모델
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='모델 체크포인트 경로')
    parser.add_argument('--num-classes', type=int, default=7,
                        help='클래스 개수')
    
    # 평가 설정
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                        help='IoU threshold for matching')
    parser.add_argument('--img-size', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    
    # 출력
    parser.add_argument('--output-dir', type=str, default='./eval_results',
                        help='결과 저장 디렉토리')
    parser.add_argument('--save-predictions', action='store_true',
                        help='예측 결과 저장')
    
    return parser.parse_args()


def compute_detection_metrics(predictions, targets, iou_threshold=0.5, conf_threshold=0.5, num_classes=7):
    """
    Detection 메트릭 계산 (클래스별)
    
    Returns:
        metrics_per_class: 클래스별 메트릭
        overall_metrics: 전체 메트릭
    """
    from collections import defaultdict
    
    # 클래스별 통계
    class_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'scores': []})
    
    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes']
        pred_labels = pred['labels']
        pred_scores = pred['scores']
        
        target_boxes = target['boxes']
        target_labels = target['labels']
        
        # Confidence threshold 적용
        keep = pred_scores >= conf_threshold
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        pred_scores = pred_scores[keep]
        
        # 각 클래스별 처리
        for class_id in range(1, num_classes):  # 배경 제외
            # 해당 클래스의 prediction
            pred_mask = pred_labels == class_id
            class_pred_boxes = pred_boxes[pred_mask]
            class_pred_scores = pred_scores[pred_mask]
            
            # 해당 클래스의 target
            target_mask = target_labels == class_id
            class_target_boxes = target_boxes[target_mask]
            
            num_pred = len(class_pred_boxes)
            num_target = len(class_target_boxes)
            
            if num_pred == 0 and num_target == 0:
                continue
            elif num_pred == 0:
                class_stats[class_id]['fn'] += num_target
                continue
            elif num_target == 0:
                class_stats[class_id]['fp'] += num_pred
                continue
            
            # IoU 계산
            ious = box_iou(class_pred_boxes, class_target_boxes)
            
            # Matching
            matched_targets = set()
            
            for i, score in enumerate(class_pred_scores):
                max_iou, max_idx = ious[i].max(0)
                
                if max_iou >= iou_threshold and max_idx.item() not in matched_targets:
                    class_stats[class_id]['tp'] += 1
                    matched_targets.add(max_idx.item())
                    class_stats[class_id]['scores'].append(score.item())
                else:
                    class_stats[class_id]['fp'] += 1
            
            # False negatives
            class_stats[class_id]['fn'] += num_target - len(matched_targets)
    
    # 클래스별 메트릭 계산
    metrics_per_class = {}
    
    for class_id in range(1, num_classes):
        stats = class_stats[class_id]
        tp = stats['tp']
        fp = stats['fp']
        fn = stats['fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics_per_class[class_id] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    # 전체 메트릭 (macro average)
    total_tp = sum(m['tp'] for m in metrics_per_class.values())
    total_fp = sum(m['fp'] for m in metrics_per_class.values())
    total_fn = sum(m['fn'] for m in metrics_per_class.values())
    
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    overall_metrics = {
        'precision': overall_precision,
        'recall': overall_recall,
        'f1_score': overall_f1,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn
    }
    
    return metrics_per_class, overall_metrics


def compute_segmentation_metrics(predictions, targets, iou_threshold=0.5, conf_threshold=0.5):
    """
    Segmentation 메트릭 계산 (Dice, IoU)
    """
    dice_scores = []
    iou_scores = []
    
    for pred, target in zip(predictions, targets):
        pred_masks = pred['masks']
        pred_scores = pred['scores']
        target_masks = target['masks']
        
        # Confidence threshold
        keep = pred_scores >= conf_threshold
        pred_masks = pred_masks[keep]
        
        if len(pred_masks) == 0 or len(target_masks) == 0:
            continue
        
        # Binary masks
        pred_masks = (pred_masks > 0.5).squeeze(1).cpu().numpy()
        target_masks = target_masks.cpu().numpy()
        
        # 각 예측 mask에 대해 가장 가까운 target mask 찾기
        for pred_mask in pred_masks:
            best_dice = 0
            best_iou = 0
            
            for target_mask in target_masks:
                # Dice
                intersection = np.logical_and(pred_mask, target_mask).sum()
                dice = 2.0 * intersection / (pred_mask.sum() + target_mask.sum() + 1e-8)
                
                # IoU
                union = np.logical_or(pred_mask, target_mask).sum()
                iou = intersection / (union + 1e-8)
                
                if dice > best_dice:
                    best_dice = dice
                    best_iou = iou
            
            dice_scores.append(best_dice)
            iou_scores.append(best_iou)
    
    mean_dice = np.mean(dice_scores) if dice_scores else 0
    mean_iou = np.mean(iou_scores) if iou_scores else 0
    
    return {
        'dice': mean_dice,
        'iou': mean_iou,
        'num_samples': len(dice_scores)
    }


def main(args):
    """메인 평가 함수"""
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print(f"Mask R-CNN Evaluation - {args.split.upper()} Set")
    print("="*80)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Using device: {device}")
    
    # ==================== 데이터셋 ====================
    print("\n" + "="*80)
    print("Loading Dataset")
    print("="*80)
    
    dataset = AIHubEndoscopicDataset(
        root_dir=args.data_root,
        split=args.split,
        transforms=get_val_transforms(img_size=args.img_size),
        train_samples_per_class=args.train_samples_per_class,
        val_samples_per_class=args.val_samples_per_class,
        test_samples_per_class=args.test_samples_per_class
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"\n✓ Dataset loaded: {len(dataset)} images")
    
    # ==================== 모델 ====================
    print("\n" + "="*80)
    print("Loading Model")
    print("="*80)
    
    model = get_maskrcnn_model(
        num_classes=args.num_classes,
        pretrained_backbone=False
    )
    
    # 체크포인트 로드
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\n✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded from: {args.checkpoint}")
    
    # ==================== 평가 ====================
    print("\n" + "="*80)
    print("Running Evaluation")
    print("="*80)
    
    results = evaluate(model, data_loader, device)
    
    print(f"\n✓ Evaluation complete!")
    print(f"  Total predictions: {len(results['predictions'])}")
    
    # ==================== Detection 메트릭 ====================
    print("\n" + "="*80)
    print("Detection Metrics")
    print("="*80)
    
    metrics_per_class, overall_metrics = compute_detection_metrics(
        results['predictions'],
        results['targets'],
        iou_threshold=args.iou_threshold,
        conf_threshold=args.conf_threshold,
        num_classes=args.num_classes
    )
    
    print(f"\nOverall Metrics (Confidence >= {args.conf_threshold}):")
    print(f"  Precision: {overall_metrics['precision']:.4f}")
    print(f"  Recall: {overall_metrics['recall']:.4f}")
    print(f"  F1-Score: {overall_metrics['f1_score']:.4f}")
    print(f"  TP: {overall_metrics['tp']}, FP: {overall_metrics['fp']}, FN: {overall_metrics['fn']}")
    
    print(f"\nPer-Class Metrics:")
    print(f"  {'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'TP':<6} {'FP':<6} {'FN':<6}")
    print(f"  {'-'*86}")
    
    class_names = AIHubEndoscopicDataset.CLASSES
    for class_id in sorted(metrics_per_class.keys()):
        m = metrics_per_class[class_id]
        class_name = class_names.get(class_id, f'Class_{class_id}')
        print(f"  {class_name:<20} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1_score']:<12.4f} {m['tp']:<6} {m['fp']:<6} {m['fn']:<6}")
    
    # ==================== Segmentation 메트릭 ====================
    print("\n" + "="*80)
    print("Segmentation Metrics")
    print("="*80)
    
    seg_metrics = compute_segmentation_metrics(
        results['predictions'],
        results['targets'],
        iou_threshold=args.iou_threshold,
        conf_threshold=args.conf_threshold
    )
    
    print(f"\nSegmentation Performance:")
    print(f"  Mean Dice Coefficient: {seg_metrics['dice']:.4f}")
    print(f"  Mean IoU: {seg_metrics['iou']:.4f}")
    print(f"  Number of masks evaluated: {seg_metrics['num_samples']}")
    
    # ==================== 결과 저장 ====================
    print("\n" + "="*80)
    print("Saving Results")
    print("="*80)
    
    # JSON으로 저장
    results_dict = {
        'checkpoint': args.checkpoint,
        'split': args.split,
        'conf_threshold': args.conf_threshold,
        'iou_threshold': args.iou_threshold,
        'detection_metrics': {
            'overall': overall_metrics,
            'per_class': {class_names[k]: v for k, v in metrics_per_class.items()}
        },
        'segmentation_metrics': seg_metrics
    }
    
    results_path = output_dir / f'eval_results_{args.split}.json'
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_path}")
    
    # Predictions 저장
    if args.save_predictions:
        pred_path = output_dir / f'predictions_{args.split}.pth'
        torch.save(results, pred_path)
        print(f"✓ Predictions saved to: {pred_path}")
    
    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80)


if __name__ == "__main__":
    args = parse_args()
    
    print("\n" + "="*80)
    print("Evaluation Configuration")
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
