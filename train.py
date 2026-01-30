"""
Training Script for Mask R-CNN on AI Hub Endoscopic Dataset
================================================================
학습 파이프라인:
1. 데이터 로딩 (Train 1000장/class, Val 150장/class)
2. 모델 초기화 (ResNet50 + Mask R-CNN)
3. 학습 루프
4. Loss 출력
5. 체크포인트 저장
================================================================
"""

import os
import sys
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

# 프로젝트 모듈 import
from models.mask_rcnn import get_maskrcnn_model, print_model_structure
from datasets.dataset import AIHubEndoscopicDataset, collate_fn
from utils.transforms import get_train_transforms, get_val_transforms
from utils.engine import (
    train_one_epoch,
    evaluate,
    save_checkpoint,
    load_checkpoint,
    warmup_lr_scheduler,
    compute_metrics
)


def parse_args():
    """커맨드라인 인자 파싱"""
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on AI Hub Endoscopic Dataset'
    )
    
    # ==================== 데이터 설정 ====================
    parser.add_argument('--data-root', type=str, required=True,
                        help='AI Hub 데이터 루트 디렉토리')
    parser.add_argument('--train-samples-per-class', type=int, default=1000,
                        help='클래스별 학습 샘플 수')
    parser.add_argument('--val-samples-per-class', type=int, default=150,
                        help='클래스별 검증 샘플 수')
    parser.add_argument('--test-samples-per-class', type=int, default=400,
                        help='클래스별 테스트 샘플 수')
    
    # ==================== 모델 설정 ====================
    parser.add_argument('--num-classes', type=int, default=7,
                        help='클래스 개수 (배경 포함, AI Hub: 7)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='ImageNet pretrained backbone 사용')
    parser.add_argument('--trainable-backbone-layers', type=int, default=5,
                        help='학습 가능한 backbone layer 수 (0-5)')
    parser.add_argument('--resume', type=str, default='',
                        help='체크포인트 경로 (재개)')
    
    # ==================== 학습 설정 ====================
    parser.add_argument('--epochs', type=int, default=50,
                        help='총 epoch 수')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='배치 크기 (GPU 메모리에 따라 조정)')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='학습률')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                        help='가중치 감쇠')
    
    # LR Scheduler
    parser.add_argument('--lr-scheduler', type=str, default='step',
                        choices=['step', 'cosine', 'plateau'],
                        help='Learning rate scheduler')
    parser.add_argument('--lr-step-size', type=int, default=10,
                        help='StepLR step size')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='StepLR gamma')
    
    # Warmup
    parser.add_argument('--warmup-epochs', type=int, default=1,
                        help='Warmup epochs')
    parser.add_argument('--warmup-factor', type=float, default=0.1,
                        help='Warmup factor')
    
    # ==================== 기타 설정 ====================
    parser.add_argument('--img-size', type=int, default=512,
                        help='입력 이미지 크기')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='데이터로더 워커 수')
    parser.add_argument('--output-dir', type=str, default='./checkpoints',
                        help='체크포인트 저장 디렉토리')
    parser.add_argument('--device', type=str, default='cuda',
                        help='학습 디바이스 (cuda/cpu)')
    parser.add_argument('--print-freq', type=int, default=50,
                        help='로그 출력 빈도')
    parser.add_argument('--save-freq', type=int, default=5,
                        help='체크포인트 저장 빈도 (epoch)')
    parser.add_argument('--eval-freq', type=int, default=5,
                        help='평가 빈도 (epoch)')
    parser.add_argument('--mixed-precision', action='store_true',
                        help='Mixed precision training 사용')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def set_seed(seed):
    """Reproducibility를 위한 seed 설정"""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main(args):
    """메인 학습 함수"""
    
    # Seed 설정
    set_seed(args.seed)
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("AI Hub Endoscopic Image - Mask R-CNN Training")
    print("="*80)
    
    # Device 설정
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Using device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # ==================== 데이터셋 로드 ====================
    print("\n" + "="*80)
    print("STEP 1: Loading Datasets")
    print("="*80)
    
    train_dataset = AIHubEndoscopicDataset(
        root_dir=args.data_root,
        split='train',
        transforms=get_train_transforms(img_size=args.img_size),
        train_samples_per_class=args.train_samples_per_class,
        val_samples_per_class=args.val_samples_per_class,
        test_samples_per_class=args.test_samples_per_class,
        seed=args.seed
    )
    
    val_dataset = AIHubEndoscopicDataset(
        root_dir=args.data_root,
        split='val',
        transforms=get_val_transforms(img_size=args.img_size),
        train_samples_per_class=args.train_samples_per_class,
        val_samples_per_class=args.val_samples_per_class,
        test_samples_per_class=args.test_samples_per_class,
        seed=args.seed
    )
    
    print(f"\n✓ Datasets loaded successfully!")
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val: {len(val_dataset)} images")
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"\n✓ DataLoaders created!")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # ==================== 모델 생성 ====================
    print("\n" + "="*80)
    print("STEP 2: Creating Model")
    print("="*80)
    
    model = get_maskrcnn_model(
        num_classes=args.num_classes,
        pretrained_backbone=args.pretrained,
        trainable_backbone_layers=args.trainable_backbone_layers
    )
    model.to(device)
    
    print_model_structure(model)
    
    # ==================== Optimizer & Scheduler ====================
    print("\n" + "="*80)
    print("STEP 3: Setting up Optimizer & Scheduler")
    print("="*80)
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    print(f"\n✓ Optimizer: SGD")
    print(f"  Learning rate: {args.lr}")
    print(f"  Momentum: {args.momentum}")
    print(f"  Weight decay: {args.weight_decay}")
    
    # LR Scheduler
    if args.lr_scheduler == 'step':
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.lr_step_size,
            gamma=args.lr_gamma
        )
    elif args.lr_scheduler == 'cosine':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs
        )
    elif args.lr_scheduler == 'plateau':
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=args.lr_gamma,
            patience=5
        )
    
    print(f"\n✓ LR Scheduler: {args.lr_scheduler}")
    
    # Mixed Precision
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None
    if args.mixed_precision:
        print(f"\n✓ Mixed precision training enabled")
    
    # ==================== Resume ====================
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"\n✓ Resuming from checkpoint: {args.resume}")
            start_epoch, _ = load_checkpoint(
                model, optimizer, args.resume, device, lr_scheduler
            )
            start_epoch += 1
    
    # ==================== Training Loop ====================
    print("\n" + "="*80)
    print("STEP 4: Training")
    print("="*80)
    print(f"\nStarting training from epoch {start_epoch} to {args.epochs}...")
    print(f"Batch size: {args.batch_size}")
    print(f"Total iterations per epoch: {len(train_loader)}")
    
    best_loss = float('inf')
    best_metrics = {}
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*80}")
        
        # ==================== Train ====================
        metric_logger = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
            print_freq=args.print_freq,
            scaler=scaler
        )
        
        # LR Scheduler step
        if args.lr_scheduler != 'plateau':
            lr_scheduler.step()
        
        current_loss = metric_logger.meters['loss'].global_avg
        
        # ==================== Evaluate ====================
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            print(f"\n{'='*80}")
            print(f"Validation - Epoch {epoch + 1}")
            print(f"{'='*80}")
            
            results = evaluate(model, val_loader, device)
            
            # 메트릭 계산
            metrics = compute_metrics(results['predictions'], results['targets'])
            
            print(f"\n✓ Validation Results:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1_score']:.4f}")
            print(f"  TP: {metrics['true_positives']}, FP: {metrics['false_positives']}, FN: {metrics['false_negatives']}")
            
            # Plateau scheduler
            if args.lr_scheduler == 'plateau':
                lr_scheduler.step(current_loss)
        
        # ==================== Save Checkpoint ====================
        if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1:
            checkpoint_path = output_dir / f'maskrcnn_epoch_{epoch + 1:03d}.pth'
            save_checkpoint(
                model, optimizer, epoch, current_loss, checkpoint_path, lr_scheduler
            )
            
            # Save best model
            if current_loss < best_loss:
                best_loss = current_loss
                best_path = output_dir / 'maskrcnn_best.pth'
                save_checkpoint(
                    model, optimizer, epoch, current_loss, best_path, lr_scheduler
                )
                print(f"✓ New best model! Loss: {best_loss:.4f}")
        
        # Current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nCurrent LR: {current_lr:.6f}")
    
    # ==================== Training Complete ====================
    print("\n" + "="*80)
    print("Training Completed!")
    print("="*80)
    print(f"\n✓ Best loss: {best_loss:.4f}")
    print(f"✓ Checkpoints saved in: {output_dir}")
    print(f"✓ Total epochs: {args.epochs}")
    
    print("\n다음 단계:")
    print(f"  1. 평가: python eval.py --checkpoint {output_dir}/maskrcnn_best.pth --data-root {args.data_root}")
    print(f"  2. 추론: python inference.py --checkpoint {output_dir}/maskrcnn_best.pth --input <image_path>")


if __name__ == "__main__":
    args = parse_args()
    
    # Configuration 출력
    print("\n" + "="*80)
    print("Training Configuration")
    print("="*80)
    config_dict = vars(args)
    for key in sorted(config_dict.keys()):
        print(f"  {key}: {config_dict[key]}")
    
    try:
        main(args)
    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print("Training interrupted by user")
        print("="*80)
    except Exception as e:
        print("\n\n" + "="*80)
        print(f"Error occurred: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()
        sys.exit(1)
