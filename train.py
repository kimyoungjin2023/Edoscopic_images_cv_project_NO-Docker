"""
Training Script for Mask R-CNN
- 전체 학습 파이프라인
- Configuration 설정
- Model, DataLoader, Optimizer 초기화
- 학습 루프 및 모델 저장
"""

import os
import sys
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime

# 프로젝트 모듈 import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.mask_rcnn import get_maskrcnn_resnet50
from datasets.dataset import EndoscopicDataset, collate_fn
from utils.transforms import get_train_transforms, get_val_transforms
from utils.engine import train_one_epoch, evaluate, save_checkpoint, warmup_lr_scheduler


def parse_args():
    """커맨드라인 인자 파싱"""
    parser = argparse.ArgumentParser(description='Train Mask R-CNN on Endoscopic Images')
    
    # 데이터 관련
    parser.add_argument('--data-root', type=str, default='./data/images/train',
                        help='이미지 루트 디렉토리')
    parser.add_argument('--train-ann', type=str, default='./data/annotations/train.json',
                        help='학습 annotation 파일')
    parser.add_argument('--val-ann', type=str, default='./data/annotations/val.json',
                        help='검증 annotation 파일')
    
    # 모델 관련
    parser.add_argument('--num-classes', type=int, default=5,
                        help='클래스 개수 (배경 포함)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='ImageNet pretrained backbone 사용')
    parser.add_argument('--resume', type=str, default='',
                        help='체크포인트 경로 (재개)')
    
    # 학습 관련
    parser.add_argument('--epochs', type=int, default=50,
                        help='총 epoch 수')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='배치 크기')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='학습률')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                        help='가중치 감쇠')
    parser.add_argument('--lr-step-size', type=int, default=10,
                        help='LR 감소 step')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='LR 감소 비율')
    
    # 기타
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
                        help='체크포인트 저장 빈도')
    
    return parser.parse_args()


def main(args):
    """메인 학습 함수"""
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device 설정
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ==================== 데이터셋 로드 ====================
    print("\n" + "="*50)
    print("Loading datasets...")
    print("="*50)
    
    # Train dataset
    train_dataset = EndoscopicDataset(
        root=args.data_root,
        annotation_file=args.train_ann,
        transforms=get_train_transforms(img_size=args.img_size),
        mode='train'
    )
    
    # Validation dataset
    val_dataset = EndoscopicDataset(
        root=args.data_root.replace('train', 'val'),
        annotation_file=args.val_ann,
        transforms=get_val_transforms(img_size=args.img_size),
        mode='val'
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Number of classes: {args.num_classes}")
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # ==================== 모델 생성 ====================
    print("\n" + "="*50)
    print("Creating model...")
    print("="*50)
    
    model = get_maskrcnn_resnet50(
        num_classes=args.num_classes,
        pretrained_backbone=args.pretrained
    )
    model.to(device)
    
    # 모델 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    # ==================== Optimizer & Scheduler ====================
    # SGD optimizer (Mask R-CNN 논문 설정)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma
    )
    
    # ==================== Resume from checkpoint ====================
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            from utils.engine import load_checkpoint
            start_epoch, _ = load_checkpoint(model, optimizer, args.resume, device)
            start_epoch += 1
        else:
            print(f"No checkpoint found at '{args.resume}'")
    
    # ==================== Training Loop ====================
    print("\n" + "="*50)
    print("Start training...")
    print("="*50)
    
    best_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 50)
        
        # Train
        metric_logger = train_one_epoch(
            model,
            optimizer,
            train_loader,
            device,
            epoch,
            print_freq=args.print_freq
        )
        
        # Update learning rate
        lr_scheduler.step()
        
        # Evaluate
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print("\nRunning validation...")
            eval_metric = evaluate(model, val_loader, device)
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1:
            current_loss = metric_logger.meters['loss'].global_avg
            
            checkpoint_path = os.path.join(
                args.output_dir,
                f'maskrcnn_epoch_{epoch + 1}.pth'
            )
            save_checkpoint(model, optimizer, epoch, current_loss, checkpoint_path)
            
            # Save best model
            if current_loss < best_loss:
                best_loss = current_loss
                best_path = os.path.join(args.output_dir, 'maskrcnn_best.pth')
                save_checkpoint(model, optimizer, epoch, current_loss, best_path)
                print(f"New best model saved! Loss: {best_loss:.4f}")
    
    print("\n" + "="*50)
    print("Training completed!")
    print("="*50)
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved in: {args.output_dir}")


if __name__ == "__main__":
    args = parse_args()
    
    # Print configuration
    print("\n" + "="*50)
    print("Training Configuration")
    print("="*50)
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    
    main(args)
