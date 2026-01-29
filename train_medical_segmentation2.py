#!/usr/bin/env python3
"""
의료 영상 Segmentation 학습 스크립트 (궤양, 암, 용종 포함)
Train/Validation split 포함
"""

import torch
import os
import argparse
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from tqdm import tqdm
import time

# Import 경로 수정
try:
    from datasets.medical_folder_dataset import MedicalFolderDataset, collate_fn_filter_empty
except ImportError:
    from datasets.medical_folder_dataset import MedicalFolderDataset, collate_fn_filter_empty


def get_device():
    """최적의 device 선택"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🖥️ Using device: CUDA - {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        print("⚠️ MPS available but disabled (Mask R-CNN compatibility)")
        device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        print(f"🖥️ Using device: CPU")
    return device


def get_model(num_classes):
    """Mask R-CNN 모델 생성"""
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")

    # Box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, 256, num_classes
    )

    return model


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """1 epoch 학습"""
    model.train()
    epoch_loss = 0
    num_batches = 0
    skipped_batches = 0
    
    loss_components = {
        'loss_classifier': 0,
        'loss_box_reg': 0,
        'loss_mask': 0,
        'loss_objectness': 0,
        'loss_rpn_box_reg': 0
    }

    start_time = time.time()
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        # 빈 batch 스킵
        if batch is None:
            skipped_batches += 1
            continue
        
        images, targets = batch
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        try:
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            # NaN 체크
            if torch.isnan(loss):
                print(f"\n⚠️ NaN at epoch {epoch}, batch {batch_idx}")
                skipped_batches += 1
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            
            for k, v in loss_dict.items():
                if k in loss_components:
                    loss_components[k] += v.item()
            
            # Progress bar 업데이트
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg': f'{epoch_loss/num_batches:.4f}'
            })
        
        except Exception as e:
            print(f"\n❌ Error at batch {batch_idx}: {e}")
            skipped_batches += 1
            continue

    elapsed = time.time() - start_time
    
    if num_batches == 0:
        print("⚠️ No valid batches in this epoch!")
        return 0.0
    
    avg_loss = epoch_loss / num_batches
    
    print(f"\n[Epoch {epoch}] Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s | Skipped: {skipped_batches}")
    if epoch % 5 == 0 or epoch == 1:
        print("  Loss components:")
        for k, v in loss_components.items():
            print(f"    {k}: {v/num_batches:.4f}")
    
    return avg_loss


def validate(model, dataloader, device, epoch):
    """Validation"""
    model.eval()
    val_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Validation (Epoch {epoch})"):
            if batch is None:
                continue
            
            images, targets = batch
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            try:
                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())
                val_loss += loss.item()
                num_batches += 1
            except:
                continue
    
    if num_batches == 0:
        return 0.0
    
    avg_val_loss = val_loss / num_batches
    print(f"  📉 Val Loss: {avg_val_loss:.4f}")
    
    return avg_val_loss


def main(args):
    # Device 선택
    device = get_device()
    
    transform = transforms.ToTensor()
    
    # 데이터셋 로드
    print("\n" + "="*60)
    print("Loading Datasets")
    print("="*60)
    
    datasets = []
    
    if args.organ == 'colon' or args.organ == 'both':
        print("\n📁 Loading COLON dataset...")
        colon_dataset = MedicalFolderDataset(
            image_root=args.image_root,
            label_root=args.label_root,
            organ_type='위',
            transforms=transform,
            min_area=args.min_area,
            resize=(args.img_size, args.img_size),
            max_samples=args.max_samples  # ⭐ 500개로 제한
        )
        datasets.append(colon_dataset)
    
    if args.organ == 'stomach' or args.organ == 'both':
        print("\n📁 Loading STOMACH dataset...")
        stomach_dataset = MedicalFolderDataset(
            image_root=args.image_root,
            label_root=args.label_root,
            organ_type='위',
            transforms=transform,
            min_area=args.min_area,
            resize=(args.img_size, args.img_size),
            max_samples=args.max_samples
        )
        datasets.append(stomach_dataset)
    
    # 데이터셋 합치기
    if len(datasets) > 1:
        print(f"\n🔗 Combining datasets...")
        full_dataset = ConcatDataset(datasets)
    else:
        full_dataset = datasets[0]
    
    # ⭐ Train/Val split
    if args.val_split > 0:
        train_size = int(len(full_dataset) * (1 - args.val_split))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        print(f"\n📊 Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    else:
        train_dataset = full_dataset
        val_dataset = None
        print(f"\n📊 Total training samples: {len(train_dataset)}")
    
    print(f"📦 Batch size: {args.batch_size}")
    print(f"🔄 Batches per epoch: ~{len(train_dataset) // args.batch_size}")
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn_filter_empty,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        drop_last=True
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn_filter_empty,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0)
        )
    
    # num_classes 계산
    if isinstance(full_dataset, ConcatDataset):
        base_dataset = full_dataset.datasets[0]
    else:
        base_dataset = full_dataset
    
    num_classes = len(base_dataset.IDX_TO_CLASS) + 1  # background 포함
    print(f"🎯 Num classes (with background): {num_classes}")
    print(f"   Classes: {base_dataset.IDX_TO_CLASS}")
    
    # 모델 생성
    model = get_model(num_classes=num_classes)
    model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step, gamma=0.1
    )
    
    # 학습
    print("\n" + "="*60)
    print("🚀 Starting Training")
    print("="*60 + "\n")
    
    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float('inf')
    
    for epoch in range(1, args.num_epochs + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch
        )
        lr_scheduler.step()
        
        # Validation
        if val_loader and epoch % args.val_interval == 0:
            val_loss = validate(model, val_loader, device, epoch)
            
            # Best model 저장
            if val_loss < best_val_loss and val_loss > 0:
                best_val_loss = val_loss
                save_path = os.path.join(args.output_dir, "best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, save_path)
                print(f"  💾 Best model saved! (val_loss: {val_loss:.4f})")
        
        # 정기 저장
        if epoch % args.save_interval == 0:
            save_path = os.path.join(args.output_dir, f"model_epoch{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
            }, save_path)
            print(f"  💾 Saved: {save_path}\n")
    
    print("\n✅ Training completed!")
    
    # 최종 통계
    if val_loader:
        print(f"\n📊 Final Results:")
        print(f"   Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Mask R-CNN for medical segmentation')
    
    # 데이터 경로
    parser.add_argument('--image-root', type=str,
                    default='/Users/admin/Downloads/datasets/1.Training/1.원천데이터',
                    help='Path to image root directory')
    parser.add_argument('--label-root', type=str,
                    default='/Users/admin/Downloads/datasets/1.Training/2.라벨링데이터',
                    help='Path to label root directory')
    
    # 학습 설정
    parser.add_argument('--organ', type=str, choices=['colon', 'stomach', 'both'],
                    default='stomach',
                    help='Which organ to train (colon/stomach/both)')
    
    # ⭐ 데이터 제한 및 크기
    parser.add_argument('--max-samples', type=int, default=500,
                    help='Maximum samples per organ (default: 500)')
    parser.add_argument('--img-size', type=int, default=384,
                    help='Resize images to this size (default: 384)')
    parser.add_argument('--val-split', type=float, default=0.15,
                    help='Validation split ratio (default: 0.15 = 15%)')
    
    # 학습 파라미터
    parser.add_argument('--batch-size', type=int, default=8,
                    help='Batch size (default: 2)')
    parser.add_argument('--num-epochs', type=int, default=10,
                    help='Number of epochs (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                    help='Weight decay (default: 1e-4)')
    parser.add_argument('--lr-step', type=int, default=15,
                    help='LR scheduler step size (default: 15)')
    
    # 기타
    parser.add_argument('--num-workers', type=int, default=4,
                    help='Number of data loading workers (default: 4)')
    parser.add_argument('--save-interval', type=int, default=5,
                    help='Save model every N epochs (default: 5)')
    parser.add_argument('--val-interval', type=int, default=5,
                    help='Validate every N epochs (default: 5)')
    parser.add_argument('--min-area', type=int, default=100,
                    help='Minimum annotation area (default: 100)')
    parser.add_argument('--output-dir', type=str, default='outputs/stomach_seg',
                    help='Output directory for models')
    
    args = parser.parse_args()
    
    # 설정 출력
    print("\n" + "="*60)
    print("Training Configuration")
    print("="*60)
    print(f"Organ: {args.organ}")
    print(f"Max samples per organ: {args.max_samples}")
    print(f"Image size: {args.img_size}x{args.img_size}")
    print(f"Validation split: {args.val_split*100:.0f}%")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Output: {args.output_dir}")
    print("="*60 + "\n")
    
    main(args)