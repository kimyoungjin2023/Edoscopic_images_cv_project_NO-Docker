# train_medical_segmentation.py
import torch
import os
import argparse
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from datasets.medical_folder_dataset import MedicalFolderDataset, collate_fn_filter_empty
from tqdm import tqdm

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
    
    loss_components = {
        'loss_classifier': 0,
        'loss_box_reg': 0,
        'loss_mask': 0,
        'loss_objectness': 0,
        'loss_rpn_box_reg': 0
    }

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        # 빈 batch 스킵
        if batch is None:
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
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        except Exception as e:
            print(f"\n❌ Error at batch {batch_idx}: {e}")
            continue

    if num_batches == 0:
        print("⚠️ No valid batches in this epoch!")
        return 0.0
    
    avg_loss = epoch_loss / num_batches
    
    print(f"\n[Epoch {epoch}] Avg Loss: {avg_loss:.4f}")
    if epoch % 5 == 0:
        print("  Loss components:")
        for k, v in loss_components.items():
            print(f"    {k}: {v/num_batches:.4f}")
    
    return avg_loss


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    transform = transforms.ToTensor()
    
    # 데이터셋 로드
    datasets = []
    
    if args.organ == 'colon' or args.organ == 'both':
        print("="*60)
        print("Loading COLON dataset...")
        print("="*60)
        colon_dataset = MedicalFolderDataset(
            image_root=args.image_root,
            label_root=args.label_root,
            organ_type='대장',
            transforms=transform,
            min_area=args.min_area
        )
        datasets.append(colon_dataset)
    
    if args.organ == 'stomach' or args.organ == 'both':
        print("\n" + "="*60)
        print("Loading STOMACH dataset...")
        print("="*60)
        stomach_dataset = MedicalFolderDataset(
            image_root=args.image_root,
            label_root=args.label_root,
            organ_type='위',
            transforms=transform,
            min_area=args.min_area
        )
        datasets.append(stomach_dataset)
    
    # 데이터셋 합치기
    if len(datasets) > 1:
        print(f"\n🔗 Combining datasets...")
        train_dataset = ConcatDataset(datasets)
    else:
        train_dataset = datasets[0]
    
    print(f"\n📊 Total training samples: {len(train_dataset)}")
    
    # DataLoader
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn_filter_empty
    )
    
    # 모델 생성 (background(0) + ulcer(1) + cancer(2) + tumor(3))
    model = get_model(num_classes=4)
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
    print("Starting training...")
    print("="*60 + "\n")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for epoch in range(1, args.num_epochs + 1):
        avg_loss = train_one_epoch(model, dataloader, optimizer, device, epoch)
        lr_scheduler.step()
        
        # 모델 저장
        if epoch % args.save_interval == 0:
            save_path = os.path.join(args.output_dir, f"model_epoch{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path)
            print(f"  💾 Saved: {save_path}\n")
    
    print("✅ Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # 데이터 경로
    parser.add_argument('--image-root', type=str,
                       default='/Users/admin/Downloads/datasets/1.Training/1.원천데이터')
    parser.add_argument('--label-root', type=str,
                       default='/Users/admin/Downloads/datasets/1.Training/2.라벨링데이터')
    
    # 학습 설정
    parser.add_argument('--organ', type=str, choices=['colon', 'stomach', 'both'],
                       default='colon')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--lr-step', type=int, default=10)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--save-interval', type=int, default=5)
    parser.add_argument('--min-area', type=int, default=100)
    parser.add_argument('--output-dir', type=str, default='outputs/colon_seg')
    
    args = parser.parse_args()
    main(args)