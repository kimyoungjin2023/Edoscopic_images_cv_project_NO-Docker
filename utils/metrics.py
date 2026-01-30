import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from models.multitask import MultiTaskMaskRCNN
from utils.dataset import MedicalFolderDataset, collate_fn
from utils.metrics import calculate_dice

@torch.no_grad()
def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 모델 로드 (배경 포함 4클래스)
    model = MultiTaskMaskRCNN(num_classes=4).to(device)
    model.load_state_dict(torch.load('models/checkpoints/model_ep10.pth')) # 최신 체크포인트
    model.eval()

    # 2. 검증 데이터셋 설정
    transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    val_ds = MedicalFolderDataset('data/val_imgs', 'data/val_labels', transform)
    loader = DataLoader(val_ds, batch_size=2, collate_fn=collate_fn)

    dice_results = []
    print("🔍 Evaluating model performance on 'yysop-dev'...")

    for images, targets in loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for i, output in enumerate(outputs):
            if len(output['masks']) > 0:
                # Segmentation 성능 측정
                pred = (output['masks'][0, 0] > 0.5).cpu().numpy().astype(bool)
                true = targets[i]['masks'][0].cpu().numpy().astype(bool)
                dice_results.append(calculate_dice(pred, true))
            
            # Detection(Box) 결과는 필요 시 여기서 시각화하거나 mAP 계산 로직 추가

    print(f"\n📊 Final Mean Dice Score: {np.mean(dice_results):.4f}")

if __name__ == "__main__":
    evaluate()