import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from models.multitask import MultiTaskMaskRCNN
from utils.dataset import MedicalFolderDataset, collate_fn
from tqdm import tqdm # 학습 진행률을 보여주는 라이브러리

def train():
    # GPU 사용이 가능하면 cuda, 아니면 cpu를 장치로 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 모델 가중치를 저장할 폴더 생성
    os.makedirs('models/checkpoints', exist_ok=True)

    # 1. 전처리 설정: 이 버전은 증강 없이 기본 변환만 수행
    # Resize: 512x512 크기로 맞춤 / ToTensor: 0~255 값을 0~1 사이의 텐서로 변환
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    # 2. 데이터셋 및 데이터로더 설정
    # collate_fn: 이미지마다 병변 개수가 달라도 배치로 묶을 수 있게 도와줌
    dataset = MedicalFolderDataset(
        image_root='data/1.원천데이터', 
        label_root='data/2.라벨링데이터',
        transforms=transform
    )
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    # 3. 모델 초기화 (배경 포함 클래스 4개) 및 GPU 전송
    model = MultiTaskMaskRCNN(num_classes=4).to(device)
    
    # 4. 최적화 알고리즘 (AdamW): 오차를 줄이기 위해 모델의 가중치를 수정하는 역할
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    print(f"🚀 [yysop-dev] No-Aug Baseline 학습 시작 (Device: {device})")
    
    for epoch in range(1, 11): # 10번 반복 학습
        model.train() # 모델을 학습 모드로 설정
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for images, targets in pbar:
            # 이미지와 정답 데이터를 GPU로 전송
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # 5. 모델 예측 및 손실 계산
            # Mask R-CNN은 내부적으로 Classifier, Box, Mask Loss를 모두 계산해서 줌
            loss_dict = model(images, targets)
            total_loss = sum(loss for loss in loss_dict.values()) # 모든 손실의 합

            # 6. 역전파 (Backpropagation)
            optimizer.zero_grad()  # 이전 루프의 기울기 초기화
            total_loss.backward()  # 현재 오차로 기울기 계산
            optimizer.step()       # 가중치 업데이트

            epoch_loss += total_loss.item()
            
            # 실시간으로 전체 손실값과 마스크 전용 손실값을 표시
            pbar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'Mask': f'{loss_dict["loss_mask"].item():.4f}'
            })

        # 에폭이 끝날 때마다 모델 상태 저장
        torch.save(model.state_dict(), f"models/checkpoints/baseline_ep{epoch}.pth")

if __name__ == "__main__":
    train()