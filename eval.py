# eval

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from models.multitask import MultiTaskMaskRCNN
from utils.dataset import MedicalFolderDataset, collate_fn
from utils.metrics import calculate_dice  # metrics.py에서 계산 함수 호출

@torch.no_grad() # 평가 시에는 그래디언트 계산을 꺼서 메모리를 절약함
def evaluate():
    # GPU 사용 가능 여부 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 모델 아키텍처 로드 및 가중치 파일(.pth) 불러오기
    # 배경을 포함한 총 클래스 수(4)를 일치시켜야 합니다.
    model = MultiTaskMaskRCNN(num_classes=4).to(device)
    model.load_state_dict(torch.load('models/checkpoints/model_ep10.pth')) 
    model.eval() # 모델을 평가 모드로 전환 (Dropout, Batchnorm 등 고정)

    # 2. 검증용 데이터셋 설정 (증강 없이 Resize와 Tensor 변환만 적용)
    transform = transforms.Compose([
        transforms.Resize((512, 512)), 
        transforms.ToTensor()
    ])
    
    # 검증 데이터 경로 설정 (본인 폴더명에 맞게 확인 필요)
    val_ds = MedicalFolderDataset('data/val_imgs', 'data/val_labels', transform)
    loader = DataLoader(val_ds, batch_size=2, collate_fn=collate_fn)

    dice_results = []
    print(f"🔍 [yysop-dev] {device} 환경에서 모델 평가를 시작합니다...")

    for images, targets in loader:
        images = [img.to(device) for img in images]
        outputs = model(images) # 모델 예측 수행

        for i, output in enumerate(outputs):
            # 모델이 마스크를 예측했을 경우에만 점수 계산
            if len(output['masks']) > 0:
                # 0.5 임계값을 기준으로 예측 영역(True)과 배경(False) 구분
                pred = (output['masks'][0, 0] > 0.5).cpu().numpy().astype(bool)
                # 정답 마스크도 불리언 타입으로 변환
                true = targets[i]['masks'][0].cpu().numpy().astype(bool)
                
                # Dice Score 계산 및 결과 리스트 저장
                score = calculate_dice(pred, true)
                dice_results.append(score)
            
            # Tip: 박스 탐지 성능(mAP)을 보고 싶다면 output['boxes']와 output['scores'] 활용 가능

    # 최종 결과 출력 (전체 이미지에 대한 평균 Dice 점수)
    final_score = np.mean(dice_results) if dice_results else 0
    print(f"\n📊 [Evaluation Result] Mean Dice Score: {final_score:.4f}")

if __name__ == "__main__":
    evaluate()