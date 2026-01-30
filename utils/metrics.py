# metrics

import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedMaskLoss(nn.Module):
    """
    [설계안 반영] BCE + Dice 결합 손실 함수
    - BCE: 픽셀 하나하나의 정답 여부를 판단 (안정적인 초기 학습)
    - Dice: 전체 영역의 겹침 정도를 판단 (불균형한 의료 데이터에 효과적)
    """
    def __init__(self, weight_dice=0.6):
        super().__init__()
        self.weight_dice = weight_dice

    def dice_loss(self, inputs, targets, smooth=1e-6):
        # 0~1 사이 확률값으로 변환 후 1차원으로 펼침
        inputs = torch.sigmoid(inputs).view(-1)
        targets = targets.view(-1)
        
        # 교집합 계산
        intersection = (inputs * targets).sum()
        
        # Dice 계수 공식 기반 Loss 산출 (1 - Dice)
        dice_coeff = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice_coeff

    def forward(self, mask_logits, targets):
        # 1. 픽셀 단위 오차(BCE)와 영역 단위 오차(Dice)를 각각 계산
        bce = F.binary_cross_entropy_with_logits(mask_logits, targets)
        dice = self.dice_loss(mask_logits, targets)
        
        # 2. 두 손실을 가중치(weight_dice)에 따라 결합하여 최종 반환
        return (1 - self.weight_dice) * bce + self.weight_dice * dice

def calculate_dice(pred_mask, true_mask):
    """
    [평가지표] 실제 픽셀 단위 Dice Score 계산 (Numpy/Boolean 기반)
    - pred_mask: 모델의 예측 마스크 (True/False)
    - true_mask: 실제 정답 마스크 (True/False)
    """
    intersection = (pred_mask & true_mask).sum()
    total = pred_mask.sum() + true_mask.sum()
    
    # 분모가 0이 되는 것을 방지하기 위해 아주 작은 값(1e-6) 추가
    return (2. * intersection) / (total + 1e-6)