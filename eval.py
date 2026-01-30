import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedMaskLoss(nn.Module):
    """BCE + Dice 결합 손실 함수"""
    def __init__(self, weight_dice=0.6):
        super().__init__()
        self.weight_dice = weight_dice

    def dice_loss(self, inputs, targets, smooth=1e-6):
        inputs = torch.sigmoid(inputs).view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        return 1 - ((2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth))

    def forward(self, mask_logits, targets):
        bce = F.binary_cross_entropy_with_logits(mask_logits, targets)
        dice = self.dice_loss(mask_logits, targets)
        return (1 - self.weight_dice) * bce + self.weight_dice * dice

def calculate_dice(pred_mask, true_mask):
    """픽셀 단위 Dice Score 계산"""
    intersection = (pred_mask & true_mask).sum()
    total = pred_mask.sum() + true_mask.sum()
    return (2. * intersection) / (total + 1e-6)