# utils/metrics.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedMaskLoss(nn.Module):
    def __init__(self, weight_dice=0.6):
        super().__init__()
        self.weight_dice = weight_dice

    def dice_loss(self, inputs, targets, smooth=1e-6):
        inputs = torch.sigmoid(inputs).view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice

    def forward(self, mask_logits, targets):
        bce = F.binary_cross_entropy_with_logits(mask_logits, targets)
        dice = self.dice_loss(mask_logits, targets)
        return (1 - self.weight_dice) * bce + self.weight_dice * dice