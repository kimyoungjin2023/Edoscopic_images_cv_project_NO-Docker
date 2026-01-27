# models/fast_rcnn.py
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import RoIAlign

class FastRCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(FastRCNN, self).__init__()
        backbone = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
        self.roi_align = RoIAlign(output_size=(7,7), spatial_scale=1/16, sampling_ratio=-1)
        self.fc = nn.Sequential(
            nn.Linear(2048*7*7, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )
        self.bbox_regressor = nn.Linear(1024, num_classes*4)

    def forward(self, images, rois):
        features = self.feature_extractor(images)
        pooled = self.roi_align(features, rois)
        pooled = pooled.view(pooled.size(0), -1)
        cls_logits = self.fc(pooled)
        bbox_preds = self.bbox_regressor(pooled)
        return cls_logits, bbox_preds
