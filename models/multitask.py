# models/multitask.py
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

class MultiTaskMaskRCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(MultiTaskMaskRCNN, self).__init__()
        # Pre-trained Mask R-CNN 로드
        self.model = maskrcnn_resnet50_fpn(weights="DEFAULT")

        # [Detection] Head 수정: 클래스 수에 맞춤
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # [Segmentation] Head 수정: 클래스 수에 맞춤
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

    def forward(self, images, targets=None):
        # 학습 시에는 loss를, 추론 시에는 결과값을 반환함
        return self.model(images, targets)