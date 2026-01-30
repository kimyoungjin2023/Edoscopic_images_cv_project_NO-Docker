import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

class MultiTaskMaskRCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(MultiTaskMaskRCNN, self).__init__()
        # 사전 학습된 Mask R-CNN 로드
        self.model = maskrcnn_resnet50_fpn(weights="DEFAULT")

        # [Detection] Head 수정: Classification + Box Regression
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # [Segmentation] Head 수정: Mask Prediction
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

    def forward(self, images, targets=None):
        # 학습 시 loss 반환, 추론 시 결과 반환
        return self.model(images, targets)