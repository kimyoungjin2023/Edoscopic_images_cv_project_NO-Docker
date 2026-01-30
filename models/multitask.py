# multitask

import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

class MultiTaskMaskRCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(MultiTaskMaskRCNN, self).__init__()
        
        # 1. COCO 데이터로 미리 학습된 기본 모델 로드
        # ResNet-50을 뼈대(Backbone)로 사용하는 표준 Mask R-CNN
        self.model = maskrcnn_resnet50_fpn(weights="DEFAULT")

        # 2. [Detection] 사각형 박스를 그리는 부분(Box Predictor) 수정
        # 기본 모델은 클래스가 91개용이므로, 우리 데이터(4개)에 맞춰 입력을 변경
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # 3. [Segmentation] 픽셀을 따는 부분(Mask Predictor) 수정
        # 마스크를 생성하는 레이어의 채널 수를 우리 클래스 수에 맞게 맞춤
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    def forward(self, images, targets=None):
        """
        학습 모드: images와 targets를 모두 넣어 Loss(손실값) 딕셔너리를 반환받음
        추론 모드: images만 넣어 예측 결과(Boxes, Labels, Masks) 리스트를 반환받음
        """
        return self.model(images, targets)