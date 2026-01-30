"""
Mask R-CNN with ResNet50 Backbone
================================================================
구조:
1. Backbone: ResNet50 (Conv layers only, no FC/MLP)
2. FPN: Feature Pyramid Network
3. RPN: Region Proposal Network
4. RoIAlign: Region of Interest Align
5. Heads: Box Head (detection) + Mask Head (segmentation)

이 모델은 Classification, Detection, Segmentation을 동시에 수행합니다.
================================================================
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import MultiScaleRoIAlign
from collections import OrderedDict


def get_maskrcnn_model(num_classes, pretrained_backbone=True, trainable_backbone_layers=5):
    """
    ResNet50 backbone을 사용하는 Mask R-CNN 모델 생성 (권장 방법)
    
    Args:
        num_classes (int): 클래스 개수 (배경 포함)
        pretrained_backbone (bool): ImageNet pretrained weights 사용 여부
        trainable_backbone_layers (int): 학습 가능한 backbone layer 수 (0-5)
    
    Returns:
        model: Mask R-CNN 모델
    """
    # ResNet50 + FPN backbone 생성
    backbone = resnet_fpn_backbone(
        'resnet50',
        pretrained=pretrained_backbone,
        trainable_layers=trainable_backbone_layers
    )
    
    # RoI Pooling for box head
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],  # FPN levels: P2, P3, P4, P5
        output_size=7,
        sampling_ratio=2
    )
    
    # RoI Pooling for mask head
    mask_roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=14,
        sampling_ratio=2
    )
    
    # Mask R-CNN 모델 생성
    model = MaskRCNN(
        backbone,
        num_classes=num_classes,
        box_roi_pool=roi_pooler,
        mask_roi_pool=mask_roi_pooler,
        # RPN parameters
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
        # Box parameters
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512,
        box_positive_fraction=0.25,
    )
    
    # Box predictor 교체 (classification + bbox regression)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Mask predictor 교체 (segmentation)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )
    
    return model


class ExplicitMaskRCNN(nn.Module):
    """
    명시적 구조를 보여주는 Mask R-CNN
    (교육 목적 - 실제 사용시에는 get_maskrcnn_model 권장)
    
    구조를 명확히 보여주기 위해 각 컴포넌트를 분리하여 정의
    """
    
    def __init__(self, num_classes, pretrained_backbone=True):
        super(ExplicitMaskRCNN, self).__init__()
        
        self.num_classes = num_classes
        
        # ==================== 1. Backbone: ResNet50 ====================
        print("Building Backbone: ResNet50 (Conv layers only)...")
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained_backbone else None)
        
        # Conv layers만 사용 (FC layer 제거)
        self.backbone_conv1 = resnet.conv1
        self.backbone_bn1 = resnet.bn1
        self.backbone_relu = resnet.relu
        self.backbone_maxpool = resnet.maxpool
        
        # ResNet Blocks
        self.backbone_layer1 = resnet.layer1  # C2: 256 channels, stride 4
        self.backbone_layer2 = resnet.layer2  # C3: 512 channels, stride 8
        self.backbone_layer3 = resnet.layer3  # C4: 1024 channels, stride 16
        self.backbone_layer4 = resnet.layer4  # C5: 2048 channels, stride 32
        
        # ==================== 2. FPN (Feature Pyramid Network) ====================
        print("Building FPN...")
        # Lateral connections
        self.fpn_lateral5 = nn.Conv2d(2048, 256, kernel_size=1)
        self.fpn_lateral4 = nn.Conv2d(1024, 256, kernel_size=1)
        self.fpn_lateral3 = nn.Conv2d(512, 256, kernel_size=1)
        self.fpn_lateral2 = nn.Conv2d(256, 256, kernel_size=1)
        
        # Output convolutions
        self.fpn_output5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fpn_output4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fpn_output3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fpn_output2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        # ==================== 3. RPN (Region Proposal Network) ====================
        print("Building RPN...")
        self.rpn_conv = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # 9 anchors per location (3 scales × 3 aspect ratios)
        self.rpn_cls_logits = nn.Conv2d(256, 9, kernel_size=1)
        self.rpn_bbox_pred = nn.Conv2d(256, 9 * 4, kernel_size=1)
        
        print(f"Model initialized with {num_classes} classes")
        print(f"  - Backbone: ResNet50 (pretrained={pretrained_backbone})")
        print(f"  - FPN: 4 levels (P2, P3, P4, P5)")
        print(f"  - RPN: 9 anchors per location")
    
    def forward_backbone(self, x):
        """
        ResNet50 Backbone Forward
        
        Args:
            x: (B, 3, H, W) input images
        
        Returns:
            c2, c3, c4, c5: Feature maps at different scales
        """
        # Initial conv
        x = self.backbone_conv1(x)
        x = self.backbone_bn1(x)
        x = self.backbone_relu(x)
        x = self.backbone_maxpool(x)
        
        # ResNet stages
        c2 = self.backbone_layer1(x)    # 1/4 resolution
        c3 = self.backbone_layer2(c2)   # 1/8 resolution
        c4 = self.backbone_layer3(c3)   # 1/16 resolution
        c5 = self.backbone_layer4(c4)   # 1/32 resolution
        
        return c2, c3, c4, c5
    
    def forward_fpn(self, c2, c3, c4, c5):
        """
        FPN Forward (Top-down pathway)
        
        Args:
            c2, c3, c4, c5: Feature maps from backbone
        
        Returns:
            p2, p3, p4, p5: FPN feature pyramids
        """
        # Top-down pathway
        p5 = self.fpn_lateral5(c5)
        p4 = self.fpn_lateral4(c4) + nn.functional.interpolate(
            p5, scale_factor=2, mode='nearest'
        )
        p3 = self.fpn_lateral3(c3) + nn.functional.interpolate(
            p4, scale_factor=2, mode='nearest'
        )
        p2 = self.fpn_lateral2(c2) + nn.functional.interpolate(
            p3, scale_factor=2, mode='nearest'
        )
        
        # Output convolutions (reduce aliasing)
        p5 = self.fpn_output5(p5)
        p4 = self.fpn_output4(p4)
        p3 = self.fpn_output3(p3)
        p2 = self.fpn_output2(p2)
        
        return p2, p3, p4, p5
    
    def forward_rpn(self, features):
        """
        RPN Forward
        
        Args:
            features: List of feature maps [p2, p3, p4, p5]
        
        Returns:
            objectness, bbox_deltas for each feature level
        """
        rpn_outputs = []
        
        for feature in features:
            # Shared conv
            t = self.rpn_conv(feature)
            t = nn.functional.relu(t)
            
            # Classification (objectness)
            objectness = self.rpn_cls_logits(t)
            
            # Bbox regression
            bbox_deltas = self.rpn_bbox_pred(t)
            
            rpn_outputs.append({
                'objectness': objectness,
                'bbox_deltas': bbox_deltas
            })
        
        return rpn_outputs
    
    def forward(self, images, targets=None):
        """
        Forward pass
        
        Note: 실제 학습/추론을 위해서는 torchvision의 MaskRCNN을 사용하세요.
        이 구현은 구조를 명확히 보여주기 위한 교육 목적입니다.
        """
        # 1. Backbone
        c2, c3, c4, c5 = self.forward_backbone(images)
        
        # 2. FPN
        p2, p3, p4, p5 = self.forward_fpn(c2, c3, c4, c5)
        features = [p2, p3, p4, p5]
        
        # 3. RPN
        rpn_outputs = self.forward_rpn(features)
        
        # Note: RoI Heads (Box + Mask) 부분은 torchvision 구현이 복잡하므로
        # 실제 사용시에는 get_maskrcnn_model() 함수를 사용하세요.
        
        return {
            'features': features,
            'rpn_outputs': rpn_outputs
        }


def print_model_structure(model):
    """모델 구조 출력"""
    print("\n" + "="*60)
    print("Mask R-CNN Model Structure")
    print("="*60)
    
    # Backbone
    print("\n1. BACKBONE (ResNet50):")
    print(f"   - conv1: {model.backbone.body.conv1}")
    print(f"   - layer1 (C2): {len(model.backbone.body.layer1)} blocks")
    print(f"   - layer2 (C3): {len(model.backbone.body.layer2)} blocks")
    print(f"   - layer3 (C4): {len(model.backbone.body.layer3)} blocks")
    print(f"   - layer4 (C5): {len(model.backbone.body.layer4)} blocks")
    
    # FPN
    print("\n2. FPN (Feature Pyramid Network):")
    print(f"   - fpn_inner4: {model.backbone.fpn.inner_blocks[3]}")
    print(f"   - fpn_layer4: {model.backbone.fpn.layer_blocks[3]}")
    print(f"   - Output levels: P2, P3, P4, P5")
    
    # RPN
    print("\n3. RPN (Region Proposal Network):")
    print(f"   - head: {model.rpn.head}")
    print(f"   - anchor_generator: {model.rpn.anchor_generator}")
    
    # RoI Heads
    print("\n4. RoI HEADS:")
    print(f"   - box_roi_pool: {model.roi_heads.box_roi_pool}")
    print(f"   - box_head: {model.roi_heads.box_head}")
    print(f"   - box_predictor: {model.roi_heads.box_predictor}")
    print(f"   - mask_roi_pool: {model.roi_heads.mask_roi_pool}")
    print(f"   - mask_head: {model.roi_heads.mask_head}")
    print(f"   - mask_predictor: {model.roi_heads.mask_predictor}")
    
    # 파라미터 수
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n5. PARAMETERS:")
    print(f"   - Total: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"   - Trainable: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    print("="*60)
    print("Mask R-CNN Model Module Test")
    print("="*60)
    
    # 모델 생성
    num_classes = 7  # AI Hub 내시경: 6 classes + background
    print(f"\nCreating Mask R-CNN with {num_classes} classes...")
    
    model = get_maskrcnn_model(
        num_classes=num_classes,
        pretrained_backbone=True,
        trainable_backbone_layers=5
    )
    
    # 구조 출력
    print_model_structure(model)
    
    # 테스트 입력
    print("Testing forward pass...")
    model.eval()
    
    dummy_images = [torch.randn(3, 512, 512), torch.randn(3, 512, 512)]
    
    with torch.no_grad():
        outputs = model(dummy_images)
    
    print(f"\n✓ Forward pass successful!")
    print(f"  Number of images: {len(dummy_images)}")
    print(f"  Output keys: {outputs[0].keys()}")
    
    # 학습 모드 테스트
    print("\nTesting training mode...")
    model.train()
    
    dummy_targets = [
        {
            'boxes': torch.FloatTensor([[10, 10, 100, 100]]),
            'labels': torch.LongTensor([1]),
            'masks': torch.zeros((1, 512, 512), dtype=torch.uint8)
        },
        {
            'boxes': torch.FloatTensor([[20, 20, 120, 120]]),
            'labels': torch.LongTensor([2]),
            'masks': torch.zeros((1, 512, 512), dtype=torch.uint8)
        }
    ]
    
    loss_dict = model(dummy_images, dummy_targets)
    print(f"\n✓ Training mode successful!")
    print(f"  Loss keys: {loss_dict.keys()}")
    for k, v in loss_dict.items():
        print(f"    {k}: {v.item():.4f}")
    
    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)
    
    print("\nUsage Example:")
    print("""
    from models.mask_rcnn import get_maskrcnn_model
    
    # 모델 생성
    model = get_maskrcnn_model(num_classes=7, pretrained_backbone=True)
    model.to(device)
    
    # 학습 모드
    model.train()
    loss_dict = model(images, targets)
    
    # 추론 모드
    model.eval()
    with torch.no_grad():
        predictions = model(images)
    """)
