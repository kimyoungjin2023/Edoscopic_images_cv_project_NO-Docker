# models/multitask.py
import torch.nn as nn
from models.unet import UNet
from models.fast_rcnn import FastRCNN

class MultiTaskModel(nn.Module):
    def __init__(self, num_classes=2):
        super(MultiTaskModel, self).__init__()
        self.segmentation = UNet(in_channels=3, out_channels=1)
        self.detection = FastRCNN(num_classes=num_classes)

    def forward(self, images, rois):
        seg_out = self.segmentation(images)
        cls_logits, bbox_preds = self.detection(images, rois)
        return seg_out, cls_logits, bbox_preds
