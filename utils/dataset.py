import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset

class MedicalFolderDataset(Dataset):
    def __init__(self, image_root, label_root, transforms=None):
        self.image_root = image_root
        self.label_root = label_root
        self.transforms = transforms
        self.imgs = sorted([f for f in os.listdir(image_root) if not f.startswith('.')])
        self.labels = sorted([f for f in os.listdir(label_root) if not f.startswith('.')])

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.image_root, self.imgs[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.label_root, self.labels[idx]), 0)

        obj_ids = np.unique(mask)[1:] # 0(배경) 제외
        masks = (mask == obj_ids[:, None, None]).astype(np.uint8)

        boxes = []
        for i in range(len(obj_ids)):
            pos = np.where(masks[i])
            boxes.append([np.min(pos[1]), np.min(pos[0]), np.max(pos[1]), np.max(pos[0])])

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(obj_ids, dtype=torch.int64),
            "masks": torch.as_tensor(masks, dtype=torch.uint8),
            "image_id": torch.tensor([idx])
        }

        if self.transforms:
            img = self.transforms(img)
        return img, target

    def __len__(self):
        return len(self.imgs)

def collate_fn(batch):
    return tuple(zip(*batch))