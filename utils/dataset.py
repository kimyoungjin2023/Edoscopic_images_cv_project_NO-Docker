import torch
from torch.utils.data import Dataset
import cv2
import os

class EndoscopyDataset(Dataset):
    def __init__(self, img_dir, ann_dir, transform=None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transform = transform
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        ann_path = os.path.join(self.ann_dir, self.images[idx].replace('.jpg', '.json'))

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # segmentation mask 로드 (예시: PNG)
        mask_path = img_path.replace('.jpg', '_mask.png')
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # detection bbox 로드 (예시: JSON)
        # 실제 구현에서는 json 파싱 필요
        bboxes = torch.tensor([[50, 50, 150, 150]])  # dummy bbox

        if self.transform:
            image = self.transform(image)

        return image, mask, bboxes
