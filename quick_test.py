# check_validation.py
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from datasets.medical_folder_dataset import MedicalFolderDataset, collate_fn_filter_empty

transform = transforms.ToTensor()

dataset = MedicalFolderDataset(
    image_root='/Users/admin/Downloads/datasets/1.Training/1.원천데이터',
    label_root='/Users/admin/Downloads/datasets/1.Training/2.라벨링데이터',
    organ_type='대장',
    transforms=transform,
    resize=(384, 384),
    max_samples=500
)

# Train/Val split
train_size = int(len(dataset) * 0.9)
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

# Validation 샘플 체크
val_loader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=collate_fn_filter_empty
)

valid_batches = 0
for batch in val_loader:
    if batch is not None:
        valid_batches += 1

print(f"Valid validation batches: {valid_batches} / {len(val_loader)}")