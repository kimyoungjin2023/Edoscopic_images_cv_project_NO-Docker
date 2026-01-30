import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.multitask import MultiTaskModel
from utils.dataset import EndoscopyDataset
from utils.transforms import get_transforms
from utils.metrics import dice_coeff, iou

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = EndoscopyDataset(img_dir="data/images", ann_dir="data/annotations", transform=get_transforms())
loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = MultiTaskModel(num_classes=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
seg_loss_fn = nn.BCELoss()
det_loss_fn = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    for images, masks, bboxes in loader:
        images, masks, bboxes = images.to(device), masks.to(device), bboxes.to(device)

        seg_out, cls_logits, bbox_preds = model(images, bboxes)

        loss_seg = seg_loss_fn(seg_out, masks.unsqueeze(1).float())
        loss_cls = det_loss_fn(cls_logits, torch.zeros(cls_logits.size(0), dtype=torch.long).to(device))
        loss = loss_seg + loss_cls

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
