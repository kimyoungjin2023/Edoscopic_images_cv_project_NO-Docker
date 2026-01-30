import torch
from models.multitask import MultiTaskModel
from utils.dataset import EndoscopyDataset
from utils.transforms import get_transforms
from utils.metrics import dice_coeff, iou

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = EndoscopyDataset(img_dir="data/images", ann_dir="data/annotations", transform=get_transforms())
model = MultiTaskModel(num_classes=2).to(device)
model.load_state_dict(torch.load("checkpoint.pth"))
model.eval()

dice_scores, iou_scores = [], []
with torch.no_grad():
    for images, masks, bboxes in dataset:
        images, masks = images.unsqueeze(0).to(device), masks.to(device)
        seg_out, _, _ = model(images, bboxes.unsqueeze(0).to(device))
