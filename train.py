import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from models.multitask import MultiTaskMaskRCNN
from utils.dataset import MedicalFolderDataset, collate_fn
from tqdm import tqdm

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    
    # 데이터 경로 설정
    ds = MedicalFolderDataset('data/1.원천데이터', 'data/2.라벨링데이터', transform)
    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_fn)

    model = MultiTaskMaskRCNN(num_classes=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    print("🚀 Training Start on yysop-dev")
    for epoch in range(1, 11):
        model.train()
        for imgs, tars in tqdm(loader, desc=f"Epoch {epoch}"):
            imgs = [i.to(device) for i in imgs]
            tars = [{k: v.to(device) for k, v in t.items()} for t in tars]
            
            loss_dict = model(imgs, tars)
            loss = sum(l for l in loss_dict.values())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), f"models/checkpoints/model_ep{epoch}.pth")

if __name__ == "__main__":
    train()