import torchvision.transforms as T

def get_transforms():
    return T.Compose([
        T.ToTensor(),
        T.Resize((256,256)),
        T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])
