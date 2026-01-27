import torch
import cv2
import argparse
from models.multitask import MultiTaskModel
from utils.transforms import get_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(checkpoint_path, num_classes=2):
    model = MultiTaskModel(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def run_inference(model, image_path):
    # 이미지 로드
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = get_transforms()
    tensor_img = transform(image_rgb).unsqueeze(0).to(device)

    # 더미 ROI (실제 구현에서는 detection 후보 영역 필요)
    rois = torch.tensor([[[0, 0, image.shape[1], image.shape[0]]]], dtype=torch.float).to(device)

    with torch.no_grad():
        seg_out, cls_logits, bbox_preds = model(tensor_img, rois)

    # Segmentation 결과 후처리
    seg_mask = (seg_out.squeeze().cpu().numpy() > 0.5).astype("uint8") * 255
    cv2.imwrite("segmentation_result.png", seg_mask)

    # Detection 결과 후처리 (여기서는 더미 bbox)
    pred_class = torch.argmax(cls_logits, dim=1).item()
    bbox = bbox_preds[0].cpu().numpy()

    # bbox 그리기
    x1, y1, x2, y2 = map(int, bbox[:4])
    cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.putText(image, f"Class {pred_class}", (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    cv2.imwrite("detection_result.png", image)

    print("Inference complete! Results saved as segmentation_result.png and detection_result.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pth", help="Path to model checkpoint")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    run_inference(model, args.image)

# python inference.py --checkpoint checkpoint.pth --image data/test/sample.jpg