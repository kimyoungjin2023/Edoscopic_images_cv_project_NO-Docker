# AI Hub 내시경 이미지 - Mask R-CNN 프로젝트

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

ResNet50 backbone을 사용하는 Mask R-CNN 모델로 AI Hub 내시경 이미지에 대해 **Classification, Object Detection, Instance Segmentation**을 동시에 수행하는 프로젝트입니다.

---

## 📋 목차

- [프로젝트 개요](#프로젝트-개요)
- [데이터셋 정보](#데이터셋-정보)
- [모델 구조](#모델-구조)
- [설치 방법](#설치-방법)
- [데이터 준비](#데이터-준비)
- [학습](#학습)
- [평가](#평가)
- [추론](#추론)
- [프로젝트 구조](#프로젝트-구조)

---

## 🎯 프로젝트 개요

### 목표
하나의 Mask R-CNN 모델로 내시경 이미지에서:
1. **Classification**: 병변 종류 분류 (궤양, 용종, 암)
2. **Object Detection**: 병변 위치 탐지 (Bounding Box)
3. **Instance Segmentation**: 병변 영역 분할 (Pixel-wise Mask)

### 특징
- ✅ ResNet50 backbone (Conv layers only, FC 제외)
- ✅ FPN (Feature Pyramid Network)
- ✅ RPN (Region Proposal Network)
- ✅ RoIAlign
- ✅ Multi-task learning (Detection + Segmentation)
- ✅ 의료 영상 특화 전처리
- ✅ 명확한 데이터 분할 (Train/Val/Test)

---

## 📊 데이터셋 정보

### AI Hub 내시경 이미지 합성 데이터셋

**출처**: [AI Hub - 내시경 이미지 합성데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71666)

### 데이터 구성
- **총 40,000장** (위 20,000장 + 대장 20,000장)
- **고해상도**: 2048 × 2048 pixels
- **Annotation**: COCO-style JSON (bbox + segmentation mask)

### 클래스 정의

| Class ID | Class Name | Description |
|----------|------------|-------------|
| 0 | background | 배경 |
| 1 | stomach_ulcer | 위 궤양 |
| 2 | stomach_polyp | 위 용종 |
| 3 | stomach_cancer | 위 암 |
| 4 | colon_ulcer | 대장 궤양 |
| 5 | colon_polyp | 대장 용종 |
| 6 | colon_cancer | 대장 암 |

### 데이터 분할

본 프로젝트는 다음과 같이 데이터를 분할합니다:

| Split | 클래스별 샘플 수 | 총 샘플 수 |
|-------|------------------|------------|
| **Train** | 1,000장 | 6,000장 |
| **Validation** | 150장 | 900장 |
| **Test** | 250-500장 | 1,500-3,000장 |

---

## 🏗 모델 구조

### Mask R-CNN Architecture

```
Input Image (3, H, W)
    ↓
┌─────────────────────────────────────────┐
│  1. Backbone: ResNet50                  │
│     - Conv layers only (No FC/MLP)      │
│     - Output: C2, C3, C4, C5            │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  2. FPN (Feature Pyramid Network)       │
│     - Top-down pathway                  │
│     - Lateral connections               │
│     - Output: P2, P3, P4, P5            │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  3. RPN (Region Proposal Network)       │
│     - 9 anchors per location            │
│     - Objectness + BBox regression      │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  4. RoIAlign                             │
│     - Fixed-size feature extraction     │
│     - Box Head: 7×7                     │
│     - Mask Head: 14×14                  │
└─────────────────────────────────────────┘
    ↓
┌────────────────────┬────────────────────┐
│  5a. Box Head      │  5b. Mask Head     │
│   - Classification │   - Segmentation   │
│   - BBox Regression│   - Pixel-wise     │
└────────────────────┴────────────────────┘
```

### 주요 구성 요소

1. **Backbone (ResNet50)**
   - ImageNet pretrained weights
   - Conv layers만 사용
   - 4개 scale의 feature maps (C2, C3, C4, C5)

2. **FPN**
   - Multi-scale feature extraction
   - 256 channels per level

3. **RPN**
   - Anchor-based proposal generation
   - 3 scales × 3 aspect ratios = 9 anchors

4. **RoI Heads**
   - Box predictor: Classification + BBox regression
   - Mask predictor: Instance segmentation

---

## 🔧 설치 방법

### 1. 환경 요구사항

```bash
- Python >= 3.8
- CUDA >= 11.0 (GPU 사용 시)
- PyTorch >= 2.0
```

### 2. 패키지 설치

```bash
# 리포지토리 클론
git clone https://github.com/kimyoungjin2023/Edoscopic_images_cv_project_NO-Docker.git
cd Edoscopic_images_cv_project_NO-Docker

# 필수 패키지 설치
pip install -r requirements.txt
```

또는 개별 설치:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python albumentations pycocotools matplotlib tqdm pyyaml scikit-learn scipy
```

---

## 📂 데이터 준비

### 1. AI Hub 데이터셋 다운로드

AI Hub에서 "내시경 이미지 합성데이터"를 다운로드하고 압축을 해제합니다.

### 2. 디렉토리 구조

```
data/
├── 01.원천데이터/
│   ├── 1.위/
│   │   ├── 1.궤양/        # 5,000장
│   │   ├── 2.용종/        # 5,000장
│   │   └── 3.암/          # 10,000장
│   └── 2.대장/
│       ├── 1.궤양/        # 5,000장
│       ├── 2.용종/        # 5,000장
│       └── 3.암/          # 10,000장
└── 02.라벨링데이터/
    ├── 1.위/
    │   ├── 1.궤양/        # JSON files
    │   ├── 2.용종/
    │   └── 3.암/
    └── 2.대장/
        ├── 1.궤양/
        ├── 2.용종/
        └── 3.암/
```

### 3. JSON Annotation 예시

```json
{
  "version": "4.2.7",
  "shapes": [
    {
      "label": "01_stomach_ulcer_generation",
      "organ": 0,
      "lesion": 0,
      "location": 1,
      "points": [[x1, y1], [x2, y2], ...],
      "shape_type": "polygon"
    }
  ],
  "imagePath": "1_1_03827.png",
  "imageHeight": 2048,
  "imageWidth": 2048
}
```

---

## 🚀 학습

### 기본 학습

```bash
python train.py \
    --data-root ./data \
    --num-classes 7 \
    --epochs 50 \
    --batch-size 2 \
    --lr 0.005 \
    --output-dir ./checkpoints
```

### 주요 파라미터

#### 데이터 설정
```bash
--data-root                  # 데이터 루트 디렉토리
--train-samples-per-class    # 클래스별 학습 샘플 수 (기본: 1000)
--val-samples-per-class      # 클래스별 검증 샘플 수 (기본: 150)
--test-samples-per-class     # 클래스별 테스트 샘플 수 (기본: 400)
```

#### 모델 설정
```bash
--num-classes                # 클래스 개수 (기본: 7)
--pretrained                 # ImageNet pretrained backbone 사용
--trainable-backbone-layers  # 학습 가능한 backbone layer 수 (0-5, 기본: 5)
```

#### 학습 설정
```bash
--epochs                     # 총 epoch 수 (기본: 50)
--batch-size                 # 배치 크기 (기본: 2)
--lr                         # 학습률 (기본: 0.005)
--momentum                   # SGD momentum (기본: 0.9)
--weight-decay               # 가중치 감쇠 (기본: 0.0005)
--lr-scheduler               # LR scheduler (step/cosine/plateau)
--mixed-precision            # Mixed precision training
```

### 학습 재개

```bash
python train.py \
    --resume ./checkpoints/maskrcnn_epoch_020.pth \
    --epochs 50
```

### GPU 메모리 부족 시

```bash
# 배치 크기 감소
python train.py --batch-size 1

# 이미지 크기 감소
python train.py --img-size 384 --batch-size 2

# Mixed precision training
python train.py --mixed-precision
```

---

## 📈 평가

### Validation 평가

```bash
python eval.py \
    --data-root ./data \
    --checkpoint ./checkpoints/maskrcnn_best.pth \
    --split val \
    --conf-threshold 0.5
```

### Test 평가

```bash
python eval.py \
    --data-root ./data \
    --checkpoint ./checkpoints/maskrcnn_best.pth \
    --split test \
    --conf-threshold 0.5
```

### 평가 메트릭

#### Detection Metrics
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)
- 클래스별 성능 분석

#### Segmentation Metrics
- **Dice Coefficient**: 2 × |A ∩ B| / (|A| + |B|)
- **IoU (Intersection over Union)**: |A ∩ B| / |A ∪ B|

### 결과 저장

평가 결과는 JSON 형식으로 저장됩니다:

```
eval_results/
├── eval_results_val.json    # 검증 결과
├── eval_results_test.json   # 테스트 결과
└── predictions_test.pth      # 예측 결과 (optional)
```

---

## 🔍 추론

### 단일 이미지 추론

```bash
python inference.py \
    --input ./test_image.jpg \
    --checkpoint ./checkpoints/maskrcnn_best.pth \
    --conf-threshold 0.5 \
    --output-dir ./inference_results
```

### 폴더 내 모든 이미지 추론

```bash
python inference.py \
    --input ./test_images/ \
    --checkpoint ./checkpoints/maskrcnn_best.pth \
    --output-dir ./inference_results
```

### 출력 결과

각 이미지에 대해 다음 파일이 생성됩니다:

1. **`[이미지명]_overlay.jpg`**
   - 원본 이미지 + Detection + Segmentation
   - 실제 사용에 적합

2. **`[이미지명]_detailed.png`**
   - 원본, Detection, Segmentation을 나란히 표시
   - 분석 및 검증용

### 예측 결과 형식

```python
{
    'boxes': [[x1, y1, x2, y2], ...],       # Bounding boxes
    'labels': [1, 3, 2, ...],               # Class IDs
    'scores': [0.95, 0.87, 0.82, ...],      # Confidence scores
    'masks': [mask1, mask2, mask3, ...]     # Segmentation masks
}
```

---

## 📁 프로젝트 구조

```
Edoscopic_images_cv_project_NO-Docker/
│
├── train.py                    # 학습 스크립트
├── eval.py                     # 평가 스크립트
├── inference.py                # 추론 스크립트
├── requirements.txt            # 필수 패키지
├── README.md                   # 프로젝트 문서
│
├── models/
│   ├── __init__.py
│   └── mask_rcnn.py           # Mask R-CNN 모델 정의
│                              #   - ResNet50 backbone
│                              #   - FPN, RPN, RoIAlign
│                              #   - Box & Mask heads
│
├── datasets/
│   ├── __init__.py
│   └── dataset.py             # AI Hub 데이터셋 로더
│                              #   - COCO-style annotation 파싱
│                              #   - Train/Val/Test 분할
│                              #   - 클래스별 샘플링
│
├── utils/
│   ├── __init__.py
│   ├── transforms.py          # 의료 영상 특화 Transform
│   │                          #   - CLAHE, Sharpen
│   │                          #   - Augmentation
│   └── engine.py              # 학습/평가 엔진
│                              #   - Training loop
│                              #   - Evaluation
│                              #   - Metrics
│
├── checkpoints/               # 체크포인트 저장 (자동 생성)
│   ├── maskrcnn_best.pth
│   └── maskrcnn_epoch_*.pth
│
├── eval_results/              # 평가 결과 (자동 생성)
│   └── eval_results_*.json
│
└── inference_results/         # 추론 결과 (자동 생성)
    ├── *_overlay.jpg
    └── *_detailed.png
```

---

## 💡 Tips & Best Practices

### 학습 관련

1. **Learning Rate Tuning**
   - 초기 LR: 0.005 (배치 크기 2 기준)
   - 배치 크기가 2배 증가하면 LR도 2배 증가
   - Warmup 사용 권장

2. **Data Augmentation**
   - 의료 영상 특성 고려
   - CLAHE로 조명 불균일 보정
   - Sharpen으로 텍스처 강조
   - 과도한 변형은 지양

3. **배치 크기**
   - GPU 메모리에 따라 조정 (1~4 권장)
   - 작은 배치에서는 BatchNorm 대신 GroupNorm 고려

### 추론 관련

1. **Confidence Threshold**
   - 높은 정밀도: 0.7~0.9
   - 높은 재현율: 0.3~0.5
   - 균형: 0.5 (기본값)

2. **Post-processing**
   - NMS IoU threshold: 0.5 (기본값)
   - 너무 작은 mask는 필터링 고려

---

## 🐛 문제 해결

### CUDA Out of Memory

```bash
# 해결 방법 1: 배치 크기 감소
python train.py --batch-size 1

# 해결 방법 2: 이미지 크기 감소
python train.py --img-size 384

# 해결 방법 3: Mixed precision training
python train.py --mixed-precision
```

### 데이터 로딩 오류

```python
# 데이터 경로 확인
python -c "from datasets.dataset import AIHubEndoscopicDataset; \
           dataset = AIHubEndoscopicDataset('./data', 'train')"
```

### Transform 오류

```python
# Albumentations 버전 확인
pip install albumentations==1.3.1 --upgrade
```

---

## 📚 참고 자료

### Papers
- [Mask R-CNN (He et al., 2017)](https://arxiv.org/abs/1703.06870)
- [ResNet (He et al., 2015)](https://arxiv.org/abs/1512.03385)
- [Feature Pyramid Networks (Lin et al., 2017)](https://arxiv.org/abs/1612.03144)

### Documentation
- [PyTorch Documentation](https://pytorch.org/docs/)
- [torchvision.models.detection](https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection)
- [Albumentations Documentation](https://albumentations.ai/docs/)

---

## 📝 License

This project is licensed under the MIT License.

---

## 👥 Contributors

- OZ Coding School CV Study Team
- [kimyoungjin2023](https://github.com/kimyoungjin2023)

---

## 🙏 Acknowledgments

- AI Hub for providing the endoscopic image dataset
- PyTorch team for torchvision implementation
- Albumentations team for augmentation library

---

## 📧 Contact

문의사항이 있으시면 GitHub Issues를 이용해 주세요.

---

**Happy Training! 🚀**
