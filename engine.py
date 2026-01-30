"""
Training and Evaluation Engine for Mask R-CNN
================================================================
학습/평가 루프, 메트릭 계산, 체크포인트 관리
================================================================
"""

import torch
import sys
import time
import datetime
from collections import deque, defaultdict
import numpy as np
from tqdm import tqdm


class MetricLogger:
    """메트릭 로깅 클래스"""
    
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
    
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)
    
    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{attr}'")
    
    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)
    
    def add_meter(self, name, meter):
        self.meters[name] = meter
    
    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable),
                        eta=eta_string,
                        meters=str(self),
                        time=str(iter_time),
                        data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB
                    ))
                else:
                    print(log_msg.format(
                        i, len(iterable),
                        eta=eta_string,
                        meters=str(self),
                        time=str(iter_time),
                        data=str(data_time)
                    ))
            
            i += 1
            end = time.time()
        
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f'{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)')


class SmoothedValue:
    """이동 평균 계산"""
    
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt
    
    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n
    
    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()
    
    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()
    
    @property
    def global_avg(self):
        return self.total / self.count if self.count != 0 else 0
    
    @property
    def max(self):
        return max(self.deque)
    
    @property
    def value(self):
        return self.deque[-1]
    
    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value
        )


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50, scaler=None):
    """
    1 epoch 학습
    
    Args:
        model: Mask R-CNN 모델
        optimizer: 옵티마이저
        data_loader: 학습 데이터로더
        device: cuda/cpu
        epoch: 현재 epoch
        print_freq: 로그 출력 빈도
        scaler: GradScaler for mixed precision training
    
    Returns:
        metric_logger: 학습 메트릭
    """
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            loss_value = losses.item()
            
            if not np.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training")
                print(loss_dict)
                sys.exit(1)
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        
        # Update metrics
        metric_logger.update(loss=losses, **loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    return metric_logger


@torch.no_grad()
def evaluate(model, data_loader, device):
    """
    검증/테스트 평가
    
    Args:
        model: Mask R-CNN 모델
        data_loader: 검증/테스트 데이터로더
        device: cuda/cpu
    
    Returns:
        results: 평가 결과
    """
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Evaluation:'
    
    results = {
        'predictions': [],
        'targets': []
    }
    
    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        outputs = model(images)
        
        # CPU로 이동하여 저장
        outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs]
        targets = [{k: v.cpu() for k, v in t.items()} for t in targets]
        
        results['predictions'].extend(outputs)
        results['targets'].extend(targets)
    
    return results


def compute_metrics(predictions, targets, iou_threshold=0.5):
    """
    Detection 및 Segmentation 메트릭 계산
    
    Args:
        predictions: 예측 결과 리스트
        targets: 정답 라벨 리스트
        iou_threshold: IoU threshold for matching
    
    Returns:
        metrics: dict with precision, recall, mAP, etc.
    """
    from collections import defaultdict
    
    metrics = defaultdict(list)
    
    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes']
        pred_labels = pred['labels']
        pred_scores = pred['scores']
        
        target_boxes = target['boxes']
        target_labels = target['labels']
        
        if len(pred_boxes) == 0 or len(target_boxes) == 0:
            continue
        
        # Box IoU 계산
        ious = box_iou(pred_boxes, target_boxes)
        
        # Matching
        for i, (pred_box, pred_label, pred_score) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
            max_iou, max_idx = ious[i].max(0)
            
            if max_iou >= iou_threshold and pred_label == target_labels[max_idx]:
                metrics['true_positives'].append(1)
            else:
                metrics['false_positives'].append(1)
    
    # Precision, Recall 계산
    tp = sum(metrics.get('true_positives', []))
    fp = sum(metrics.get('false_positives', []))
    fn = len(targets) - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn
    }


def box_iou(boxes1, boxes2):
    """
    Box IoU 계산
    
    Args:
        boxes1: (N, 4) tensor [x1, y1, x2, y2]
        boxes2: (M, 4) tensor [x1, y1, x2, y2]
    
    Returns:
        iou: (N, M) tensor
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    
    union = area1[:, None] + area2 - inter
    
    iou = inter / union
    return iou


def save_checkpoint(model, optimizer, epoch, loss, filename, scheduler=None):
    """
    체크포인트 저장
    
    Args:
        model: 모델
        optimizer: 옵티마이저
        epoch: 현재 epoch
        loss: 현재 loss
        filename: 저장 경로
        scheduler: learning rate scheduler (optional)
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, filename)
    print(f"✓ Checkpoint saved: {filename}")


def load_checkpoint(model, optimizer, filename, device, scheduler=None):
    """
    체크포인트 로드
    
    Args:
        model: 모델
        optimizer: 옵티마이저
        filename: 체크포인트 경로
        device: cuda/cpu
        scheduler: learning rate scheduler (optional)
    
    Returns:
        epoch, loss
    """
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"✓ Checkpoint loaded: {filename}")
    print(f"  Resume from epoch {epoch}, loss: {loss:.4f}")
    
    return epoch, loss


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    """
    Learning rate warmup scheduler
    
    Args:
        optimizer: 옵티마이저
        warmup_iters: warmup iteration 수
        warmup_factor: 초기 lr 배수
    """
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


if __name__ == "__main__":
    print("="*60)
    print("Training Engine Module")
    print("="*60)
    
    print("\nAvailable functions:")
    print("  - train_one_epoch(): 1 epoch 학습")
    print("  - evaluate(): 모델 평가")
    print("  - compute_metrics(): 메트릭 계산")
    print("  - save_checkpoint(): 체크포인트 저장")
    print("  - load_checkpoint(): 체크포인트 로드")
    print("  - warmup_lr_scheduler(): LR warmup")
    
    print("\nUsage Example:")
    print("""
    from utils.engine import train_one_epoch, evaluate, save_checkpoint
    
    # 학습
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch)
        
        # 평가
        results = evaluate(model, val_loader, device)
        
        # 저장
        save_checkpoint(model, optimizer, epoch, loss, 'checkpoint.pth')
    """)
