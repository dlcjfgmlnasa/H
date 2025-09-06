import numpy as np
from typing import Dict, Optional, Union

def calculate_segmentation_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
    ignore_index: Optional[int] = None,
    epsilon: float = 1e-6
) -> Dict[str, float]:

    preds = np.asarray(preds)
    targets = np.asarray(targets)

    # 2. 혼동 행렬(Confusion Matrix) 계산
    mask = (targets != ignore_index)
    preds_masked = preds[mask]
    targets_masked = targets[mask]

    conf_matrix = np.bincount(
        num_classes * targets_masked.astype(np.int32) + preds_masked.astype(np.int32),
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes).astype(np.float32)

    tp = np.diag(conf_matrix)
    fp = conf_matrix.sum(axis=1) - tp  # torch.sum(dim=1) -> np.sum(axis=1)
    fn = conf_matrix.sum(axis=0) - tp  # torch.sum(dim=0) -> np.sum(axis=0)

    per_class_iou = tp / (tp + fp + fn + epsilon)
    per_class_dice = 2 * tp / (2 * tp + fp + fn + epsilon)
    per_class_precision = tp / (tp + fp + epsilon)
    per_class_recall = tp / (tp + fn + epsilon)

    micro_iou = tp.sum() / (tp.sum() + fp.sum() + fn.sum() + epsilon)
    micro_dice = 2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum() + epsilon)
    macro_iou = per_class_iou.mean()
    macro_dice = per_class_dice.mean()

    support = conf_matrix.sum(axis=1)
    weighted_iou = (per_class_iou * support / (support.sum() + epsilon)).sum()
    weighted_dice = (per_class_dice * support / (support.sum() + epsilon)).sum()

    # 5. 결과 딕셔너리 반환
    results = {
        "iou_micro": micro_iou.item(),
        "iou_macro": macro_iou.item(),
        "iou_weighted": weighted_iou.item(),
        "dice_micro": micro_dice.item(),
        "dice_macro": macro_dice.item(),
        "dice_weighted": weighted_dice.item(),
        "per_class_iou": per_class_iou.tolist(),
        "per_class_dice": per_class_dice.tolist(),
        "per_class_precision": per_class_precision.tolist(),
        "per_class_recall": per_class_recall.tolist(),
    }
    return results