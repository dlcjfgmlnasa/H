import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for Binary and Multi-class classification.

    Args:
        gamma (float): focusing parameter (default: 2.0)
        alpha (float or list): balance parameter, scalar for binary, list for multi-class
        reduction (str): 'none' | 'mean' | 'sum'
    """

    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([alpha, 1 - alpha])  # for binary classification
        elif isinstance(alpha, list):
            self.alpha = torch.tensor(alpha)  # for multi-class
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            inputs: (N, C) logits (unnormalized scores) or (N,) for binary
            targets: (N,) ground truth labels [0..C-1]
        """
        if inputs.ndim > 1 and inputs.size(1) > 1:  # multi-class
            logpt = F.log_softmax(inputs, dim=1)
            pt = torch.exp(logpt)
            logpt = logpt.gather(1, targets.unsqueeze(1)).squeeze(1)
            pt = pt.gather(1, targets.unsqueeze(1)).squeeze(1)
        else:  # binary
            inputs = inputs.view(-1)
            targets = targets.view(-1)
            logpt = -F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
            pt = torch.exp(logpt)

        if self.alpha is not None:
            if inputs.ndim > 1 and inputs.size(1) > 1:  # multi-class
                at = self.alpha.to(inputs.device).gather(0, targets)
            else:  # binary
                at = self.alpha.to(inputs.device)[targets.long()]
            logpt = logpt * at

        loss = -((1 - pt) ** self.gamma) * logpt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class SoftDiceLossMultiClass(nn.Module):
    """
    Multiclass Soft Dice Loss
    logits: (B, C, T)  -- raw outputs (before softmax)
    targets: (B,) or (B, T) int64 -- class indices
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(1)
        probs = torch.softmax(logits, dim=1)  # (B, C, T)

        # one-hot targets (B, C, T)
        if targets.dim() == 2:  # (B, T)
            targets_1h = F.one_hot(targets, num_classes=num_classes).permute(0, 2, 1).float()
        elif targets.dim() == 3:  # already one-hot (B, C, T)
            targets_1h = targets.float()
        else:
            raise ValueError("targets must be (B, T) or (B, C, T)")

        dims = (0, 2)  # sum over batch and time
        intersection = torch.sum(probs * targets_1h, dims)
        denominator = torch.sum(probs * probs, dims) + torch.sum(targets_1h * targets_1h, dims) + self.eps
        dice_score = 2. * intersection / denominator
        dice_loss = 1 - dice_score.mean()
        return dice_loss


class CrossEntropyDiceLoss(nn.Module):
    """
    Combined CrossEntropy + Dice Loss
    """
    def __init__(self, weight: float = 0.5, class_weights: torch.Tensor = None, ignore_index=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)  # class_weights: (C,) tensor if needed
        self.dice = SoftDiceLossMultiClass()
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, C, T)
        targets: (B, T) int64
        """
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.weight * ce_loss + (1 - self.weight) * dice_loss


class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-6, ignore_index=None):
        """
        다중 클래스 세그멘테이션을 위한 Dice Loss를 계산합니다.

        Args:
            num_classes (int): 클래스의 총 개수 (배경 포함)
            smooth (float): 0으로 나누는 것을 방지하기 위한 스무딩 값
            ignore_index (int, optional): 손실 계산에서 무시할 레이블.
        """
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)

        if self.ignore_index is not None:
            valid_mask = targets != self.ignore_index
            probs = probs * valid_mask.unsqueeze(1)

            targets_masked = targets.clone()
            targets_masked[~valid_mask] = 0
            targets_one_hot = F.one_hot(targets_masked, num_classes=self.num_classes)
        else:
            targets_one_hot = F.one_hot(targets, num_classes=self.num_classes)

        dims = (0, -1) + tuple(range(1, targets.dim()))
        targets_one_hot = targets_one_hot.permute(*dims).contiguous()

        if self.ignore_index is not None:
            targets_one_hot = targets_one_hot * valid_mask.unsqueeze(1)

        # 4. 교집합(intersection)과 합집합(cardinality) 계산
        axes = tuple(range(2, logits.dim()))
        intersection = torch.sum(probs * targets_one_hot, dim=axes)
        cardinality = torch.sum(probs + targets_one_hot, dim=axes)

        # 5. Dice 계수 계산
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)

        # 6. Dice Loss 계산
        dice_loss = 1 - dice_score

        return dice_loss.mean()


# --- 코드 실행 예제 ---
if __name__ == '__main__':
    NUM_CLASSES = 5
    IGNORE_INDEX = -1  # 무시할 레이블 값

    # 가상 모델 출력 (logits) 및 정답 레이블 생성
    logits = torch.randn(4, NUM_CLASSES, 1000)
    targets = torch.randint(0, NUM_CLASSES, (4, 1000))

    # 일부 레이블을 ignore_index로 설정
    targets[0, 10:20] = IGNORE_INDEX
    targets[2, 50:100] = IGNORE_INDEX
    print(f"Targets에 포함된 고유 값: {torch.unique(targets)}")

    # ignore_index를 포함하여 Dice Loss 객체 생성
    criterion = DiceLoss(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX)

    # 손실 계산
    loss = criterion(logits, targets)

    print(f"\n계산된 Dice Loss: {loss.item()}")