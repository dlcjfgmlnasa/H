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
    def __init__(self, weight: float = 0.5, class_weights: torch.Tensor = None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=0)  # class_weights: (C,) tensor if needed
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


if __name__ == '__main__':
    B, C, T = 4, 3, 100
    logits = torch.randn(B, C, T, requires_grad=True)
    targets = torch.randint(0, C, (B, T))

    criterion = CrossEntropyDiceLoss(weight=0.7)  # CE 70%, Dice 30%
    loss = criterion(logits, targets)
    # loss.backward()
    print("Loss:", loss.item())