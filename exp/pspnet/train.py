# -*- coding:utf-8 -*-
import os
import json
import argparse
import warnings
from typing import Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import metric
from exp.pspnet.model import PSPNet1D
from data.utils import get_dataset
from utils.loss import CrossEntropyDiceLoss

warnings.filterwarnings('ignore')


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch_x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return batch_x.to(torch.float32, non_blocking=True).to(device)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='ahi', choices=['heartbeat', 'ahi'])
    parser.add_argument('--save_path', type=str, default=os.path.join('..', '..', 'result'))

    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--ce_dice_alpha', default=0.7, type=float)
    parser.add_argument('--weight_decay', default=1e-2, type=float)
    parser.add_argument('--grad_clip', default=1.0, type=float)

    parser.add_argument('--device', default='cuda:1', type=str)
    parser.add_argument('--seed', default=42, type=int)
    return parser.parse_args()


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        # Model
        self.model: nn.Module = PSPNet1D(
            in_channels=4,
            out_channels=2,
            stem_channels=32,
            stage_channels=(32, 64, 128, 128),
            stage_blocks=(2, 2, 2, 1),
            stem_kernel=11,
            block_kernel=5,
        ).to(self.device)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs)
        self.criterion = CrossEntropyDiceLoss()
        self.scaler = torch.cuda.amp.GradScaler()

        # Data Loader
        self.train_loader, self.eval_loader = self._build_dataloaders()

    def _make_input(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        # x = torch.stack([data['ECG_1'], data['ECG_2']], dim=1)  # (B, 2, T)
        x = torch.stack([data['AIRFLOW'], data['THOR RES'], data['ABDO RES'], data["SaO2"]], dim=1)
        return to_device(x, self.device)

    def train_one_epoch(self, epoch: int):
        self.model.train()
        for data, target in self.train_loader:
            self.optimizer.zero_grad()

            x = self._make_input(data)
            y = target.long().to(self.device)
            y[y > 0] = 1        # To binary classification

            with torch.cuda.amp.autocast():
                logits = self.model(x)
                loss = self.criterion(logits, y)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        self.scheduler.step(epoch=epoch)

    @torch.inference_mode()
    def eval_one_epoch(self, epoch: int) -> Dict:
        self.model.eval()
        preds_all, reals_all = [], []

        for data, target in self.eval_loader:
            x = self._make_input(data)
            y = target.long().to(self.device)
            y[y > 0] = 1        # To binary classification

            logits = self.model(x)
            preds = logits.argmax(dim=1)

            preds_all.append(preds.detach().cpu())
            reals_all.append(y.detach().cpu())

        y_pred = torch.cat(preds_all, dim=0).reshape(-1).numpy()
        y_true = torch.cat(reals_all, dim=0).reshape(-1).numpy()

        result = metric.calculate_segmentation_metrics(y_pred, y_true, num_classes=2)
        accuracy, iou_macro, dice_macro = result['accuracy'], result['iou_macro'], result['dice_macro']
        print(f'[Epoch]: {epoch:03d} => '
              f'[Accuracy] : {accuracy*100:.2f} [IoU Macro] : {iou_macro*100:.2f} [Dice Macro] : {dice_macro*100:.2f}')
        return result

    def _build_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        train_dataset, eval_dataset = get_dataset(name=self.args.dataset_name)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
        )
        eval_loader = DataLoader(
            dataset=eval_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            drop_last=False,
        )
        return train_loader, eval_loader

    def run(self):
        results = []
        for epoch in range(1, self.args.epochs + 1):
            self.train_one_epoch(epoch)
            result = self.eval_one_epoch(epoch)
            results.append(result)

        file_path = os.path.join(args.save_path, args.dataset_name, 'pspnet.json')
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    args = get_args()
    set_seed(args.seed)
    Trainer(args).run()
