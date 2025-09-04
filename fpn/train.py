# -*- coding:utf-8 -*-
import torch
import argparse
from fpn.model import MultiModalFPN1D, PyramidSegHead1D
from torch.utils.data import DataLoader, Dataset
from data.utils import get_dataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='heartbeat', choices=['heartbeat', 'ahi'])
    parser.add_argument('--base_path', default='/data/segmentation/mit_bit', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    return parser.parse_args()


class Trainer(object):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = MultiModalFPN1D(
            modalities={'ecg': 2},
            backbone_cfg=dict(
                stem_channels=64,
                stage_channels=(128, 256, 512, 512),
                stage_blocks=(2, 2, 2, 2),
            ),
            fpn_out_channels=192,
            use_c2_in_fpn=False,  # Build P3~P5
            fusion="attn",  # "sum" | "concat" | "attn"
        )
        self.head = PyramidSegHead1D(in_ch=192, out_ch=1)

    def train(self):
        dataset = get_dataset(name=self.args.dataset_name, base_path=self.args.base_path)
        columns = dataset.signal_cols
        dataloader = DataLoader(dataset=dataset, batch_size=self.args.batch_size)

        for epoch in range(self.args.epochs):
            for data, mask in dataloader:
                data = {'ecg': torch.stack([data['ECG_1'], data['ECG_2']], dim=1)}
                out = self.model(data)
                out = self.head(out, original_len=1250).squeeze()
                print(out.shape)
                print(mask.shape)
                exit()

        pass


if __name__ == '__main__':
    Trainer(get_args()).train()
