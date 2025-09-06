# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class FTN(nn.Module):
    """
    Fixed Template Network
     - Compares input EEG segments with fixed templates in a learned feature space.
     - Returns classification logits + latent representation
    """
    def __init__(self, n_channels, n_samples, n_classes, hidden_dim=64):
        super().__init__()
        self.n_classes = n_classes

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, (1, 9), padding=(0, 4), bias=False),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, hidden_dim, (n_channels, 1), bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.Tanh()
        )

        # Fixed templates (frozen, not updated during training)
        self.register_buffer("templates", torch.randn(n_classes, 1, n_channels, n_samples))

        # Classification head
        self.fc = nn.Linear(hidden_dim, n_classes)


    def forward(self, x, return_feat=False):
        """
        Args:
            x (Tensor): EEG input of shape (B, 1, C, T)
            return_feat (bool):
                - True → return (logits, representation)
                - False → return logits only

        Returns:
            logits (Tensor): (B, n_classes) classification output
            feat_x (Tensor): (B, hidden_dim) latent representation
        """
        # EEG feature extraction
        feat_x = self.feature_extractor(x)          # (B, hidden_dim, 1, T')
        feat_x = feat_x.mean(dim=-1).squeeze(-1)    # (B, hidden_dim)

        # Template feature extraction
        t = self.feature_extractor(self.templates)  # (C, hidden_dim, 1, T')
        t = t.mean(dim=-1).squeeze(-1)              # (C, hidden_dim)

        # Cosine similarity between EEG features and templates
        sim = F.linear(F.normalize(feat_x, dim=1),
                       F.normalize(t, dim=1))       # (B, C)

        logits = self.fc(sim)

        if return_feat:
            return logits, feat_x
        return logits, feat_x


class DTN(nn.Module):
    """
    Dynamic Template Network
     - Extracts conv features from EEG
     - Global average pooling for dimension reduction
     - Returns latent representation only
    """
    def __init__(self, n_bands, n_features, n_channels, n_samples, n_classes,
                 band_kernel=9, pooling_kernel=2,
                 dropout=0.5, momentum=None, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

        # One-hot table for template update
        self.register_buffer('encode_table', torch.eye(n_classes, dtype=torch.long))

        # Feature extractor
        self.feature_extractor = nn.Sequential(OrderedDict([
            ('band_layer', nn.Conv2d(1, n_bands, (1, band_kernel),
                                     padding=(0, band_kernel // 2), bias=False)),
            ('spatial_layer', nn.Conv2d(n_bands, n_features, (n_channels, 1),
                                        bias=False)),
            ('temporal_layer1', nn.Conv2d(n_features, n_features, (1, pooling_kernel),
                                          stride=(1, pooling_kernel), bias=False)),
            ('bn', nn.BatchNorm2d(n_features)),
            ('tanh', nn.Tanh()),
            ('temporal_layer2', nn.Conv2d(n_features, n_features, (1, band_kernel),
                                          padding=(0, band_kernel // 2), bias=False)),
        ]))

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_drop = nn.Dropout(dropout)
        self.instance_norm = nn.InstanceNorm2d(1)

        # Initialize running templates
        with torch.no_grad():
            x = torch.zeros(1, 1, n_channels, n_samples)
            feat = self.feature_extractor(x)
            self._register_templates(n_classes, *feat.shape[1:])


    def _register_templates(self, n_classes, *args):
        """ Initialize class-specific templates """
        self.register_buffer('running_template', torch.zeros(n_classes, *args))
        nn.init.xavier_uniform_(self.running_template, gain=1)


    def _update_templates(self, x, y):
        """ Update templates with exponential moving average (EMA) """
        with torch.no_grad():
            self.num_batches_tracked += 1
            factor = (1.0 / float(self.num_batches_tracked)) if self.momentum is None else self.momentum

            mask = F.one_hot(y, num_classes=self.running_template.shape[0]).float()  # (B, n_classes)
            features = self.feature_extractor(x)                                     # (B, F, C’, T’)

            mask_data = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * features.unsqueeze(1)
            new_template = mask_data.sum(0) / (mask.sum(0).view(-1, 1, 1, 1) + self.eps)

            self.running_template = (1 - factor) * self.running_template + factor * new_template


    def forward(self, x, y=None, return_feat=True):
        """
        x: (B, 1, C, T)
        return_feat=True → (logits=None, feat)
        """
        x = self.instance_norm(x)
        feat = self.feature_extractor(x)        # (B, F, C’, T’)
        feat = self.global_pool(feat)           # (B, F, 1, 1)
        feat = feat.view(feat.size(0), -1)      # (B, F)
        feat = self.fc_drop(feat)

        # Update templates if training
        if self.training and y is not None:
            self._update_templates(x, y)

        if return_feat:
            return None, feat

        return None, feat


# -*- coding:utf-8 -*-
import torch
import torch.nn as nn


class StimulusEncoder(nn.Module):
    """
    Stimulus Encoder
    - Encodes sinusoidal reference signals (sin, cos) into latent features.
    - Input: (B, T, 2) → B=batch, T=time length, 2=[sin, cos]
    - Output: (B, D) → flattened latent representation
    """
    def __init__(self, in_dim=2, hidden_dim=64):
        """
        Args:
            in_dim (int): input dimension, usually 2 (sin, cos)
            hidden_dim (int): output embedding dimension (D)
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # preserves temporal structure and frequency information
        self.encoder = nn.Sequential(
            nn.Conv1d(in_dim, 32, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(32, hidden_dim, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # global temporal pooling → single embedding
        )

    def forward(self, stim):
        """
        Args:
            stim (Tensor): (B, T, 2) sinusoidal stimulus signals

        Returns:
            Tensor: (B, hidden_dim) latent representation
        """
        stim = stim.permute(0, 2, 1)    # (B, T, 2) -> (B, 2, T)
        feat = self.encoder(stim)       # (B, hidden_dim, 1)
        feat = feat.squeeze(-1)         # (B, hidden_dim)

        return feat


# -*- coding:utf-8 -*-
import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self, chans, samples, dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16):
        super().__init__()
        self.chans = chans
        self.samples = samples

        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), padding=(0, kernLength // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        self.depthwiseConv = nn.Conv2d(F1, F1 * D, (chans, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.elu = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropoutRate)

        self.separableConv = nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.avgpool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropoutRate)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, chans, samples)   # (B,1,C,T)
            out = self.forward_features(dummy)
            self.out_dim = out.view(1, -1).shape[1]

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwiseConv(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.avgpool1(x)
        x = self.drop1(x)
        x = self.separableConv(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.avgpool2(x)
        x = self.drop2(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.view(x.size(0), -1)   # flatten
        return x

# -*- coding:utf-8 -*-
import torch
import torch.nn as nn

class DualAttention(nn.Module):
    """
    Dual Attention module
    - EEG → Key/Value
    - Stimulus & Template → Query
    - Attention outputs (A_stim, A_temp) fused by concatenation + depthwise conv
    - Projection head for SSL (self-supervised learning)
    """
    def __init__(self, d_eeg, d_query, d_model, num_heads=4, proj_dim=64):
        """
        Args:
            d_eeg (int): EEG feature dimension (flattened from EEG encoder)
            d_query (int): Query feature dimension (Stimulus/Template encoder output)
            d_model (int): Transformer hidden dimension
            num_heads (int): number of attention heads
            proj_dim (int): final projection dimension
        """
        super().__init__()

        # EEG → Key/Value
        self.key = nn.Linear(d_eeg, d_model)
        self.value = nn.Linear(d_eeg, d_model)

        # Stimulus/Template → Query
        self.query_temp = nn.Linear(d_query, d_model)
        self.query_stim = nn.Linear(d_query, d_model)

        # Multi-head attention
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)

        # Fusion: [A_temp, A_stim] concat + depthwise conv
        self.fusion_conv = nn.Sequential(
            nn.Conv1d(2 * d_model, 2 * d_model, kernel_size=3, padding=1, groups=2 * d_model),  # depthwise
            nn.ReLU(),
            nn.Conv1d(2 * d_model, d_model, kernel_size=1)  # pointwise
        )

        # Projection head (MLP)
        self.proj_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, proj_dim)
        )


    def forward(self, eeg_feat, stim_feat, temp_feat):
        """
        Args:
            eeg_feat  : (B, D_eeg) from EEG encoder
            stim_feat : (B, D_query) from Stimulus encoder
            temp_feat : (B, D_query) from Template encoder

        Returns:
            proj      : (B, proj_dim) final representation
            A_stim    : (B, d_model) stimulus attention output
            A_temp    : (B, d_model) template attention output
        """
        # EEG → Key/Value
        K = self.key(eeg_feat).unsqueeze(1)  # (B, 1, d_model)
        V = self.value(eeg_feat).unsqueeze(1)  # (B, 1, d_model)

        # Stimulus / Template → Query
        Q_stim = self.query_stim(stim_feat).unsqueeze(1)  # (B, 1, d_model)
        Q_temp = self.query_temp(temp_feat).unsqueeze(1)  # (B, 1, d_model)

        # Attention
        A_stim, _ = self.attn(Q_stim, K, V)  # (B, 1, d_model)
        A_temp, _ = self.attn(Q_temp, K, V)  # (B, 1, d_model)

        # Fusion: concat along channel → depthwise + pointwise conv
        A_cat = torch.cat([A_stim, A_temp], dim=-1)  # (B, 1, 2*d_model)
        A_cat = A_cat.permute(0, 2, 1)  # (B, 2*d_model, 1)
        A_fused = self.fusion_conv(A_cat)  # (B, d_model, 1)
        A_fused = A_fused.squeeze(-1)  # (B, d_model)

        # Projection head
        proj = self.proj_head(A_fused)  # (B, proj_dim)

        return proj, A_stim.squeeze(1), A_temp.squeeze(1)


# -*- coding:utf-8 -*-
import torch
import torch.nn as nn

class EEGBranch(nn.Module):
    """
    EEG branch using EEGNet as encoder.
    Input : (B, 1, C, T)
    Output : (B, D_eeg) flattened EEG representation
    """
    def __init__(self, chans, samples):
        super().__init__()
        self.encoder = EEGNet(chans=chans, samples=samples)
        self.out_dim = self.encoder.out_dim

    def forward(self, x):
        feat = self.encoder(x)  # (B, out_dim)
        return feat


class StimulusBranch(nn.Module):
    """
    Stimulus branch using StimulusEncoder.
    Input : (B, T, 2) sinusoidal references (sin, cos)
    Output: (B, D_stim)
    """
    def __init__(self, hidden_dim=128, n_harmonics=3):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.encoder = StimulusEncoder(in_dim=2 * n_harmonics, hidden_dim=hidden_dim)

    def forward(self, stim):
        """
        stim: (B, T, 2) fundamental sin/cos
        Returns: (B, D_stim)
        """
        base_sin, base_cos = stim[..., 0], stim[..., 1]              # (B, T)

        harmonics = []
        for h in range(1, self.n_harmonics + 1):
            harmonics.append(torch.sin(h * torch.arcsin(base_sin)))  # sin(hf)
            harmonics.append(torch.cos(h * torch.arccos(base_cos)))  # cos(hf)

        stim_harm = torch.stack(harmonics, dim=-1)  # (B, T, 2*n_harmonics)
        feat = self.encoder(stim_harm)              # (B, hidden_dim) = (B, D_stim)
        return feat


class TemplateBranch(nn.Module):
    """
    Template branch using DTN.
    Input : (B, 1, C, T), labels (optional)
    Output : (B, D_temp) latent representation
    """
    def __init__(self, n_bands, n_features, n_channels, n_samples, n_classes, D_temp=64):
        super().__init__()
        self.network = DTN(n_bands=n_bands, n_features=n_features,
                           n_channels=n_channels, n_samples=n_samples,
                           n_classes=n_classes)

        # Projection: (B, n_features) → (B, D_temp)
        self.proj = nn.Linear(n_features, D_temp)

    def forward(self, x, y=None):
        _, feat = self.network(x, y, return_feat=True)      # (B, n_features)
        feat = self.proj(feat)                              # (B, D_temp)
        return feat


# -*- coding:utf-8 -*-
import mne
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

from moabb.paradigms import SSVEP
from moabb.datasets import Nakanishi2015
from moabb.datasets import Lee2019_SSVEP


class ARDataset(Dataset):
    def __init__(self, npz_file, n_classes=None, T_stim=None):
        """
        npz_file: path to .npz file
        T_stim: stimulus length (default = EEG segment length)
        """
        data = np.load(npz_file, allow_pickle=True)
        self.epochs = data["epochs"]   # (N,C,T)
        self.labels = data["labels"]   # (N,)
        self.tasks  = data["tasks"]    # (N,)
        self.ch_names = data["ch_names"]
        self.sfreq = float(data["sfreq"])

        self.N, self.C, self.T = self.epochs.shape
        self.T_stim = self.T if T_stim is None else T_stim

        # Map frequency ↔ class index
        unique_freqs = sorted(np.unique(self.labels))
        self.freq2class = {f: i for i, f in enumerate(unique_freqs)}
        self.class2freq = {i: f for f, i in self.freq2class.items()}
        self.n_classes = len(unique_freqs)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        eeg = self.epochs[idx]  # (C,T)
        label_hz = int(self.labels[idx])
        class_label = self.freq2class[label_hz]

        # EEG input
        eeg = torch.tensor(eeg, dtype=torch.float32).unsqueeze(0)  # (1,C,T)

        # Generate sinusoidal references
        t = np.arange(self.T_stim) / self.sfreq
        f = label_hz
        stim = np.stack([np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t)], axis=-1)  # (T_stim,2)
        stim = torch.tensor(stim, dtype=torch.float32)

        task = self.tasks[idx]

        assert eeg.shape[-1] == stim.shape[0], "EEG segment length and stimulus length must match (2s)."
        return eeg, stim, class_label, task


class Nakanishi2015Dataset(Dataset):
    def __init__(self, subjects=[1], pick_channels="all"):
        dataset = Nakanishi2015()
        paradigm = SSVEP()
        X, labels, meta = paradigm.get_data(dataset=dataset, subjects=subjects)

        # Label encoding
        le = LabelEncoder()
        self.labels = le.fit_transform(labels)
        self.freqs = le.classes_.astype(float)
        self.sfreq = 256.0

        # True channel names from the original paper
        ch_names = ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"]

        # Build MNE object
        info = mne.create_info(ch_names=ch_names, sfreq=self.sfreq, ch_types="eeg")
        raw = mne.EpochsArray(X.astype(np.float32), info)

        # Pick channels if requested
        if pick_channels != "all":
            raw.pick(pick_channels)

        # Bandpass filter
        raw.filter(l_freq=1, h_freq=40, fir_design="firwin", verbose=False)

        # Final data
        self.epochs = raw.get_data().astype(np.float32)  # (N, C, T)
        self.N, self.C, self.T = self.epochs.shape
        self.n_classes = len(np.unique(self.labels))
        self.ch_names = raw.info["ch_names"]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        eeg_np = self.epochs[idx]  # (C, T)
        label = int(self.labels[idx])
        f = float(self.freqs[label])

        eeg = torch.tensor(eeg_np, dtype=torch.float32).unsqueeze(0)  # (1, C, T)

        # Reference signals
        t = np.arange(self.T, dtype=np.float32) / self.sfreq
        stim_np = np.stack([np.sin(2 * np.pi * f * t),
                            np.cos(2 * np.pi * f * t)], axis=-1)  # (T, 2)
        stim = torch.tensor(stim_np, dtype=torch.float32)

        return eeg, stim, label


class Lee2019Dataset(Dataset):
    def __init__(self, subjects=[1], train=True, rfreq=250, pick_channels="all"):
        super().__init__()
        paradigm = SSVEP()
        dataset = Lee2019_SSVEP()

        X, labels, meta = paradigm.get_data(dataset=dataset, subjects=subjects)

        if train:
            session_mask = (meta['session'] == "0")
        else:
            session_mask = (meta['session'] == "1")

        X = X[session_mask]
        labels = labels[session_mask]

        le = LabelEncoder()
        y = le.fit_transform(labels)

        # Channel mapping (ch1 ~ ch62)
        mapping_lee2019 = {
            "ch1": "Fp1", "ch2": "Fp2", "ch3": "Fp7", "ch4": "F3", "ch5": "Fz", "ch6": "F4", "ch7": "F8",
            "ch8": "FC5", "ch9": "FC1", "ch10": "FC2", "ch11": "FC6", "ch12": "T7", "ch13": "C3",
            "ch14": "Cz", "ch15": "C4", "ch16": "T8", "ch17": "TP9", "ch18": "CP5", "ch19": "CP1",
            "ch20": "CP2", "ch21": "CP6", "ch22": "TP10", "ch23": "P7", "ch24": "P3", "ch25": "Pz",
            "ch26": "P4", "ch27": "P8", "ch28": "PO9", "ch29": "O1", "ch30": "Oz", "ch31": "O2",
            "ch32": "PO10", "ch33": "FC3", "ch34": "FC4", "ch35": "C5", "ch36": "C1", "ch37": "C2",
            "ch38": "C6", "ch39": "CP3", "ch40": "CPz", "ch41": "CP4", "ch42": "P1", "ch43": "P2",
            "ch44": "POz", "ch45": "FT9", "ch46": "FTT9h", "ch47": "TPP7h", "ch48": "TP7", "ch49": "TPP9h",
            "ch50": "FT10", "ch51": "FTT10h", "ch52": "TPP8h", "ch53": "TP8", "ch54": "TPP10h", "ch55": "F9",
            "ch56": "F10", "ch57": "AF7", "ch58": "AF3", "ch59": "AF4", "ch60": "AF8", "ch61": "PO3", "ch62": "PO4"
        }

        ch_names = [mapping_lee2019.get(f"ch{i+1}", f"ch{i+1}") for i in range(X.shape[1])]

        info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types="eeg")
        raw = mne.EpochsArray(X.astype(np.float32), info)

        if pick_channels != "all":
            raw.pick(pick_channels)

        raw.resample(rfreq)
        raw.filter(l_freq=1, h_freq=40, fir_design="firwin", verbose=False)

        self.X = torch.tensor(raw.get_data(), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.freqs = le.classes_

        self.N, self.C, self.T = self.X.shape
        self.n_classes = len(np.unique(self.y))
        self.ch_names = raw.info["ch_names"]

        # Reference signals
        self.stim_refs = []
        t = np.arange(self.T) / rfreq
        for label in self.y:
            f = float(self.freqs[label])
            ref = np.stack([np.sin(2 * np.pi * f * t),
                            np.cos(2 * np.pi * f * t)], axis=-1)
            self.stim_refs.append(ref)
        self.stim_refs = torch.tensor(np.array(self.stim_refs), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        eeg = self.X[idx].unsqueeze(0)  # (1, C, T)
        stim = self.stim_refs[idx]
        label = self.y[idx]
        return eeg, stim, label


if __name__ == '__main__':
    import mne
    import warnings
    import moabb
    import numpy as np
    from moabb.datasets import Lee2019_SSVEP
    from moabb.paradigms import SSVEP

    train_dataset = Lee2019Dataset(subjects=[10], train=True)
    eval_dataset = Lee2019Dataset(subjects=[10], train=False)

    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=64)
    eval_dataloader = DataLoader(eval_dataset, batch_size=64)


    import torch.optim as optim

    n_channels = 62
    n_samples = 1000
    d_query = 64
    d_model = 128
    n_classes = 4
    lr = 0.0001

    device = 'cuda:1'

    eeg_branch = EEGBranch(chans=n_channels, samples=n_samples).to(device)
    stim_branch = StimulusBranch(hidden_dim=d_query, n_harmonics=3).to(device)
    temp_branch = TemplateBranch(n_bands=8, n_features=32,
                                 n_channels=n_channels,
                                 n_samples=n_samples,
                                 n_classes=n_classes,
                                 D_temp=d_query).to(device)
    dual_attn = DualAttention(d_eeg=eeg_branch.out_dim,
                              d_query=d_query,
                              d_model=d_model,
                              num_heads=4,
                              proj_dim=n_classes).to(device)

    params = list(eeg_branch.parameters()) + list(stim_branch.parameters()) + \
              list(temp_branch.parameters()) + list(dual_attn.parameters())

    epochs = 5000
    optimizer = optim.Adam(params, lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        for data in train_dataloader:
            optimizer.zero_grad()

            eeg_branch.train()
            stim_branch.train()
            temp_branch.train()
            dual_attn.train()

            eeg, stim, label = data
            eeg, stim, label = eeg.to(device), stim.to(device), label.to(device)
            eeg_feat = eeg_branch(eeg)  # (B, D_eeg)
            stim_feat = stim_branch(stim)  # (B, D_query)
            temp_feat = temp_branch(eeg, label)  # (B, D_query)
            logits, _, _ = dual_attn(eeg_feat, stim_feat, temp_feat)  # (B, n_classes)

            loss = criterion(logits, label)
            loss.backward()

            optimizer.step()

        if epoch % 10 == 0:
            real_list, pred_list = [], []
            for data in eval_dataloader:
                with torch.no_grad():
                    eeg_branch.eval()
                    stim_branch.eval()
                    temp_branch.eval()
                    dual_attn.eval()

                    eeg, stim, label = data
                    eeg, stim, label = eeg.to(device), stim.to(device), label.to(device)

                    eeg_feat = eeg_branch(eeg)  # (B, D_eeg)
                    stim_feat = stim_branch(stim)  # (B, D_query)
                    temp_feat = temp_branch(eeg, label)  # (B, D_query)
                    logits, _, _ = dual_attn(eeg_feat, stim_feat, temp_feat)  # (B, n_classes)

                    pred_list.extend(list(logits.argmax(dim=-1).detach().cpu().numpy()))
                    real_list.extend(list(label.detach().cpu().numpy()))
            real_list, pred_list = np.array(real_list), np.array(pred_list)
            from sklearn.metrics import classification_report

            print(f'[Epoch] : {epoch}')
            print(classification_report(real_list, pred_list))

        # scheduler.step()
