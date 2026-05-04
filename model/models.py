import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mamba_ssm import Mamba
from sklearn.metrics import r2_score

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, checkpoint_dir='/root/model/checkpoints'):
        self.val_loss_min = np.inf
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
    def __call__(self, val_loss, model):
        print(f"val_loss={val_loss:.6f}")
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f})')
        torch.save(model.state_dict(), self.checkpoint_path)
        self.val_loss_min = val_loss

# New Causal Convolution Layer with Dilation
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation, bias=False)
    def forward(self, x):
        x = self.conv(x)
        # Remove the extra padding on the right to ensure causality
        if self.padding != 0:
            return x[:, :, :-self.padding]
        return x

# Updated ConvBNLayer without 'stride' parameter
class ConvBNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, act=None):
        super().__init__()
        self.conv = CausalConv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = act
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act == 'relu':
            return F.relu(x)
        return x

# ResNetBlock without 'stride'

class TCNResBlock(nn.Module):
    """TCN residual block: ConvBN → ReLU → ConvBN + shortcut"""
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, dropout=0.1):
        super().__init__()
        self.conv1 = ConvBNLayer(in_channels, out_channels, kernel_size, dilation=dilation, act='relu')
        self.conv2 = ConvBNLayer(out_channels, out_channels, kernel_size, dilation=dilation, act='relu')
        self.dropout = nn.Dropout(dropout)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.dropout(self.conv1(x))
        out = self.conv2(out)
        return F.relu(out + residual)
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBNLayer(in_channels, out_channels, 3, dilation=1, act='relu')
        self.conv2 = ConvBNLayer(out_channels, out_channels, 3, dilation=1, act='relu')
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = ConvBNLayer(in_channels, out_channels, 1, dilation=1)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return F.relu(x + residual)

# ResNet using MaxPool1d for downsampling
class ResNet(nn.Module):
    def __init__(self, input_channels=5, output_size=96, dropout=0.1):
        super().__init__()
        self.initial = ConvBNLayer(input_channels, 64, 7, dilation=1, act='relu')
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2)
        self.layer3 = self._make_layer(128, 256, 2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, output_size)

    def _make_layer(self, in_channels, out_channels, blocks):
        layers = [ResNetBlock(in_channels, out_channels)]
        for _ in range(1, blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        x = self.pool(x)
        x = self.layer1(x)
        x = self.pool(x)
        x = self.layer2(x)
        x = self.pool(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)

class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)
    def forward(self, hidden_states):
        energy = torch.tanh(self.W(hidden_states))
        attention = F.softmax(self.V(energy), dim=1)
        attended = torch.sum(attention * hidden_states, dim=1)
        return attended

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        self.position_embedding = nn.Embedding(max_len, d_model)
    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return x + self.position_embedding(positions)

class MambaBlock(nn.Module):
    def __init__(self, input_dim, d_model=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=16,
                d_conv=4,
                expand=2,
            ) for _ in range(num_layers)
        ])
        self.proj = nn.Linear(d_model, d_model)
    def forward(self, x):
        x_embed = self.embedding(x)
        x_pos = self.pos_encoder(x_embed)
        x_pos = self.dropout(x_pos)
        for mamba_layer in self.mamba_layers:
            x_pos = mamba_layer(x_pos)
        return self.proj(x_pos)

class MultiTaskModel(nn.Module):
    def __init__(self,
                 feat_num=13,
                 future_feat_num=5,
                 tcn_channels=[64, 128, 256],
                 mamba_hidden=256,
                 pred_len=96,
                 dropout=0.1):
        super().__init__()
        self.feat_num = feat_num
        self.pred_len = pred_len
        self.tcn = nn.Sequential(
            TCNResBlock(feat_num, tcn_channels[0], 9, dilation=1, dropout=dropout),
            TCNResBlock(tcn_channels[0], tcn_channels[1], 9, dilation=2, dropout=dropout),
            TCNResBlock(tcn_channels[1], tcn_channels[2], 9, dilation=4, dropout=dropout),
        )
        self.mamba = MambaBlock(
            input_dim=tcn_channels[-1],
            d_model=mamba_hidden,
            num_layers=2,
            dropout=dropout,
        )
        self.attention = TemporalAttention(mamba_hidden)
        self.future_resnet = ResNet(input_channels=future_feat_num, output_size=pred_len, dropout=dropout)
        self.task1 = nn.Sequential(
            nn.Linear(mamba_hidden + pred_len, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, pred_len)
        )
        self.task2 = nn.Sequential(
            nn.Linear(mamba_hidden + pred_len, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, pred_len)
        )
    def forward(self, x1, x2=None):
        B = x1.size(0)
        x = x1.permute(0, 2, 1)
        x = self.tcn(x)
        x = x.permute(0, 2, 1)
        mamba_out = self.mamba(x)
        attn_vec = self.attention(mamba_out)
        if x2 is None:
            future_feat = torch.zeros((B, self.pred_len), device=x1.device, dtype=x1.dtype)
        else:
            future_in = x2.permute(0, 2, 1)
            future_feat = self.future_resnet(future_in)
        cat = torch.cat([attn_vec, future_feat], dim=1)
        out1 = self.task1(cat)
        out2 = self.task2(cat)
        out = torch.stack([out1, out2], dim=-1)
        return out

class DynamicWeightedLoss(nn.Module):
    """Dynamic weighted Huber loss as per paper Eq.(7).
    ωt linearly increases from 0.1 to 0.5 during training.
    L_total = (1 - ωt) * L_H(y1, ŷ1) + ωt * L_H(y2, ŷ2)
    """
    def __init__(self, total_epochs, delta=1.0):
        super().__init__()
        self.total_epochs = total_epochs
        self.huber = nn.HuberLoss(delta=delta)
    def forward(self, outputs, targets, epoch):
        huber_power = self.huber(outputs[..., 0], targets[..., 0])
        huber_yd15 = self.huber(outputs[..., 1], targets[..., 1])
        # ωt linearly from 0.1 to 0.5
        weight = 0.4 * (epoch / max(1, self.total_epochs)) + 0.1
        loss = (1 - weight) * huber_power + weight * huber_yd15
        return huber_power, huber_yd15, loss

def calculate_nrmse(y_pred, y_true):
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)
    range_true = np.max(y_true) - np.min(y_true)
    if range_true == 0:
        return np.nan
    nrmse = rmse / range_true
    return nrmse

def calculate_acc_nrmse(y_pred, y_true):
    nrmse = calculate_nrmse(y_pred, y_true)
    if np.isnan(nrmse):
        return np.nan
    acc = 1 - nrmse
    return acc

def calculate_nmae(y_pred, y_true):
    mae = np.mean(np.abs(y_pred - y_true))
    range_true = np.max(y_true) - np.min(y_true)
    if range_true == 0:
        return np.nan
    nmae = mae / range_true
    return nmae

def calculate_acc_nmae(y_pred, y_true):
    nmae = calculate_nmae(y_pred, y_true)
    if np.isnan(nmae):
        return np.nan
    acc = 1 - nmae
    return acc

def calculate_r2(y_pred, y_true):
    try:
        return r2_score(y_true, y_pred)
    except Exception:
        return np.nan