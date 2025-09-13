import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mamba_ssm import Mamba
from sklearn.metrics import r2_score


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, checkpoint_dir='/root/model/checkpoints'):
        self.val_loss_min = np.inf  # 添加损失记录
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


class ConvBNLayer(nn.Module):
    """卷积+BN+激活层"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, act=None):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride, padding, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = act

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act == 'relu':
            return F.relu(x)
        return x


class ResNetBlock(nn.Module):
    """残差块"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = ConvBNLayer(in_channels, out_channels, 3, stride, 'relu')
        self.conv2 = ConvBNLayer(out_channels, out_channels, 3, act='relu')
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ConvBNLayer(in_channels, out_channels, 1, stride)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return F.relu(x + residual)


class ResNet(nn.Module):
    """ResNet模型"""

    def __init__(self, input_channels=5, output_size=96):
        super().__init__()
        # 使用较小 kernel，使序列长度保持稳定
        self.initial = ConvBNLayer(input_channels, 64, 7, 1, 'relu')
        self.pool = nn.MaxPool1d(3, 2, padding=1)

        # 残差块配置
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, output_size)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = [ResNetBlock(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x expected shape: [B, C_in, seq_len]
        x = self.initial(x)
        x = self.pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)  # [B, channels, 1]
        x = torch.flatten(x, 1)
        return self.fc(x)  # [B, output_size]


class TemporalAttention(nn.Module):
    """时间注意力机制"""

    def __init__(self, hidden_size):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states):
        # hidden_states: [batch, seq_len, hidden_size]
        energy = torch.tanh(self.W(hidden_states))
        attention = F.softmax(self.V(energy), dim=1)  # [B, seq_len, 1]
        attended = torch.sum(attention * hidden_states, dim=1)  # [B, hidden_size]
        return attended


class PositionalEncoding(nn.Module):
    """可学习的位置编码"""

    def __init__(self, d_model, max_len=500):
        super().__init__()
        self.position_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return x + self.position_embedding(positions)


class MambaBlock(nn.Module):
    """用 Mamba 替代 Transformer 的模块"""

    def __init__(self, input_dim, d_model=256, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # 使用 Mamba 替换 Transformer
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=16,  # SSM state expansion factor
                d_conv=4,    # Local convolution width
                expand=2,    # Block expansion factor
            ) for _ in range(num_layers)
        ])

        # 输出保持 d_model 维度，后续 attention 使用该维度
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        x_embed = self.embedding(x)  # [batch, seq, d_model]
        x_pos = self.pos_encoder(x_embed)

        for mamba_layer in self.mamba_layers:
            x_pos = mamba_layer(x_pos)

        return self.proj(x_pos)  # [batch, seq, d_model]


class MultiTaskModel(nn.Module):
    """多任务模型（PyTorch版本）- 生成器"""

    def __init__(self,
                 feat_num=13,
                 future_feat_num=5,
                 tcn_channels=[64, 128, 256],
                 mamba_hidden=256,
                 pred_len=96):
        super().__init__()
        # TCN特征提取 (Conv1d expects [B, C, L])
        self.feat_num = feat_num
        self.pred_len = pred_len
        self.tcn = nn.Sequential(
            ConvBNLayer(feat_num, tcn_channels[0], 9, act='relu'),
            ConvBNLayer(tcn_channels[0], tcn_channels[1], 9, act='relu'),
            ConvBNLayer(tcn_channels[1], tcn_channels[2], 9, act='relu')
        )

        # ========== 替换 Transformer 为 Mamba ==========
        # MambaBlock expects input_dim == tcn_channels[-1] after permuting to [B, seq, C]
        self.mamba = MambaBlock(
            input_dim=tcn_channels[-1],
            d_model=mamba_hidden,
            num_layers=2
        )
        self.attention = TemporalAttention(mamba_hidden)

        # 未来特征处理（ResNet expects [B, C_in, seq_len]）
        self.future_resnet = ResNet(input_channels=future_feat_num, output_size=pred_len)

        # 多任务输出 heads（task1 -> ROUND(A.POWER,0), task2 -> YD15）
        self.task1 = nn.Sequential(
            nn.Linear(mamba_hidden + pred_len, 128),
            nn.ReLU(),
            nn.Linear(128, pred_len)
        )
        self.task2 = nn.Sequential(
            nn.Linear(mamba_hidden + pred_len, 128),
            nn.ReLU(),
            nn.Linear(128, pred_len)
        )

    def forward(self, x1, x2=None):
        """
        x1: [B, input_len, feat_num]
        x2: [B, pred_len, future_feat_num]  （如果有）
        返回: tensor shape (B, pred_len, 2)
        """
        B = x1.size(0)

        # TCN: convert to [B, C, L]
        x = x1.permute(0, 2, 1)  # [B, feat_num, L]
        x = self.tcn(x)  # [B, channels, L]
        x = x.permute(0, 2, 1)  # [B, L, channels]

        # Mamba expect [B, seq_len, input_dim], we set input_dim == channels
        mamba_out = self.mamba(x)  # [B, L, mamba_hidden]

        # Attention aggregate across time -> [B, mamba_hidden]
        attn_vec = self.attention(mamba_out)

        # Future features processed by ResNet to size [B, pred_len]
        if x2 is None:
            # 如果没有未来特征，使用 zeros
            future_feat = torch.zeros((B, self.pred_len), device=x1.device, dtype=x1.dtype)
        else:
            # x2: [B, pred_len, future_feat_num] -> to [B, C, L]
            future_in = x2.permute(0, 2, 1)
            future_feat = self.future_resnet(future_in)  # [B, pred_len]

        # concat attn_vec and future_feat -> 每个样本拼接为 (mamba_hidden + pred_len)
        cat = torch.cat([attn_vec, future_feat], dim=1)  # [B, mamba_hidden + pred_len]

        out1 = self.task1(cat)  # [B, pred_len]
        out2 = self.task2(cat)  # [B, pred_len]

        # 合并为 [B, pred_len, 2]
        out = torch.stack([out1, out2], dim=-1)  # [B, pred_len, 2]
        return out


class Discriminator(nn.Module):
    """GAN的判别器模型"""
    def __init__(self, seq_len, feature_dim=2, hidden_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(seq_len * feature_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len, feature_dim]
        x_flat = x.view(x.size(0), -1)
        return self.model(x_flat)


class DynamicWeightedLoss(nn.Module):
    """动态加权损失函数"""

    def __init__(self, total_epochs, delta=1.0):
        super().__init__()
        self.total_epochs = total_epochs
        self.huber = nn.HuberLoss(delta=delta)

    def forward(self, outputs, targets, epoch):
        # outputs: [B, pred_len, 2], targets: [B, pred_len, 2]
        # 计算两个目标的 Huber 损失
        huber_power = self.huber(outputs[..., 0], targets[..., 0])
        huber_yd15 = self.huber(outputs[..., 1], targets[..., 1])

        # 权重从 0.5 线性增长到 0.9
        weight = 0.4 * (epoch / max(1, self.total_epochs)) + 0.5

        return huber_power, huber_yd15, (1 - weight) * huber_power + weight * huber_yd15


def calculate_nrmse(y_pred, y_true):
    """计算归一化均方根误差（NRMSE）"""
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)
    range_true = np.max(y_true) - np.min(y_true)
    if range_true == 0:
        return np.nan
    nrmse = rmse / range_true
    return nrmse


def calculate_acc_nrmse(y_pred, y_true):
    """计算基于 NRMSE 的准确率（ACC）"""
    nrmse = calculate_nrmse(y_pred, y_true)
    if np.isnan(nrmse):
        return np.nan
    acc = 1 - nrmse
    return acc


def calculate_nmae(y_pred, y_true):
    """计算归一化平均绝对误差（NMAE）"""
    mae = np.mean(np.abs(y_pred - y_true))
    range_true = np.max(y_true) - np.min(y_true)
    if range_true == 0:
        return np.nan
    nmae = mae / range_true
    return nmae


def calculate_acc_nmae(y_pred, y_true):
    """计算基于NMAE的准确率（ACC）"""
    nmae = calculate_nmae(y_pred, y_true)
    if np.isnan(nmae):
        return np.nan
    acc = 1 - nmae
    return acc


def calculate_r2(y_pred, y_true):
    """计算 R2（确保参数顺序为 y_true, y_pred）"""
    try:
        return r2_score(y_true, y_pred)
    except Exception:
        return np.nan
