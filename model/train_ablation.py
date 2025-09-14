#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_ablation.py

Supports three ablation settings (baseline = single Mamba block):
  - mode=dmd    : baseline + DMD correction in test stage
  - mode=fusion : baseline + Fusion (use future features via ResNet and fuse)
  - mode=gan    : baseline + GAN (train with discriminator + adversarial loss)

Usage:
  python train_ablation.py --mode dmd
  python train_ablation.py --mode fusion
  python train_ablation.py --mode gan
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd


# reuse your modules
import models
from dataloader import WindDataset
from dataset import data_preprocess, feature_engineer
from utils import build_hankel_matrix, dmd_decomposition, reconstruct_error, correct_predictions, compute_L

import warnings
warnings.filterwarnings('ignore')


# ---------------------------
# Baseline models (Mamba-only)
# ---------------------------
class BaselineMamba(nn.Module):
    """Baseline: only MambaBlock + temporal pooling + simple heads"""
    def __init__(self, input_feat=13, mamba_hidden=256, pred_len=96, mamba_layers=2):
        super().__init__()
        # reuse MambaBlock from your models.py
        # It expects input_dim equal to feature dimension
        self.mamba_block = models.MambaBlock(input_dim=input_feat, d_model=mamba_hidden, num_layers=mamba_layers)
        self.attn = models.TemporalAttention(mamba_hidden)
        self.head1 = nn.Sequential(nn.Linear(mamba_hidden, 128), nn.ReLU(), nn.Linear(128, pred_len))
        self.head2 = nn.Sequential(nn.Linear(mamba_hidden, 128), nn.ReLU(), nn.Linear(128, pred_len))
        self.pred_len = pred_len

    def forward(self, x1, x2=None):
        # x1: [B, input_len, feat]
        m_out = self.mamba_block(x1)  # [B, seq_len, mamba_hidden]
        attn_vec = self.attn(m_out)   # [B, mamba_hidden]
        out1 = self.head1(attn_vec)
        out2 = self.head2(attn_vec)
        out = torch.stack([out1, out2], dim=-1)  # [B, pred_len, 2]
        return out


class BaselineMambaFusion(nn.Module):
    """Baseline + Fusion: Mamba + future ResNet fusion (concatenate attn vec + future feature)"""
    def __init__(self, input_feat=13, future_feat_num=5, mamba_hidden=256, pred_len=96, mamba_layers=2):
        super().__init__()
        self.mamba_block = models.MambaBlock(input_dim=input_feat, d_model=mamba_hidden, num_layers=mamba_layers)
        self.attn = models.TemporalAttention(mamba_hidden)
        # reuse ResNet from models.py for future features
        self.future_resnet = models.ResNet(input_channels=future_feat_num, output_size=pred_len)
        # fusion heads
        self.head1 = nn.Sequential(nn.Linear(mamba_hidden + pred_len, 128), nn.ReLU(), nn.Linear(128, pred_len))
        self.head2 = nn.Sequential(nn.Linear(mamba_hidden + pred_len, 128), nn.ReLU(), nn.Linear(128, pred_len))
        self.pred_len = pred_len

    def forward(self, x1, x2=None):
        # x1: [B, input_len, feat], x2: [B, pred_len, future_feat]
        m_out = self.mamba_block(x1)
        attn_vec = self.attn(m_out)  # [B, mamba_hidden]
        if x2 is None:
            future_feat = torch.zeros((x1.size(0), self.pred_len), device=x1.device, dtype=x1.dtype)
        else:
            future_in = x2.permute(0, 2, 1)  # [B, C, L]
            future_feat = self.future_resnet(future_in)  # [B, pred_len]
        cat = torch.cat([attn_vec, future_feat], dim=1)  # [B, mamba_hidden + pred_len]
        out1 = self.head1(cat)
        out2 = self.head2(cat)
        out = torch.stack([out1, out2], dim=-1)
        return out


# reuse Discriminator from your models.py (if present). If not, define a minimal one:
if hasattr(models, 'Discriminator'):
    Discriminator = models.Discriminator
else:
    class Discriminator(nn.Module):
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
            x_flat = x.view(x.size(0), -1)
            return self.model(x_flat)


# ---------------------------
# Training / Evaluation utils
# ---------------------------
def eval_and_save_plots(true_arr, pred_arr, corr_arr, turbine_id, plot_head_n=100):
    """
    true_arr/pred_arr/corr_arr: numpy arrays flattened (1D) for plotting and metrics.
    Also saves per-horizon metrics and plot.
    """
    os.makedirs('/root/model/plots', exist_ok=True)
    os.makedirs('/root/model/metrics', exist_ok=True)

    n_plot = min(plot_head_n, true_arr.size)
    # plot head
    plt.figure(figsize=(12, 3))
    plt.plot(true_arr[:n_plot], '-o', label='True')
    plt.plot(pred_arr[:n_plot], '-o', label='Pred')
    plt.plot(corr_arr[:n_plot], '-o', label='Corrected Pred')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'/root/model/plots/turbine_{turbine_id}_ablation_pred.png')
    plt.close()

    # compute & print metrics
    def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2))) if a.size > 0 else float('nan')
    def mae(a, b): return float(np.mean(np.abs(a - b))) if a.size > 0 else float('nan')

    print("=== Metrics (HEAD) ===")
    print(f'RMSE pred: {rmse(pred_arr[:n_plot], true_arr[:n_plot]):.4f}, RMSE corr: {rmse(corr_arr[:n_plot], true_arr[:n_plot]):.4f}')
    print(f'MAE  pred: {mae(pred_arr[:n_plot], true_arr[:n_plot]):.4f}, MAE  corr: {mae(corr_arr[:n_plot], true_arr[:n_plot]):.4f}')
    try:
        print('ACC NRMSE pred:', models.calculate_acc_nrmse(pred_arr[:n_plot], true_arr[:n_plot]))
        print('ACC NRMSE corr:', models.calculate_acc_nrmse(corr_arr[:n_plot], true_arr[:n_plot]))
        print('ACC NMAE pred:', models.calculate_acc_nmae(pred_arr[:n_plot], true_arr[:n_plot]))
        print('R2 pred:', models.calculate_r2(pred_arr[:n_plot], true_arr[:n_plot]))
    except Exception:
        pass

    print("=== Metrics (FULL) ===")
    print(f'RMSE pred: {rmse(pred_arr, true_arr):.4f}, RMSE corr: {rmse(corr_arr, true_arr):.4f}')
    print(f'MAE  pred: {mae(pred_arr, true_arr):.4f}, MAE  corr: {mae(corr_arr, true_arr):.4f}')
    try:
        print('ACC NRMSE pred:', models.calculate_acc_nrmse(pred_arr, true_arr))
        print('ACC NRMSE corr:', models.calculate_acc_nrmse(corr_arr, true_arr))
        print('ACC NMAE pred:', models.calculate_acc_nmae(pred_arr, true_arr))
        print('R2 pred:', models.calculate_r2(pred_arr, true_arr))
    except Exception:
        pass

    # per-horizon metrics (if shapes allow)
    # we expect we also have shapes (N_samples, pred_len) somewhere; try to save if available
    # caller can separately save per-horizon csv


# ---------------------------
# Main training function
# ---------------------------
def train_ablation(df, turbine_id,
                   mode='dmd',   # 'dmd' | 'fusion' | 'gan'
                   input_len=120*4, pred_len=24*4,
                   epoch_num=20, batch_size=128, lr=1e-3, patience=10,
                   num_workers=4, device=None):
    """
    mode:
      - 'dmd' : baseline + DMD in test (no GAN, no fusion)
      - 'fusion': baseline + fusion (uses future features)
      - 'gan' : baseline + GAN training (no fusion)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Datasets
    train_dataset = WindDataset(df, turbine_id, input_len=input_len, pred_len=pred_len, data_type='train')
    val_dataset = WindDataset(df, turbine_id, input_len=input_len, pred_len=pred_len, data_type='val')
    test_dataset = WindDataset(df, turbine_id, input_len=input_len, pred_len=pred_len, data_type='test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0)

    # build model according to mode
    if mode == 'fusion':
        model = BaselineMambaFusion(input_feat=train_dataset.x1.shape[1] if len(train_dataset.x1.shape)>1 else 13,
                                    future_feat_num=len(train_dataset.future_cols) if hasattr(train_dataset, 'future_cols') else 5,
                                    mamba_hidden=256, pred_len=pred_len).to(device)
    else:
        # baseline for dmd and gan
        input_feat = train_dataset.x1.shape[1] if len(train_dataset.x1.shape)>1 else 13
        model = BaselineMamba(input_feat=input_feat, mamba_hidden=256, pred_len=pred_len).to(device)

    # discriminator for GAN mode
    if mode == 'gan':
        discriminator = Discriminator(seq_len=pred_len).to(device)
        opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    else:
        discriminator = None

    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    adv_loss_fn = nn.BCELoss()
    task_loss_fn = nn.MSELoss()

    early_stopper = models.EarlyStopping(patience=patience, verbose=True, checkpoint_dir='/root/model/checkpoints')

    # training loop
    for epoch in range(epoch_num):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            x1, x2, y = [b.to(device) for b in batch]  # x2 might be used in fusion
            generated = model(x1, x2 if mode == 'fusion' else None)  # [B, pred_len, 2]
            t_loss = task_loss_fn(generated, y)

            if mode == 'gan':
                # train discriminator
                discriminator.train()
                opt_d.zero_grad()
                bs = y.size(0)
                real_lbl = torch.ones(bs, 1, device=device)
                fake_lbl = torch.zeros(bs, 1, device=device)

                try:
                    real_out = discriminator(y)
                    fake_out = discriminator(generated.detach())
                except Exception:
                    real_out = discriminator(y.float())
                    fake_out = discriminator(generated.detach().float())

                d_loss = (adv_loss_fn(real_out, real_lbl) + adv_loss_fn(fake_out, fake_lbl)) * 0.5
                d_loss.backward()
                opt_d.step()

                # train generator with adversarial + task loss
                opt.zero_grad()
                try:
                    adv_g = adv_loss_fn(discriminator(generated), real_lbl)
                except Exception:
                    adv_g = adv_loss_fn(discriminator(generated.float()), real_lbl)
                g_loss = 0.1 * adv_g + 0.9 * t_loss
                g_loss.backward()
                opt.step()
                epoch_losses.append(g_loss.item())
            else:
                # pure supervised training
                opt.zero_grad()
                t_loss.backward()
                opt.step()
                epoch_losses.append(t_loss.item())

        # validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x1, x2, y = [b.to(device) for b in batch]
                out = model(x1, x2 if mode == 'fusion' else None)
                val_losses.append(task_loss_fn(out, y).item())
        val_loss = float(np.mean(val_losses)) if len(val_losses)>0 else 0.0
        print(f"Epoch {epoch+1}/{epoch_num} | train_loss: {np.mean(epoch_losses):.6f} | val_loss: {val_loss:.6f}")
        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break

    # load best model
    best_path = '/root/model/checkpoints/best_model.pth'
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        model.to(device)
        model.eval()

    # ---------- test & (optional) DMD ----------
    true_seqs, pred_seqs, corr_seqs = [], [], []
    with torch.no_grad():
        for batch_id, batch in enumerate(test_loader):
            x1, x2, y = [b.to(device) for b in batch]
            out = model(x1, x2 if mode == 'fusion' else None)  # [1, pred_len, 2]
            out_np = out.cpu().numpy()[0]
            y_np = y.cpu().numpy()[0]

            # choose second channel like before
            if out_np.ndim==2 and out_np.shape[-1] >= 2:
                pred_ts = out_np[:, 1].astype(float)
            else:
                pred_ts = out_np.reshape(-1).astype(float)

            if y_np.ndim==2 and y_np.shape[-1] >= 2:
                true_ts = y_np[:, 1].astype(float)
            else:
                true_ts = y_np.reshape(-1).astype(float)

            # DMD correction if requested (mode == 'dmd')
            if mode == 'dmd':
                try:
                    error_sequence = true_ts - pred_ts
                    original_length = len(error_sequence)
                    K = 20
                    if original_length > 5:
                        Xd = error_sequence[:-1]
                        Yd = error_sequence[1:]
                        L = compute_L(Xd, K)
                        X2 = build_hankel_matrix(Xd, K, L)
                        Y2 = build_hankel_matrix(Yd, K, L)
                        Phi, eigs = dmd_decomposition(X2, Y2)
                        recon_err = reconstruct_error(Phi, eigs, initial_error=X2[:, 0], K=K, original_length=original_length)
                        corr_pred = correct_predictions(pred_ts, recon_err)
                    else:
                        corr_pred = pred_ts.copy()
                except Exception as e:
                    print(f"DMD correction failed sample {batch_id}: {e}")
                    corr_pred = pred_ts.copy()
            else:
                corr_pred = pred_ts.copy()

            true_seqs.append(true_ts)
            pred_seqs.append(pred_ts)
            corr_seqs.append(corr_pred)

    # stack to arrays
    true_arr2d = np.stack(true_seqs, axis=0)   # (N_samples, pred_len)
    pred_arr2d = np.stack(pred_seqs, axis=0)
    corr_arr2d = np.stack(corr_seqs, axis=0)

    # flatten for plotting / summary
    y_true_flat = true_arr2d.ravel()
    y_pred_flat = pred_arr2d.ravel()
    y_corr_flat = corr_arr2d.ravel()

    # save per-horizon metrics
    per_rmse = np.sqrt(np.mean((pred_arr2d - true_arr2d) ** 2, axis=0))
    per_mae  = np.mean(np.abs(pred_arr2d - true_arr2d), axis=0)
    per_rmse_corr = np.sqrt(np.mean((corr_arr2d - true_arr2d) ** 2, axis=0))
    per_mae_corr  = np.mean(np.abs(corr_arr2d - true_arr2d), axis=0)
    horizon_df = {
        'rmse_pred': per_rmse,
        'mae_pred': per_mae,
        'rmse_corr': per_rmse_corr,
        'mae_corr': per_mae_corr
    }
    import pandas as pd
    hdf = pd.DataFrame(horizon_df)
    os.makedirs('/root/model/metrics', exist_ok=True)
    hdf.to_csv(f'/root/model/metrics/turbine_{turbine_id}_ablation_per_horizon.csv', index=True)
    print(f"Saved per-horizon CSV to /root/model/metrics/turbine_{turbine_id}_ablation_per_horizon.csv")

    # final eval & plot
    eval_and_save_plots(y_true_flat, y_pred_flat, y_corr_flat, turbine_id, plot_head_n=100)


# ---------------------------
# CLI entry
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='baseline', choices=['baseline','dmd', 'fusion', 'gan'],
                    help='ablation mode: baseline|dmd|fusion|gan')
    parser.add_argument('--data', type=str, default='/root/model/data', help='data directory')
    parser.add_argument('--turbine', type=int, default=None, help='single turbine id to run (default: run all files)')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    files = []
    if args.turbine is not None:
        files = [f'{args.turbine}.csv']
    else:
        files = [f for f in os.listdir(args.data) if f.split('.')[0].isdigit()]
        files = sorted(files, key=lambda x: int(x.split('.')[0]))

    for f in files:
        csv_path = os.path.join(args.data, f)
        print(f'Processing {csv_path} mode={args.mode}')
        try:
            df = pd.read_csv(csv_path, dtype=str)
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
            continue

        # basic cleaning like your original train.py
        unnamed_cols = [c for c in df.columns if c.startswith('Unnamed')]
        if unnamed_cols:
            df.drop(columns=unnamed_cols, inplace=True)
        df.columns = [c.strip() for c in df.columns]
        if 'DATATIME' not in df.columns:
            if 'datetime' in df.columns:
                df.rename(columns={'datetime': 'DATATIME'}, inplace=True)
            elif 'DATETIME' in df.columns:
                df.rename(columns={'DATETIME': 'DATATIME'}, inplace=True)
        df['DATATIME'] = pd.to_datetime(df['DATATIME'], errors='coerce', infer_datetime_format=True)
        if df['DATATIME'].isna().sum() > 0:
            df['DATATIME'] = pd.to_datetime(df['DATATIME'], errors='coerce', dayfirst=True, infer_datetime_format=True)
        # convert numerics
        for col in df.columns:
            if col == 'DATATIME': continue
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception:
                pass

        tid = int(f.split('.')[0])
        # turbine-specific quick filter
        if tid == 11 and 'YD15' in df.columns:
            df = df[(df['YD15'] != 0.0) & (df['YD15'] != -754.0)]

        try:
            df = data_preprocess(df)
            df = feature_engineer(df)
        except Exception as e:
            print(f"preprocess/feature failed for {tid}: {e}")
            continue

        # run training for this file/turbine
        try:
            train_ablation(df, turbine_id=tid, mode=args.mode,
                           input_len=120*4, pred_len=24*4,
                           epoch_num=args.epochs, batch_size=args.batch, lr=args.lr)
        except Exception as e:
            print(f"Training failed for {tid}: {e}")
            continue

    print("All done.")


if __name__ == '__main__':
    main()
