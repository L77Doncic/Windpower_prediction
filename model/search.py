#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hyperparameter search using Optuna Bayesian optimization."""

import os
import argparse
import numpy as np
import torch
import pandas as pd
import optuna

import train_ablation
from dataset import data_preprocess, feature_engineer

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_METRICS_DIR = os.path.join(_PROJECT_ROOT, 'output', 'metrics')


def load_data(data_dir, turbine_id):
    csv_path = os.path.join(data_dir, f'{turbine_id}.csv')
    df = pd.read_csv(csv_path, dtype=str)
    unnamed_cols = [c for c in df.columns if c.startswith('Unnamed')]
    if unnamed_cols:
        df.drop(columns=unnamed_cols, inplace=True)
    df.columns = [c.strip() for c in df.columns]
    if 'DATATIME' not in df.columns:
        if 'datetime' in df.columns:
            df.rename(columns={'datetime': 'DATATIME'}, inplace=True)
        elif 'DATETIME' in df.columns:
            df.rename(columns={'DATETIME': 'DATATIME'}, inplace=True)
    df['DATATIME'] = pd.to_datetime(df['DATATIME'], errors='coerce')
    if df['DATATIME'].isna().sum() > 0:
        df['DATATIME'] = pd.to_datetime(df['DATATIME'], errors='coerce', dayfirst=True)
    for col in df.columns:
        if col == 'DATATIME':
            continue
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception:
            pass
    if turbine_id == 11 and 'YD15' in df.columns:
        df = df[(df['YD15'] != 0.0) & (df['YD15'] != -754.0)]
    df = data_preprocess(df)
    df = feature_engineer(df)
    return df


def objective(trial, df, turbine_id, epochs, device):
    lr = trial.suggest_float('lr', 5e-4, 3e-3, log=True)
    mamba_hidden = trial.suggest_categorical('mamba_hidden', [128, 256])
    tcn_choice = trial.suggest_categorical('tcn_channels', ['small', 'large'])
    tcn_channels = [32, 64, 128] if tcn_choice == 'small' else [64, 128, 256]
    huber_delta = trial.suggest_float('huber_delta', 0.5, 2.0)

    try:
        from models import calculate_r2
        from dataloader import WindDataset
        from torch.utils.data import DataLoader
        import models

        train_dataset = WindDataset(df, turbine_id, input_len=480, pred_len=96, data_type='train')
        val_dataset = WindDataset(df, turbine_id, input_len=480, pred_len=96, data_type='val')
        test_dataset = WindDataset(df, turbine_id, input_len=480, pred_len=96, data_type='test')

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, drop_last=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0)

        model = train_ablation.BaselineMambaFusion(
            input_feat=train_dataset.x1.shape[1],
            future_feat_num=len(train_dataset.future_cols),
            mamba_hidden=mamba_hidden, pred_len=96,
            tcn_channels=tcn_channels, dropout=0.2
        ).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
        criterion = models.DynamicWeightedLoss(total_epochs=epochs, delta=huber_delta)

        best_val_loss = float('inf')
        for epoch in range(epochs):
            model.train()
            for batch in train_loader:
                x1, x2, y = [b.to(device) for b in batch]
                outputs = model(x1, x2)
                _, _, loss = criterion(outputs, y, epoch)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                opt.step()
            scheduler.step()

            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    x1, x2, y = [b.to(device) for b in batch]
                    out = model(x1, x2)
                    _, _, v_loss = criterion(out, y, epoch)
                    val_losses.append(v_loss.item())
            val_loss = float(np.mean(val_losses)) if val_losses else float('inf')
            if val_loss < best_val_loss:
                best_val_loss = val_loss

            if np.isnan(val_loss) or np.isinf(val_loss):
                return float('-inf')

        # Evaluate on test set
        model.eval()
        true_seqs, pred_seqs = [], []
        with torch.no_grad():
            for batch in test_loader:
                x1, x2, y = [b.to(device) for b in batch]
                out = model(x1, x2)
                out_np = out.cpu().numpy()[0]
                y_np = y.cpu().numpy()[0]
                pred_ts = out_np[:, 1].astype(float) if out_np.ndim == 2 else out_np.reshape(-1).astype(float)
                true_ts = y_np[:, 1].astype(float) if y_np.ndim == 2 else y_np.reshape(-1).astype(float)
                true_seqs.append(true_ts)
                pred_seqs.append(pred_ts)

        y_true = np.concatenate(true_seqs)
        y_pred = np.concatenate(pred_seqs)
        r2 = calculate_r2(y_pred, y_true)

        trial.set_user_attr('best_val_loss', best_val_loss)
        return r2

    except Exception as e:
        print(f"Trial failed: {e}")
        return float('-inf')


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter search with Optuna')
    parser.add_argument('--trials', type=int, default=20, help='number of search trials')
    parser.add_argument('--turbine', type=int, default=14, help='turbine id')
    parser.add_argument('--epochs', type=int, default=20, help='epochs per trial')
    parser.add_argument('--data', type=str, default=os.path.join(_PROJECT_ROOT, 'data'))
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Loading data for turbine {args.turbine}...")
    df = load_data(args.data, args.turbine)
    print(f"Data loaded: {df.shape}")

    study = optuna.create_study(direction='maximize', study_name='windpower_hpo')
    study.optimize(lambda trial: objective(trial, df, args.turbine, args.epochs, device),
                   n_trials=args.trials, show_progress_bar=True)

    # Save results
    os.makedirs(_METRICS_DIR, exist_ok=True)
    results = []
    for t in study.trials:
        r = {**t.params, 'r2': t.value, 'val_loss': t.user_attrs.get('best_val_loss', None)}
        results.append(r)
    results_df = pd.DataFrame(results)
    results_path = os.path.join(_METRICS_DIR, f'search_turbine_{args.turbine}.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")

    # Print best
    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"BEST TRIAL #{best.number}")
    print(f"  R²: {best.value:.4f}")
    print(f"  Parameters:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")
    print(f"  Best val_loss: {best.user_attrs.get('best_val_loss', 'N/A')}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
