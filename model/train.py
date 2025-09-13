#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

# 导入你自己的模块（请确保这些文件在同一目录或 PYTHONPATH 中）
import models
from dataloader import WindDataset
from dataset import data_preprocess, feature_engineer
from utils import build_hankel_matrix, dmd_decomposition, reconstruct_error, correct_predictions, compute_L

import warnings
warnings.filterwarnings('ignore')


def train(df, turbine_id, input_len, pred_len, epoch_num, batch_size, learning_rate, patience):
    # 创建 Dataset
    train_dataset = WindDataset(df, turbine_id, input_len=input_len, pred_len=pred_len, data_type='train')
    val_dataset = WindDataset(df, turbine_id, input_len=input_len, pred_len=pred_len, data_type='val')
    test_dataset = WindDataset(df, turbine_id, input_len=input_len, pred_len=pred_len, data_type='test')

    print(f'LEN | train_dataset:{len(train_dataset)}, val_dataset:{len(val_dataset)}, test_dataset:{len(test_dataset)}')

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化生成器和判别器
    generator = models.MultiTaskModel().to(device)
    discriminator = models.Discriminator(seq_len=pred_len).to(device)

    # 设置优化器
    opt_g = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # 损失函数
    adversarial_loss = torch.nn.BCELoss()
    task_loss = torch.nn.MSELoss()

    train_epochs_loss_g, train_epochs_loss_d = [], []
    valid_epochs_loss = []
    early_stopping = models.EarlyStopping(patience=patience, verbose=True,
                                          checkpoint_dir=f'../checkpoints')

    # 训练
    for epoch in tqdm(range(epoch_num), desc="Epochs"):
        # =====================Train============================
        epoch_loss_g, epoch_loss_d = [], []
        generator.train()
        discriminator.train()

        for batch_id, data in enumerate(train_loader):
            x1, x2, y = [d.to(device) for d in data]  # x1: [B, input_len, feat], x2: [B, pred_len, future_feat], y: [B, pred_len, 2]

            # 真实标签和虚假标签
            real_labels = torch.ones(x1.size(0), 1).to(device)
            fake_labels = torch.zeros(x1.size(0), 1).to(device)

            # -------------------
            #  训练判别器
            # -------------------
            opt_d.zero_grad()

            # 用真实数据训练
            real_loss = adversarial_loss(discriminator(y), real_labels)

            # 用生成数据训练
            generated_y = generator(x1, x2)  # generator 返回 [B, pred_len, 2]
            fake_loss = adversarial_loss(discriminator(generated_y.detach()), fake_labels)

            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            opt_d.step()

            # -------------------
            #  训练生成器
            # -------------------
            opt_g.zero_grad()

            # GAN损失 + 任务损失
            gan_loss = adversarial_loss(discriminator(generated_y), real_labels)
            t_loss = task_loss(generated_y, y)

            # 结合损失
            g_loss = 0.1 * gan_loss + 0.9 * t_loss

            g_loss.backward()
            opt_g.step()

            epoch_loss_g.append(g_loss.item())
            epoch_loss_d.append(d_loss.item())

        # 计算平均损失
        avg_epoch_loss_g = np.average(epoch_loss_g) if len(epoch_loss_g) > 0 else 0.0
        avg_epoch_loss_d = np.average(epoch_loss_d) if len(epoch_loss_d) > 0 else 0.0
        train_epochs_loss_g.append(avg_epoch_loss_g)
        train_epochs_loss_d.append(avg_epoch_loss_d)

        # 打印日志
        print(f"epoch={epoch+1}/{epoch_num} | Loss G: {avg_epoch_loss_g:.4f} | Loss D: {avg_epoch_loss_d:.4f}")

        # =====================valid============================
        generator.eval()
        valid_epoch_loss = []

        with torch.no_grad():
            for data in val_loader:
                x1, x2, y = [d.to(device) for d in data]
                outputs = generator(x1, x2)
                loss = task_loss(outputs, y)
                valid_epoch_loss.append(loss.item())

        val_loss = np.average(valid_epoch_loss) if len(valid_epoch_loss) > 0 else 0.0
        valid_epochs_loss.append(val_loss)
        print(f'Valid Loss: {val_loss:.4f}')

        # ==================early stopping======================
        early_stopping(val_loss, model=generator)
        if early_stopping.early_stop:
            print(f"Early stopping at Epoch {epoch+1 - patience}")
            break

    # ===================== test ============================
    # 加载 best model（EarlyStopping 已经保存 best 模型到 checkpoint）
    best_checkpoint_path = os.path.join('..', 'checkpoints', 'best_model.pth')
    if os.path.exists(best_checkpoint_path):
        generator = models.MultiTaskModel()
        generator.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
        generator.to(device)
        generator.eval()
    else:
        print(f"Warning: best model not found at {best_checkpoint_path}. Using current generator weights.")

    test_loss, test_epoch_mse1, test_epoch_mse2 = [], [], []
    true_y, pred_y, corrected_pred_y = [], [], []

    with torch.no_grad():
        for batch_id, data in enumerate(test_loader):
            x1, x2, y = [d.to(device) for d in data]

            outputs = generator(x1, x2)  # [B, pred_len, 2]

            # 解析输出
            out_all = outputs.cpu()
            if out_all.ndim == 3 and out_all.shape[-1] >= 2:
                out1 = out_all[:, :, 0]
                out2 = out_all[:, :, 1]
            else:
                out1 = out_all
                out2 = out_all

            y_cpu = y.cpu()

            # error sequences
            error_sequence2 = y_cpu.numpy().squeeze(0)[:, 1] - out2.numpy().squeeze(0)
            original_length_2 = len(error_sequence2)

            # DMD 修正流程（如果序列过短，需要做保护）
            K = 20
            if original_length_2 > 5:
                X_data_2 = error_sequence2[:-1]
                Y_data_2 = error_sequence2[1:]
                try:
                    L = compute_L(X_data_2, K)
                    X2 = build_hankel_matrix(X_data_2, K, L)
                    Y2 = build_hankel_matrix(Y_data_2, K, L)
                    Phi2, eigenvalues2 = dmd_decomposition(X2, Y2)

                    reconstructed_error2 = reconstruct_error(
                        Phi2, eigenvalues2,
                        initial_error=X2[:, 0],
                        K=K,
                        original_length=original_length_2
                    )
                    corrected_pred2 = correct_predictions(out2.numpy().squeeze(0), reconstructed_error2)
                except Exception as e:
                    print(f"DMD correction failed for turbine {turbine_id}, batch {batch_id}: {e}")
                    corrected_pred2 = out2.numpy().squeeze(0)
            else:
                corrected_pred2 = out2.numpy().squeeze(0)

            true_y.extend(y_cpu.numpy().squeeze(0)[:, 1].tolist())
            pred_y.extend(out2.detach().cpu().numpy().squeeze(0).tolist())
            corrected_pred_y.extend(corrected_pred2.tolist())

    # 可视化并保存
    plt.figure(figsize=(12, 3))
    # 只画前 100 个点以便预览
    n_plot = min(100, len(true_y))
    plt.plot(true_y[:n_plot], '-o', label="True")
    plt.plot(pred_y[:n_plot], '-o', label="Pred")
    plt.plot(corrected_pred_y[:n_plot], '-o', label="Corrected Pred")
    plt.legend()
    plt.tight_layout()

    plot_dir = '/root/model/plots'
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(f'{plot_dir}/turbine_{turbine_id}_prediction.png')
    plt.close()

    # 评估指标（依赖你 models 模块中的实现） 
    def safe_arr(a):
        return np.array(a) if len(a) > 0 else np.array([0])

    print("---------------------------------------------------------")
    print(f'ACC of NRMSE between pred and true: {models.calculate_acc_nrmse(safe_arr(pred_y[:n_plot]), safe_arr(true_y[:n_plot])):.4f}')
    print(f'ACC of NRMSE between corrected_pred and true: {models.calculate_acc_nrmse(safe_arr(corrected_pred_y[:n_plot]), safe_arr(true_y[:n_plot])):.4f}')
    print(f'ACC of NMAE between pred and true: {models.calculate_acc_nmae(safe_arr(pred_y[:n_plot]), safe_arr(true_y[:n_plot])):.4f}')
    print(f'ACC of NMAE between corrected_pred and true: {models.calculate_acc_nmae(safe_arr(corrected_pred_y[:n_plot]), safe_arr(true_y[:n_plot])):.4f}')
    print(f'R2 between pred and true: {models.calculate_r2(safe_arr(pred_y[:n_plot]), safe_arr(true_y[:n_plot])):.4f}')
    print(f'R2 between corrected_pred and true: {models.calculate_r2(safe_arr(corrected_pred_y[:n_plot]), safe_arr(true_y[:n_plot])):.4f}')
    print("---------------------------------------------------------\n\n")


# ===================== 主执行流程 =====================
if __name__ == '__main__':
    data_path = '/root/model/data'

    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    files = [f for f in os.listdir(data_path) if f.split('.')[0].isdigit()]
    files = sorted(files, key=lambda x: int(x.split('.')[0]))

    # 全局模型参数（你可以按需要调整）
    input_len = 120 * 4  # 示例
    pred_len = 24 * 4
    epoch_num = 20
    batch_size = 128
    learning_rate = 0.001
    patience = 10

    for f in files:
        csv_path = os.path.join(data_path, f)
        print(f'\nProcessing file: {csv_path}')

        # ---------- robust csv read ----------
        # 先以 string 读取，做更稳健的清洗与解析
        try:
            df = pd.read_csv(csv_path, dtype=str)
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
            continue

        # 删除Unnamed列（Excel导出时多出的索引列）
        unnamed_cols = [c for c in df.columns if c.startswith('Unnamed')]
        if unnamed_cols:
            df.drop(columns=unnamed_cols, inplace=True)

        # 去除列名前后的空白
        df.columns = [c.strip() for c in df.columns]

        # 兼容不同列名形式
        if 'DATATIME' not in df.columns:
            if 'datetime' in df.columns:
                df.rename(columns={'datetime': 'DATATIME'}, inplace=True)
            elif 'DATETIME' in df.columns:
                df.rename(columns={'DATETIME': 'DATATIME'}, inplace=True)

        if 'DATATIME' not in df.columns:
            print(f"File {f} missing DATATIME column. Columns: {df.columns.tolist()}")
            continue

        # 尝试解析时间：先常规解析，再尝试 dayfirst=True（处理 13/1/2022 类型）
        df['DATATIME'] = pd.to_datetime(df['DATATIME'], errors='coerce', infer_datetime_format=True)
        n_nat = df['DATATIME'].isna().sum()
        if n_nat > 0:
            print(f"Warning: {n_nat} DATATIME values could not be parsed with infer format. Trying dayfirst=True.")
            df['DATATIME'] = pd.to_datetime(df['DATATIME'], errors='coerce', dayfirst=True, infer_datetime_format=True)
        n_nat2 = df['DATATIME'].isna().sum()
        if n_nat2 > 0:
            print(f"After dayfirst attempt, {n_nat2} DATATIME still NaT. Sample problematic rows:")
            print(df[df['DATATIME'].isna()].head(10))
            print("Skipping this file for manual inspection.")
            continue

        # 若列是字符串型的数字字段，尝试转换为数值（避免后续处理出错）
        # 保持原有其他列为 string 的话，后面 data_preprocess 有些操作可能失败
        for col in df.columns:
            if col == 'DATATIME':
                continue
            # 尝试转换为 numeric（若不能转换则保持原字符串）
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception:
                pass

        print("Columns after reading:", df.columns.tolist())
        print("DATATIME dtype:", df['DATATIME'].dtype)

        turbine_id = int(f.split('.')[0])
        print(f'Processing turbine {turbine_id}')

        # 如果需要针对特定风机清洗
        if turbine_id == 11:
            if 'YD15' in df.columns:
                df = df[(df['YD15'] != 0.0) & (df['YD15'] != -754.0)]
            else:
                print("Warning: YD15 column not present, skipping turbine-specific filter.")

        # 执行预处理与特征工程
        try:
            df = data_preprocess(df)
            df = feature_engineer(df)
        except Exception as e:
            print(f"Data preprocessing/feature engineering failed for turbine {turbine_id}: {e}")
            continue

        # 启动训练（传入全局参数）
        try:
            train(df, turbine_id,
                  input_len=input_len,
                  pred_len=pred_len,
                  epoch_num=epoch_num,
                  batch_size=batch_size,
                  learning_rate=learning_rate,
                  patience=patience)
        except Exception as e:
            print(f"Training failed for turbine {turbine_id}: {e}")
            # 继续处理下一个文件
            continue

    print("All done.")
