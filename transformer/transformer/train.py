import os
import pandas as pd
import numpy as np
import torch
import models
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import build_hankel_matrix, dmd_decomposition, reconstruct_error, correct_predictions, compute_L
from torch.utils.data import DataLoader
from dataloader import WindDataset
from dataset import data_preprocess, feature_engineer

import warnings
warnings.filterwarnings('ignore')

# ===================== 训练函数 =====================
def train(df, turbine_id, input_len, pred_len, batch_size, learning_rate, epoch_num, patience):
    # 创建 Dataset
    train_dataset = WindDataset(df, turbine_id, input_len=input_len, pred_len=pred_len, data_type='train')
    val_dataset = WindDataset(df, turbine_id, input_len=input_len, pred_len=pred_len, data_type='val')
    test_dataset = WindDataset(df, turbine_id, input_len=input_len, pred_len=pred_len, data_type='test')

    print(f'LEN | train_dataset:{len(train_dataset)}, val_dataset:{len(val_dataset)}, test_dataset:{len(test_dataset)}')

    # 创建 DataLoader
    # 建议调试时先用 num_workers=0，确认逻辑正确后再改为 4
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=2,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=1,
        pin_memory=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.MultiTaskModel()
    model = model.to(device)

    # 设置优化器
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dw_loss = models.DynamicWeightedLoss(epoch_num)

    train_loss = []  # 每个 batch 的 loss
    train_epochs_loss = []  # 每个 epoch 的平均 loss
    valid_epochs_loss = []
    early_stopping = models.EarlyStopping(
        patience=patience,
        verbose=True,
        checkpoint_dir='c:\\Users\\18307\\Desktop\\transformer\\checkpoints'
    )

    # ===================== 训练循环 =====================
    for epoch in tqdm(range(epoch_num), desc="Training Epochs"):
        model.train()
        train_epoch_loss, train_epoch_mse1, train_epoch_mse2 = [], [], []

        for batch_id, data in enumerate(train_loader):
            x1 = data[0].to(device, non_blocking=True)
            x2 = data[1].to(device, non_blocking=True)
            y = data[2].to(device, non_blocking=True)

            opt.zero_grad()
            outputs = model(x1, x2)
            dwl1, dwl2, avg_loss = dw_loss(outputs, y, epoch)
            avg_loss.backward()
            opt.step()

            train_epoch_loss.append(avg_loss.item())
            train_loss.append(avg_loss.item())
            train_epoch_mse1.append(dwl1.item())
            train_epoch_mse2.append(dwl2.item())

        # 计算 epoch 平均 loss
        epoch_loss = np.mean(train_epoch_loss)
        epoch_mse1 = np.mean(train_epoch_mse1)
        epoch_mse2 = np.mean(train_epoch_mse2)
        train_epochs_loss.append(epoch_loss)

        print(f"Epoch {epoch+1}/{epoch_num} | "
              f"Loss: {epoch_loss:.4f}, "
              f"MSE1: {epoch_mse1:.4f}, "
              f"MSE2: {epoch_mse2:.4f}")

        # ===================== 验证 =====================
        model.eval()
        valid_epoch_loss, valid_epochs_mse1, valid_epochs_mse2 = [], [], []

        with torch.no_grad():
            for data in val_loader:
                x1, x2, y, *_ = data
                x1 = x1.to(device, non_blocking=True)
                x2 = x2.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                outputs = model(x1, x2)
                dwl1, dwl2, avg_loss = dw_loss(outputs, y, epoch)
                valid_epoch_loss.append(avg_loss.item())
                valid_epochs_mse1.append(dwl1.item())
                valid_epochs_mse2.append(dwl2.item())

        val_loss = np.mean(valid_epoch_loss)
        val_mse1 = np.mean(valid_epochs_mse1)
        val_mse2 = np.mean(valid_epochs_mse2)
        valid_epochs_loss.append(val_loss)

        print(f'Valid | Loss: {val_loss:.4f} | MSE1: {val_mse1:.4f} | MSE2: {val_mse2:.4f}')

        # ===================== 早停 =====================
        early_stopping(val_loss, model=model)
        if early_stopping.early_stop:
            print(f"Early stopping at Epoch {epoch - patience}")
            break

    # ===================== 训练可视化 =====================
    print('Train & Valid Loss Curves:')
    plt.figure(figsize=(12, 5))

    plt.subplot(121)
    plt.plot(train_loss, label="Train Batch Loss", alpha=0.7)
    plt.title("Training Loss per Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")

    plt.subplot(122)
    epochs = range(1, len(train_epochs_loss) + 1)
    plt.plot(epochs, train_epochs_loss, '-o', label="Train", markersize=3)
    plt.plot(epochs, valid_epochs_loss, '-o', label="Valid", markersize=3)
    plt.title("Epoch-wise Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ===================== 测试 =====================
    print("Loading best model for testing...")
    model = models.MultiTaskModel().to(device)
    model.load_state_dict(torch.load(
        'c:\\Users\\18307\\Desktop\\transformer\\checkpoints\\best_model.pth'
    ))
    model.eval()

    test_loss, test_epoch_mse1, test_epoch_mse2 = [], [], []
    test_accs1, test_accs2 = [], []
    true_y, pred_y, corrected_pred_y = [], [], []

    with torch.no_grad():
        for batch_id, data in enumerate(test_loader):
            x1 = data[0].to(device, non_blocking=True)
            x2 = data[1].to(device, non_blocking=True)
            y = data[2].to(device, non_blocking=True)

            outputs = model(x1, x2)
            # 移到 CPU 以便后续处理
            out1 = outputs[0].cpu().numpy().squeeze(0)
            out2 = outputs[1].cpu().numpy().squeeze(0)
            y_np = y.cpu().numpy().squeeze(0)

            # 提取真实值
            error_seq1 = y_np[:, 0] - out1  # ROUND(A.POWER,0) 误差
            error_seq2 = y_np[:, 1] - out2  # YD15 误差
            original_length = len(error_seq2)

            # DMD 修正 YD15
            K = 20
            X_data = error_seq2[:-1]
            Y_data = error_seq2[1:]
            L = compute_L(X_data, K)

            X_hankel = build_hankel_matrix(X_data, K, L)
            Y_hankel = build_hankel_matrix(Y_data, K, L)
            Phi, eigenvalues = dmd_decomposition(X_hankel, Y_hankel)

            reconstructed_error = reconstruct_error(
                Phi, eigenvalues,
                initial_error=X_hankel[:, 0],
                K=K,
                original_length=original_length
            )
            corrected_pred = correct_predictions(out2, reconstructed_error)

            # 存储结果（只取 YD15）
            true_y.extend(y_np[:, 1].tolist())
            pred_y.extend(out2.tolist())
            corrected_pred_y.extend(corrected_pred.tolist())

    # ===================== 测试结果可视化 =====================
    print("Test Results (First 100 steps):")
    plt.figure(figsize=(12, 4))
    plt.plot(true_y[:100], '-o', label="True", markersize=4)
    plt.plot(pred_y[:100], '-o', label="Pred", markersize=3)
    plt.plot(corrected_pred_y[:100], '-o', label="Corrected", markersize=3)
    plt.legend()
    plt.title("Prediction vs True vs DMD-Corrected")
    plt.xlabel("Time Step")
    plt.ylabel("YD15 Value")
    plt.tight_layout()
    plt.show()

    # ===================== 指标计算 =====================
    from models import calculate_acc_nrmse, calculate_acc_nmae, calculate_r2

    print("---------------------------------------------------------")
    print(f"ACC (NRMSE) | Pred vs True:       {calculate_acc_nrmse(np.array(pred_y[:100]), np.array(true_y[:100])):.4f}")
    print(f"ACC (NRMSE) | Corrected vs True:  {calculate_acc_nrmse(np.array(corrected_pred_y[:100]), np.array(true_y[:100])):.4f}")
    print()
    print(f"ACC (NMAE)  | Pred vs True:       {calculate_acc_nmae(np.array(pred_y[:100]), np.array(true_y[:100])):.4f}")
    print(f"ACC (NMAE)  | Corrected vs True:  {calculate_acc_nmae(np.array(corrected_pred_y[:100]), np.array(true_y[:100])):.4f}")
    print()
    print(f"R²          | Pred vs True:       {calculate_r2(np.array(pred_y[:100]), np.array(true_y[:100])):.4f}")
    print(f"R²          | Corrected vs True:  {calculate_r2(np.array(corrected_pred_y[:100]), np.array(true_y[:100])):.4f}")
    print("---------------------------------------------------------\n\n")


# ===================== 主程序入口 =====================
if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)  # Windows 多进程安全启动

    data_path = r'c:\Users\18307\Desktop\transformer\data'
    files = [f for f in os.listdir(data_path) if f.split('.')[0].isdigit()]
    files = sorted(files, key=lambda x: int(x.split('.')[0]))

    # 全局超参数（可后续改为 argparse）
    INPUT_LEN = 120 * 4   # 120 minutes * 4 (15s interval) = 8 hours
    PRED_LEN = 24 * 4     # 24 minutes * 4 = 1 hour
    EPOCH_NUM = 20
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    PATIENCE = 10

    for f in files:
        file_path = os.path.join(data_path, f)
        df = pd.read_csv(file_path, parse_dates=['DATATIME'], infer_datetime_format=True)
        turbine_id = int(f.split('.')[0])
        print(f"\n🚀 Processing Turbine {turbine_id}")

        # 特殊处理 turbine 11
        if turbine_id == 11:
            df = df[(df['YD15'] != 0.0) & (df['YD15'] != -754.0)]

        # 数据预处理
        df = data_preprocess(df)
        df = feature_engineer(df)

        # 启动训练
        train(
            df=df,
            turbine_id=turbine_id,
            input_len=INPUT_LEN,
            pred_len=PRED_LEN,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            epoch_num=EPOCH_NUM,
            patience=PATIENCE
        )