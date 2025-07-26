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
import torch.nn as nn

import warnings
warnings.filterwarnings('ignore')


def train(df, turbine_id):
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
    adversarial_loss = nn.BCELoss()
    task_loss = nn.MSELoss()

    train_loss_g, train_loss_d = [], []
    train_epochs_loss_g, train_epochs_loss_d = [], []
    valid_epochs_loss = []
    early_stopping = models.EarlyStopping(patience=patience, verbose=True,
                                          checkpoint_dir=f'../checkpoints')

    # 训练
    for epoch in tqdm(range(epoch_num)):
        # =====================Train============================
        epoch_loss_g, epoch_loss_d = [], []
        generator.train()  # 开启训练模式
        discriminator.train()

        for batch_id, data in enumerate(train_loader):
            x1, x2, y = [d.to(device) for d in data]

            # 真实标签和虚假标签
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # -------------------
            #  训练判别器
            # -------------------
            opt_d.zero_grad()

            # 用真实数据训练
            real_loss = adversarial_loss(discriminator(y), real_labels)

            # 用生成数据训练
            generated_y = generator(x1, x2)
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
        avg_epoch_loss_g = np.average(epoch_loss_g)
        avg_epoch_loss_d = np.average(epoch_loss_d)
        train_epochs_loss_g.append(avg_epoch_loss_g)
        train_epochs_loss_d.append(avg_epoch_loss_d)

        # 打印日志
        print(f"epoch={epoch}/{epoch_num} | Loss G: {avg_epoch_loss_g:.4f} | Loss D: {avg_epoch_loss_d:.4f}")

        # =====================valid============================
        generator.eval()
        valid_epoch_loss = []

        with torch.no_grad():
            for data in val_loader:
                x1, x2, y = [d.to(device) for d in data]

                outputs = generator(x1, x2)
                loss = task_loss(outputs, y)
                valid_epoch_loss.append(loss.item())

        val_loss = np.average(valid_epoch_loss)
        valid_epochs_loss.append(val_loss)
        print(f'Valid Loss: {val_loss:.4f}')

        # ==================early stopping======================
        early_stopping(val_loss, model=generator)
        if early_stopping.early_stop:
            print(f"Early stopping at Epoch {epoch - patience}")
            break
    #
    # print('Train & Valid: ')
    # plt.figure(figsize=(12, 3))
    #
    # # 左侧子图：迭代级别的训练损失（所有batch的损失记录）
    # plt.subplot(121)
    # plt.plot(train_loss, label="train")  # 修改点1：移除多余的 [:] 切片
    # plt.title("Training Loss per Iteration")
    # plt.xlabel('Iteration')
    #
    # # 右侧子图：epoch级别的训练/验证损失对比
    # plt.subplot(122)
    # # 修改点2：确保使用正确的变量名 valid_epochs_loss（原代码变量名可能有拼写错误）
    # plt.plot(train_epochs_loss[1:], '-o', label="Train")  # 跳过第一个可能不稳定的epoch
    # plt.plot(valid_epochs_loss[1:], '-o', label="Valid")
    # plt.title("Epoch-wise Loss Comparison")
    # plt.xlabel('Epoch')
    # plt.legend()
    #
    # plt.tight_layout()
    # plt.show()

    # ===================== test ============================
    # 加载模型
    generator = models.MultiTaskModel()
    generator.load_state_dict(
        torch.load(f'../checkpoints/best_model.pth'))

    generator.eval()
    generator.to(device)

    test_loss, test_epoch_mse1, test_epoch_mse2 = [], [], []
    test_accs1, test_accs2 = [], []
    true_y, pred_y, corrected_pred_y = [], [], []

    with torch.no_grad():
        for batch_id, data in enumerate(test_loader):
            x1, x2, y = [d.to(device) for d in data]

            outputs = generator(x1, x2)

            outputs[0] = outputs[0].cpu()
            outputs[1] = outputs[1].cpu()
            y = y.cpu()
            # 计算误差序列
            error_sequence1 = y.numpy().squeeze(0)[:, 0] - outputs[0].numpy().squeeze(0)  # 对ROUND(A.POWER,0)的误差
            error_sequence2 = y.numpy().squeeze(0)[:, 1] - outputs[1].numpy().squeeze(0)  # 对YD15的误差
            original_length_1 = len(error_sequence1)
            original_length_2 = len(error_sequence2)

            # 应用 DMD 修正
            K = 20

            X_data_2 = error_sequence2[:-1]  # 取 t=0~T-2
            Y_data_2 = error_sequence2[1:]  # 取 t=1~T-1
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
            corrected_pred2 = correct_predictions(outputs[1].numpy().squeeze(0), reconstructed_error2)

            # 存储结果
            true_y.extend(y.cpu().numpy().squeeze(0)[:, 1].tolist())
            pred_y.extend(outputs[1].detach().cpu().numpy().squeeze(0).tolist())
            corrected_pred_y.extend(corrected_pred2.tolist())

    # 可视化对比
    plt.figure(figsize=(12, 3))
    plt.plot(true_y[:100], '-o', label="True")
    plt.plot(pred_y[:100], '-o', label="Pred")
    plt.plot(corrected_pred_y[:100], '-o', label="Corrected Pred")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("---------------------------------------------------------")
    print(
        f'ACC of NRMSE between pred and true: {models.calculate_acc_nrmse(np.array(pred_y[:100]), np.array(true_y[:100])):.4f}')
    print(
        f'ACC of NRMSE between corrected_pred and true: {models.calculate_acc_nrmse(np.array(corrected_pred_y[:100]), np.array(true_y[:100])):.4f}')

    print(
        f'ACC of NMAE between pred and true: {models.calculate_acc_nmae(np.array(pred_y[:100]), np.array(true_y[:100])):.4f}')
    print(
        f'ACC of NMAE between corrected_pred and true: {models.calculate_acc_nmae(np.array(corrected_pred_y[:100]), np.array(true_y[:100])):.4f}')

    print(
        f'R2 between pred and true: {models.calculate_r2(np.array(pred_y[:100]), np.array(true_y[:100])):.4f}')
    print(
        f'R2 between corrected_pred and true: {models.calculate_r2(np.array(corrected_pred_y[:100]), np.array(true_y[:100])):.4f}')
    print("---------------------------------------------------------\n\n")


# ===================== 主执行流程 =====================
data_path = '/home/ubuntu/workspace/Wind-power/data'
files = [f for f in os.listdir(data_path) if f.split('.')[0].isdigit()]
files = sorted(files, key=lambda x: int(x.split('.')[0]))

for f in files:
    # 数据加载与预处理
    df = pd.read_csv(os.path.join(data_path, f),
                     parse_dates=['DATATIME'],
                     infer_datetime_format=True)

    turbine_id = int(f.split('.')[0])
    print(f'Processing turbine {turbine_id}')

    # 异常数据处理
    if turbine_id == 11:
        df = df[(df['YD15'] != 0.0) & (df['YD15'] != -754.0)]

    # 执行特征工程
    df = data_preprocess(df)
    df = feature_engineer(df)

    # 模型参数配置
    input_len = 120 * 4
    pred_len = 24 * 4
    epoch_num = 20
    batch_size = 128  # ori:512
    learning_rate = 0.001  # ori:0.15
    patience = 10

    # 启动训练
    train(df, turbine_id)
