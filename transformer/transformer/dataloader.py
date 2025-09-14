import torch
import joblib
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')



class WindDataset(Dataset):
    def __init__(self, data, tid, data_type='train',
                 train_ratio=0.7, val_ratio=0.2,
                 input_len=24 * 4 * 5, pred_len=24 * 4, stride=19 * 4):
        """
        Args:
            data: 完整数据集 DataFrame
            tid: 风机编号，用于保存scaler
            data_type: train/val/test
            train_ratio: 训练集比例
            val_ratio: 验证集比例
        """
        self.data_type = data_type
        self.tid = tid
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.input_len = input_len
        self.pred_len = pred_len
        self.stride = stride

        # 全局数据划分边界
        num_samples = len(data)
        self.num_train = int(num_samples * train_ratio)
        self.num_val = int(num_samples * val_ratio)
        self.num_test = num_samples - self.num_train - self.num_val

        # 根据数据类型获取切片范围
        if data_type == 'train':
            self.border1 = 0
            self.border2 = self.num_train
        elif data_type == 'val':
            self.border1 = self.num_train
            self.border2 = self.num_train + self.num_val
        else:  # test
            self.border1 = self.num_train + self.num_val
            self.border2 = num_samples

        # 特征列配置
        self.use_cols = ['WINDSPEED', 'PREPOWER', 'WINDDIRECTION', 'TEMPERATURE', 'HUMIDITY',
                         'PRESSURE', 'ROUND(A.WS,1)', 'ROUND(A.POWER,0)', 'YD15',
                         'month', 'day', 'hour', 'minute']  # 你的特征列
        self.future_cols = ['WINDSPEED', 'WINDDIRECTION', 'TEMPERATURE', 'HUMIDITY', 'PRESSURE']  # 未来特征
        self.target_cols = ['ROUND(A.POWER,0)', 'YD15']

        # 执行数据转换
        self.transform(data)

    def transform(self, df):
        # 仅训练集拟合scaler并保存
        if self.data_type == 'train':
            # 初始化scaler
            self.scaler_x1 = StandardScaler()
            self.scaler_x2 = StandardScaler()
            self.scaler_y = StandardScaler()

            # 使用训练集数据拟合
            train_data = df.iloc[self.border1:self.border2]
            self.scaler_x1.fit(train_data[self.use_cols].values)
            self.scaler_x2.fit(train_data[self.future_cols].values)
            self.scaler_y.fit(train_data[self.target_cols].values)

            # 保存scaler
            joblib.dump(self.scaler_x1, f'output\\scaler_{self.tid}_x1.pkl')
            joblib.dump(self.scaler_x2, f'output\\scaler_{self.tid}_x2.pkl')
            joblib.dump(self.scaler_y, f'output\\scaler_{self.tid}_y.pkl')
        else:
            # 加载训练集保存的scaler
            self.scaler_x1 = joblib.load(f'output\\scaler_{self.tid}_x1.pkl')
            self.scaler_x2 = joblib.load(f'output\\scaler_{self.tid}_x2.pkl')
            self.scaler_y = joblib.load(f'output\\scaler_{self.tid}_y.pkl')

        # 对所有数据应用归一化（使用训练集的scaler参数）
        x1_norm = self.scaler_x1.transform(df[self.use_cols].values)
        x2_norm = self.scaler_x2.transform(df[self.future_cols].values)
        y_norm = self.scaler_y.transform(df[self.target_cols].values)

        # 根据数据类型切片
        self.x1 = torch.FloatTensor(x1_norm[self.border1:self.border2])
        self.x2 = torch.FloatTensor(x2_norm[self.border1:self.border2])
        self.y = torch.FloatTensor(y_norm[self.border1:self.border2])

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.input_len
        r_begin = s_end + self.stride
        r_end = r_begin + self.pred_len

        # 输入序列 (历史特征)
        seq_x1 = self.x1[s_begin:s_end]

        # 未来特征 (预测时段的已知特征)
        seq_x2 = self.x2[r_begin:r_end]

        # 目标值 (预测时段的真实值)
        seq_y = self.y[r_begin:r_end]

        return seq_x1, seq_x2, seq_y

    def __len__(self):
        return len(self.x1) - self.input_len - self.stride - self.pred_len + 1

    def inverse_transform(self, y_norm):
        """将归一化后的预测值转换回原始量纲"""
        return self.scaler_y.inverse_transform(y_norm)
