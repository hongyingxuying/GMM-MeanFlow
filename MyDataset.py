from torch.utils.data import Dataset
from scipy.io import loadmat
import random
import numpy as np
import torch


class MyDataset(Dataset):
    def __init__(self, mat_paths):
        self.alldata = []
        self.alltargets = []

        # 加载数据并做逐样本 min-max 归一化（与 FFTSignalDataset 保持一致）
        for label, mat_path in enumerate(mat_paths):
            mat_data = loadmat(mat_path)
            data = np.array(mat_data["result"])  # (N, 1, L) 或 (N, L)
            data = np.squeeze(data)
            data = data.reshape(data.shape[0], -1)
            mins = data.min(axis=1, keepdims=True)
            maxs = data.max(axis=1, keepdims=True)
            denom = maxs - mins
            data = (data - mins) / (denom + 1e-8)

            self.alldata.extend(data)
            self.alltargets.extend([label] * len(data))

        # 打乱数据和标签   
             
        combined = list(zip(self.alldata, self.alltargets))
        random.shuffle(combined)
        self.alldata, self.alltargets = zip(*combined)
        

    def __getitem__(self, index):
        x = torch.tensor(self.alldata[index], dtype=torch.float32)
        y = torch.tensor(self.alltargets[index], dtype=torch.long)
        return x, y

    def __len__(self):
        return len(self.alldata)
