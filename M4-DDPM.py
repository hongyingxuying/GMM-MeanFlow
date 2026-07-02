import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
import random
import logging
from torch.utils.tensorboard import SummaryWriter

from data_pro_FFT_extension import FFTSignalDataset
from U_Net import UNet1D
from utils import *

use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = 0
device = torch.device("cuda:0" if use_cuda else "cpu")

def createPathIfNotExist(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, data_length=1024, gmm_components=3):
        self.noise_steps = noise_steps  # 噪声步骤数
        self.beta_start = beta_start  # 初始 beta
        self.beta_end = beta_end  # 最终 beta
        self.data_length = data_length  # 图像尺寸

        # 准备噪声调度
        self.beta = self.prepare_noise_schedule().to(device)#β[1e-4:0.02,1000]
        self.alpha = (1. - self.beta).to(device)#α[1-β]
        self.alpha_hat = torch.cumprod(self.alpha, dim=0).to(device)#α_hat={α_hat_t=∏α_t}
        # 使用标准高斯噪声，不再依赖 GMM

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)  # 生成一系列 beta 值

    def generate_gmm_noise(self, shape):        
        return torch.randn(shape, device=device)

    def noise_data(self, x, t):
        # 修改为处理1维数据
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t]).to(device)
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t]).to(device)
        epsilon = self.generate_gmm_noise(x.shape)  # 标准高斯噪声
        sqrt_alpha_hat = sqrt_alpha_hat.unsqueeze(1).unsqueeze(1)
        sqrt_one_minus_alpha_hat = sqrt_one_minus_alpha_hat.unsqueeze(1).unsqueeze(1)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon#20个样本加入不同程度的噪声后返回

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))  # 从噪声步骤中随机采样出 n 个时间步

    def sample(self, model, n):#n为生成数据的数量
        logging.info(f"Sampling {n} new generated data....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 1, self.data_length)).to(device)  # 初始化输入噪声数据
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):#从T-1到0反向迭代
                t = (torch.ones(n) * i).long().to(device)  # 时间步 t
                predicted_noise = model(x, t)  # 使用模型预测生成的噪声
                alpha = self.alpha[t]  # 当前 alpha 值
                alpha_hat = self.alpha_hat[t]  # 当前 alpha_hat 值
                beta = self.beta[t]  # 当前 beta 值

                alpha = alpha.unsqueeze(1).unsqueeze(1)
                alpha_hat = alpha_hat.unsqueeze(1).unsqueeze(1)
                beta = beta.unsqueeze(1).unsqueeze(1)

                if i > 1:
                    noise = torch.randn_like(x)  # 生成随机噪声
                else:
                    noise = torch.zeros_like(x)  # 最后一步不需要噪声

                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        # x = (x.clamp(-1, 1) + 1) / 2  # 将像素值缩放到 [0, 1] 范围内
        # x = (x * 255).type(torch.uint8)  # 将像素值转换为整数（0-255）
        return x

