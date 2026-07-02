import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
import logging
import random
import math
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
    """Flow-matching helper for 1D signals."""

    def __init__(self, noise_steps=100, data_length=1024, device=None):
        self.noise_steps = noise_steps
        self.data_length = data_length
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def noise_data(self, x0, t):
        t_b = t.view(-1, 1, 1)                     # (B,1,1)
        x1 = torch.randn_like(x0)                  #x1的均值和方差与x0相同 
        v_target = x1 - x0                         # velocity target
        xt = x0 + (v_target) * t_b                 # noisy data at time t
        return xt, v_target

    def sample(self, model, n):
        model.eval()#切换到评估模式
        steps=self.noise_steps
        with torch.no_grad():
            xt = torch.randn((n, 1, self.data_length), device=self.device)
            for i in range(steps):
                si = 1.0 - float(i) / float(steps)
                si1 = 1.0 - float(i + 1) / float(steps)
                ds = si1 - si  # si经历从1到0的变化；si>si1，因此ds为正值

                # Euler
                t0 = torch.full((n,), si, device=self.device)
                v = model(xt, t0)
                xt = xt + ds * v

        model.train()#切换回训练模式
        return xt
