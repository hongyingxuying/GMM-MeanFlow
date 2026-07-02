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
from sklearn.mixture import GaussianMixture
from data_pro_FFT_extension import FFTSignalDataset
from U_Net import UNet1D
from utils import *
import numpy as np

use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = 0
device = torch.device("cuda:0" if use_cuda else "cpu")


def createPathIfNotExist(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path


class Diffusion:
    """Flow-matching helper for 1D signals with GMM prior noise."""

    def __init__(self, noise_steps=100, data_length=1024, device=None, gmm_components=5):
        self.noise_steps = noise_steps
        self.data_length = data_length
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.gmm_components = gmm_components
        self.gmm_max_components = 8
        self.gmm_reg_covar = 1e-5
        self.gmm_fitted = False
        self.gmm_means_np = None
        self.gmm_covariances_np = None
        self.gmm_weights_np = None
        self.gmm_data_mean_np = None
        self.gmm_data_std_np = None
        self.gmm_means = None
        self.gmm_covariances = None
        self.gmm_weights = None
        self.gmm_data_mean = None
        self.gmm_data_std = None

    def fit_gmm(self, data_samples):
        if isinstance(data_samples, torch.Tensor):
            data_samples = data_samples.detach().cpu().numpy()

        data_samples = np.asarray(data_samples)
        if data_samples.ndim == 3:
            n_samples, _, data_length = data_samples.shape
            data_flat = data_samples.reshape(n_samples, data_length)
        elif data_samples.ndim == 2:
            n_samples, data_length = data_samples.shape
            data_flat = data_samples
        else:
            raise ValueError("data_samples must have shape (N, 1, L) or (N, L)")

        if n_samples < 2:
            raise ValueError("Need at least 2 samples to fit GMM")

        data_flat = data_flat.astype(np.float64)
        self.gmm_data_mean_np = data_flat.mean(axis=0, keepdims=True)
        self.gmm_data_std_np = data_flat.std(axis=0, keepdims=True)
        self.gmm_data_std_np = np.maximum(self.gmm_data_std_np, 1e-6)
        data_norm = (data_flat - self.gmm_data_mean_np) / self.gmm_data_std_np

        max_components = int(max(1, min(self.gmm_max_components, n_samples)))
        candidate_components = range(1, max_components + 1)
        best_bic = np.inf
        best_gmm = None

        for n_components in candidate_components:
            try:
                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type='diag',
                    init_params='random',
                    n_init=10,
                    max_iter=300,
                    random_state=42,
                    reg_covar=self.gmm_reg_covar,
                    tol=1e-4
                )
                gmm.fit(data_norm)
                bic = gmm.bic(data_norm)
                if bic < best_bic:
                    best_bic = bic
                    best_gmm = gmm
            except Exception:
                continue

        if best_gmm is None:
            n_components = int(max(1, min(self.gmm_components, n_samples)))
            best_gmm = GaussianMixture(
                n_components=n_components,
                covariance_type='diag',
                init_params='random',
                n_init=10,
                max_iter=300,
                random_state=42,
                reg_covar=self.gmm_reg_covar,
                tol=1e-4
            )
            best_gmm.fit(data_norm)

        self.gmm_components = int(best_gmm.weights_.shape[0])
        self.gmm_means_np = best_gmm.means_.astype(np.float64)
        self.gmm_covariances_np = best_gmm.covariances_.astype(np.float64)
        self.gmm_weights_np = best_gmm.weights_.astype(np.float64)

        self.gmm_data_mean = torch.from_numpy(self.gmm_data_mean_np.astype(np.float32)).to(self.device)
        self.gmm_data_std = torch.from_numpy(self.gmm_data_std_np.astype(np.float32)).to(self.device)
        self.gmm_means = torch.from_numpy(self.gmm_means_np.astype(np.float32)).to(self.device)
        self.gmm_covariances = torch.from_numpy(self.gmm_covariances_np.astype(np.float32)).to(self.device)
        self.gmm_weights = torch.from_numpy(self.gmm_weights_np.astype(np.float32)).to(self.device)
        self.gmm_fitted = True

        print(f"  GMM fitted: components={self.gmm_components}, BIC={best_bic:.2f}")

    def save_gmm_params(self, save_path):
        if not self.gmm_fitted:
            raise RuntimeError("GMM is not fitted, cannot save parameters")
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        np.savez(
            save_path,
            means=self.gmm_means_np,
            covariances=self.gmm_covariances_np,
            weights=self.gmm_weights_np,
            data_mean=self.gmm_data_mean_np,
            data_std=self.gmm_data_std_np,
        )

    def load_gmm_params(self, load_path):
        params = np.load(load_path, allow_pickle=False)
        self.gmm_means_np = params['means'].astype(np.float64)
        self.gmm_covariances_np = params['covariances'].astype(np.float64)
        self.gmm_weights_np = params['weights'].astype(np.float64)
        if 'data_mean' in params and 'data_std' in params:
            self.gmm_data_mean_np = params['data_mean'].astype(np.float64)
            self.gmm_data_std_np = params['data_std'].astype(np.float64)
        else:
            self.gmm_data_mean_np = np.zeros((1, self.gmm_means_np.shape[1]), dtype=np.float64)
            self.gmm_data_std_np = np.ones((1, self.gmm_means_np.shape[1]), dtype=np.float64)
        self.gmm_components = int(self.gmm_weights_np.shape[0])

        self.gmm_data_mean = torch.from_numpy(self.gmm_data_mean_np.astype(np.float32)).to(self.device)
        self.gmm_data_std = torch.from_numpy(self.gmm_data_std_np.astype(np.float32)).to(self.device)
        self.gmm_means = torch.from_numpy(self.gmm_means_np.astype(np.float32)).to(self.device)
        self.gmm_covariances = torch.from_numpy(self.gmm_covariances_np.astype(np.float32)).to(self.device)
        self.gmm_weights = torch.from_numpy(self.gmm_weights_np.astype(np.float32)).to(self.device)
        self.gmm_fitted = True

    def _sample_gmm_noise(self, shape, device):        
        batch_size, _, data_length = shape

        if not self.gmm_fitted:
            return torch.randn((batch_size, 1, data_length), device=device)

        component_idx = np.random.choice(
            self.gmm_components,
            size=batch_size,
            p=self.gmm_weights_np
        )

        noise_samples = []
        for idx in component_idx:
            mean = self.gmm_means[idx].to(device)
            var = self.gmm_covariances[idx].to(device)
            std = torch.sqrt(var + 1e-8)
            eps = torch.randn(data_length, device=device)
            sample_norm = mean + std * eps
            if self.gmm_data_mean is not None and self.gmm_data_std is not None:
                sample = sample_norm * self.gmm_data_std.squeeze(0).to(device) + self.gmm_data_mean.squeeze(0).to(device)
            else:
                sample = sample_norm
            noise_samples.append(sample)

        total_noise = torch.stack(noise_samples, dim=0).unsqueeze(1)
        return total_noise

    def noise_data(self, x0, t):
        t_b = t.view(-1, 1, 1)                     # (B,1,1)
        x1 = self._sample_gmm_noise(x0.shape, x0.device)  # Batch x 1 x data_length
        v_target = x1 - x0                         # velocity target
        xt = x0 + (v_target) * t_b                 # noisy data at time t
        return xt, v_target

    def sample(self,model,n,steps=None,sampler: str = None,heun_warmup: int = 20,
        early_stop: bool = True,early_stop_tol: float = 1e-4,
        early_stop_patience: int = 3,early_stop_check_every: int = 5,):

        model.eval()
        steps = max(1, int(steps if steps is not None else self.noise_steps))
        sampler = (sampler or "hybrid").lower()

        heun_warmup = int(max(0, min(heun_warmup, steps)))
        early_stop_patience = int(max(1, early_stop_patience))
        early_stop_check_every = int(max(1, early_stop_check_every))

        # Reuse time tensors to reduce allocations
        t0 = torch.empty((n,), device=self.device)
        t1 = torch.empty((n,), device=self.device)
        t_mid = torch.empty((n,), device=self.device)

        use_amp = (self.device.type == "cuda")
        amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                xt = self._sample_gmm_noise((n, 1, self.data_length), self.device)  # 初始噪声
                v_prev = None  # for AB2
                small_update_hits = 0

                for i in range(steps):
                    ti = 1.0 - float(i) / float(steps)
                    ti1 = 1.0 - float(i + 1) / float(steps)
                    ds = ti1 - ti

                    t0.fill_(ti)
                    t1.fill_(ti1)

                    # Choose integrator
                    if sampler == "hybrid":
                        # warmup with Heun, then AB2
                        if i < heun_warmup:
                            v0 = model(xt, t0)
                            x_euler = xt + ds * v0
                            v1 = model(x_euler, t1)
                            v_eff = 0.5 * (v0 + v1)
                            xt = xt + ds * v_eff
                        else:
                            v = model(xt, t0)
                            if v_prev is None:
                                v_eff = v
                            else:
                                v_eff = 1.5 * v - 0.5 * v_prev
                            xt = xt + ds * v_eff
                            v_prev = v

                    elif sampler == "euler":
                        v_eff = model(xt, t0)
                        xt = xt + ds * v_eff

                    elif sampler in ("heun", "rk2"):
                        v0 = model(xt, t0)
                        x_euler = xt + ds * v0
                        v1 = model(x_euler, t1)
                        v_eff = 0.5 * (v0 + v1)
                        xt = xt + ds * v_eff

                    elif sampler == "midpoint":
                        t_mid.fill_(ti + 0.5 * ds)
                        v0 = model(xt, t0)
                        x_mid = xt + 0.5 * ds * v0
                        v_eff = model(x_mid, t_mid)
                        xt = xt + ds * v_eff

                    elif sampler == "ab2":
                        v = model(xt, t0)
                        if v_prev is None:
                            v_eff = v
                        else:
                            v_eff = 1.5 * v - 0.5 * v_prev
                        xt = xt + ds * v_eff
                        v_prev = v

                    elif sampler == "rk4":
                        t_mid.fill_(ti + 0.5 * ds)
                        k1 = model(xt, t0)
                        k2 = model(xt + 0.5 * ds * k1, t_mid)
                        k3 = model(xt + 0.5 * ds * k2, t_mid)
                        k4 = model(xt + ds * k3, t1)
                        v_eff = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
                        xt = xt + ds * v_eff

                    else:
                        raise ValueError(f"Unknown sampler: {sampler}")

                    # Optional early-exit: check only every few steps to reduce CPU sync
                    if early_stop and ((i + 1) % early_stop_check_every == 0):
                        # mean absolute update magnitude
                        update_mag = (ds * v_eff).abs().mean().item()
                        if update_mag < early_stop_tol:
                            small_update_hits += 1
                            if small_update_hits >= early_stop_patience:
                                break
                        else:
                            small_update_hits = 0

        model.train()
        return xt
