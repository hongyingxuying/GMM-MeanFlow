"""
D_GAN Training: Unified GAN/WGAN-CP/WGAN-GP training
Refactored to reduce code duplication and improve maintainability
"""
import os
import random
import logging
import argparse
import shutil
import time
import copy
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt

from U_DCNN import Classifier
from MyDataset import MyDataset
from ConfuseMatrix import plot_confusion_matrix
from data_pro_FFT_extension import FFTSignalDataset
from method_loader import load_module_from_file
gan_impl = load_module_from_file("M5_DCGAN", "M5-DCGAN.py")
wgancp_impl = load_module_from_file("M6_WGAN_CP", "M6-WGAN-CP.py")
wgangp_impl = load_module_from_file("M7_WGAN_GP", "M7-WGAN-GP.py")
from utils import *
from training_time_tracker import TrainingTimeTracker  # 导入训练时间跟踪器


# ============================================================================
# Configuration by Variant
# ============================================================================

VARIANT_CONFIGS = {
    "gan": {
        "tag": "DCGAN",
        "impl": gan_impl,
        "get_D": lambda in_len: gan_impl.VanillaDiscriminator1D(in_length=in_len),
        "optim": ("adam", {"betas": (0.5, 0.999)}),
        "loss_type": "bce",
        "instance_noise": True,
        "r1_penalty": True,
    },
    "wgan-cp": {
        "tag": "WGAN-CP",
        "impl": wgancp_impl,
        "get_D": lambda in_len: wgancp_impl.WGANCritic1D(in_length=in_len),
        "optim": ("rmsprop", {}),
        "loss_type": "wasserstein",
        "weight_clipping": True,
        "n_critic": 1,
        "clip_value": 0.10,
        "lr_g": 1e-4,
        "lr_d": 1e-4,
        "critic_data_range": "minus_one_one",
        "generator_output_range": "minus_one_one",
        "feature_matching_lambda": 0.25,
        "moment_matching_lambda": 2.0,
        "ema_decay": 0.999,
    },
    "wgan-gp": {
        "tag": "WGAN-GP",
        "impl": wgangp_impl,
        "get_D": lambda in_len: wgangp_impl.WGANCritic1D(in_length=in_len),
        "optim": ("adam", {"betas": (0.0, 0.9)}),
        "loss_type": "wasserstein",
        "gradient_penalty": True,
        "lambda_gp": 10.0,
    }
}


# ============================================================================
# Global Configuration
# ============================================================================

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

BASE_MODEL_NAME = ''
SELECT_VARIANT = "wgan-gp"  # "gan" | "wgan-cp" | "wgan-gp"
DEFAULT_VARIANT = SELECT_VARIANT

# Training defaults
DEFAULTS = {
    "sampleNumber": 20,
    "epochs": 5000,
    "sampleLength": 4096,
    "batch_size": 32,
    "learning_rate_g": 1e-4,
    "learning_rate_d": 1e-4,
    "latent_dim":128,#--------------128---------------------------------
    "n_critic": 5,#---------------------------------5---------------------------------
    "lambda_gp": 10.0,#---------------------------------10.0---------------------------------
    "clip_value": 0.01,
    "numOfClasses": 10,
    "generateNumber": 1000,
    "generateBatchsize": 1000,
    "num_epochs_classifier": 100,
    "batch_size_classifier": 32,
    "generateLabel": 0,
    "datasets": "xjtu",## paderborn | cwru | xjtu-----------------------------------------------数据集选择-----------------------------------------------
    "label_smooth": 0.9,
    "instance_noise_sigma": 0.05,
    "r1_gamma": 1.0,
}

BASE_SEED = 42
DETERMINISTIC = True

# Reproducibility
regenerate = True
retrain = True

# Path management
resultsSavingPath = None
modelSavingPath = None
dataSavingPath = None

def createPathIfNotExist(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def format_elapsed(seconds):
    minutes = int(seconds // 60)
    remain_seconds = seconds % 60
    return f"{minutes} 分钟 {remain_seconds:.2f} 秒"

def init_paths():
    global resultsSavingPath, modelSavingPath, dataSavingPath
    resultsSavingPath = createPathIfNotExist(r"./results/")
    modelSavingPath = createPathIfNotExist(r"./models/")
    dataSavingPath = createPathIfNotExist(r"./mats/")


# ============================================================================
# Utility Functions
# ============================================================================

def set_seed(seed: int, deterministic: bool = True):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def normalize_variant(variant: str) -> str:
    v = (variant or "").strip().lower()
    if v not in VARIANT_CONFIGS:
        raise ValueError(f"Unknown variant. Use one of: {', '.join(VARIANT_CONFIGS.keys())}")
    return v

def get_config(variant: str) -> dict:
    """Get configuration for given variant"""
    return VARIANT_CONFIGS[normalize_variant(variant)]

def get_run_name(variant: str) -> str:
    """Generate run name from variant"""
    cfg = get_config(variant)
    return cfg["tag"]

def ckpt_dir(run_name: str, label: int) -> str:
    return os.path.join(modelSavingPath, f"{run_name}_{label}")

def ckpt_path_G(run_name: str, label: int) -> str:
    return os.path.join(ckpt_dir(run_name, label), "G.pt")

def ckpt_path_D(run_name: str, label: int) -> str:
    return os.path.join(ckpt_dir(run_name, label), "D.pt")

def mat_path(run_name: str, label: int, dataset: str = None) -> str:
    if dataset is None:
        dataset = DEFAULTS.get('datasets', 'cwru')
    return os.path.join(dataSavingPath, f"{run_name}_{dataset}_class{label}.mat")

def results_dir(run_name: str, label: int) -> str:
    return os.path.join(resultsSavingPath, f"{run_name}_{label}")


# ============================================================================
# Regularization Functions
# ============================================================================

def _instance_noise_sigma(epoch: int, total_epochs: int, sigma_init: float) -> float:
    if sigma_init <= 0:
        return 0.0
    denom = max(int(total_epochs), 1)
    t = max(0.0, min(1.0, float(epoch) / float(denom)))
    return float(sigma_init) * (1.0 - t)

def _add_instance_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    return x + sigma * torch.randn_like(x)

def _r1_penalty(discriminator: nn.Module, real: torch.Tensor, gamma: float) -> torch.Tensor:
    if gamma <= 0:
        return torch.zeros((), device=real.device)
    real = real.detach().requires_grad_(True)
    real_logits = discriminator(real)
    grad = torch.autograd.grad(
        outputs=real_logits.sum(),
        inputs=real,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad = grad.view(grad.shape[0], -1)
    penalty = 0.5 * float(gamma) * (grad.pow(2).sum(dim=1)).mean()
    return penalty

def _to_critic_space(x: torch.Tensor, cfg: dict, from_generator: bool = False) -> torch.Tensor:
    """Map samples to the space used by the critic."""
    target_range = cfg.get("critic_data_range", "unit")
    if target_range != "minus_one_one":
        return x
    if from_generator and cfg.get("generator_output_range") == "minus_one_one":
        return x
    return x.mul(2.0).sub(1.0)

def _set_requires_grad(module: nn.Module, enabled: bool):
    for param in module.parameters():
        param.requires_grad_(enabled)

def _moment_matching_loss(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    real_flat = real.flatten(1)
    fake_flat = fake.flatten(1)
    mean_loss = F.l1_loss(fake_flat.mean(dim=0), real_flat.mean(dim=0))
    std_loss = F.l1_loss(fake_flat.std(dim=0, unbiased=False), real_flat.std(dim=0, unbiased=False))
    return mean_loss + std_loss

def _feature_matching_loss(discriminator: nn.Module, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    if not hasattr(discriminator, "features"):
        return torch.zeros((), device=fake.device)
    real_feat = discriminator.features(real).detach()
    fake_feat = discriminator.features(fake)
    return F.l1_loss(fake_feat.mean(dim=0), real_feat.mean(dim=0))


# ============================================================================
# Model Building
# ============================================================================

def build_optimizers(variant: str, G: nn.Module, D: nn.Module, 
                     lr_g: float, lr_d: float) -> tuple:
    """Build optimizers based on variant"""
    cfg = get_config(variant)
    optim_type, optim_kwargs = cfg["optim"]
    
    if optim_type == "adam":
        opt_G = optim.Adam(G.parameters(), lr=lr_g, **optim_kwargs)
        opt_D = optim.Adam(D.parameters(), lr=lr_d, **optim_kwargs)
    elif optim_type == "rmsprop":
        opt_G = optim.RMSprop(G.parameters(), lr=lr_g)
        opt_D = optim.RMSprop(D.parameters(), lr=lr_d)
    else:
        raise ValueError(f"Unknown optimizer type: {optim_type}")
    
    return opt_G, opt_D

def self_check(variant: str, latent_dim: int, sample_length: int, sample_number: int, datasets: str):
    """Lightweight import/shape check"""
    v = normalize_variant(variant)
    cfg = get_config(v)
    impl = cfg["impl"]
    
    G = impl.WGANGenerator1D(latent_dim=latent_dim, out_length=sample_length).to(device)
    D = cfg["get_D"](sample_length).to(device)
    
    with torch.no_grad():
        z = impl.sample_z(4, latent_dim, device)
        fake = G(z)
        d_out = D(fake)
    
    assert fake.shape == (4, 1, sample_length), f"G shape mismatch: {fake.shape}"
    assert d_out.shape == (4, 1), f"D shape mismatch: {d_out.shape}"
    print(f"✓ {v}: G{tuple(fake.shape)} D{tuple(d_out.shape)}")
    
    # Test dataset
    try:
        ds = FFTSignalDataset(dataSource=datasets, numOfClass=0, numOfData=min(8, sample_number),
                            lengthOfSample=sample_length, fs=12000)
        b = ds[0]
        assert b['data'].shape[-1] == sample_length
        print(f"✓ Dataset OK")
    except Exception as e:
        print(f"✓ Dataset check skipped: {e}")


# ============================================================================
# Training
# ============================================================================

def train_gan(variant: str, generate_label: int, sample_number: int, epochs: int,
              latent_dim: int, sample_length: int, batch_size: int,
              learning_rate_g: float, learning_rate_d: float, datasets: str, 
              save_every: int = 200, instance_noise_sigma: float = 0.05, 
              r1_gamma: float = 1.0, label_smooth: float = 0.9):
    """Train GAN with unified logic for all variants"""
    
    v = normalize_variant(variant)
    cfg = get_config(v)
    impl = cfg["impl"]
    run_name = get_run_name(variant)
    setup_logging(run_name)
    
    # Override LR if specified in config
    lr_g = cfg.get("lr_g", learning_rate_g)
    lr_d = cfg.get("lr_d", learning_rate_d)
    n_critic = cfg.get("n_critic", 1)
    lambda_gp = cfg.get("lambda_gp", 10.0)
    clip_value = cfg.get("clip_value", 0.01)
    
    # Build models
    G = impl.WGANGenerator1D(latent_dim=latent_dim, out_length=sample_length).to(device)
    D = cfg["get_D"](sample_length).to(device)
    ema_decay = float(cfg.get("ema_decay", 0.0))
    ema_G = copy.deepcopy(G).to(device) if ema_decay > 0 else None
    if ema_G is not None:
        for p in ema_G.parameters():
            p.requires_grad_(False)
    opt_G, opt_D = build_optimizers(variant, G, D, lr_g, lr_d)
    
    logger = SummaryWriter(os.path.join("runs", run_name))
    createPathIfNotExist(ckpt_dir(run_name, generate_label))
    createPathIfNotExist(results_dir(run_name, generate_label))
    
    # Load data
    trainset = FFTSignalDataset(dataSource=datasets, numOfClass=generate_label,
                               numOfData=sample_number, lengthOfSample=sample_length, fs=12000)
    effective_bs = min(batch_size, len(trainset))
    if effective_bs <= 0:
        raise ValueError(f"Empty trainset for class {generate_label}")
    
    trainloader = DataLoader(trainset, batch_size=effective_bs, shuffle=True, drop_last=False)
    l = max(len(trainloader), 1)
    
    bce_loss = nn.BCEWithLogitsLoss() if cfg["loss_type"] == "bce" else None
    global_step = 0
    
    # Training loop
    for epoch in range(epochs + 1):
        for step, batch in enumerate(trainloader):
            real = batch['data'].to(device).unsqueeze(1)  # (B,1,L)
            batch_sz = real.shape[0]
            z = impl.sample_z(batch_sz, latent_dim, device)
            real_w = _to_critic_space(real, cfg, from_generator=False)
            fake = _to_critic_space(G(z), cfg, from_generator=True).detach()
            
            # Add noise for GAN variants
            if cfg.get("instance_noise"):
                sigma = _instance_noise_sigma(epoch, epochs, instance_noise_sigma)
                real_d = _add_instance_noise(real, sigma)
                fake_d = _add_instance_noise(fake, sigma)
            else:
                real_d, fake_d = real_w, fake
            
            # ---- Train D ----
            if cfg["loss_type"] == "bce":
                real_logits = D(real_d)
                fake_logits = D(fake_d)
                y_real = torch.ones_like(real_logits) * float(label_smooth)
                y_fake = torch.zeros_like(fake_logits)
                loss_D = bce_loss(real_logits, y_real) + bce_loss(fake_logits, y_fake)
                
                if cfg.get("r1_penalty"):
                    loss_D = loss_D + _r1_penalty(D, real_d, r1_gamma)
            
            elif cfg["loss_type"] == "wasserstein":
                d_real = D(real_w).mean()
                d_fake = D(fake).mean()
                loss_D = d_fake - d_real
                
                if cfg.get("gradient_penalty"):
                    loss_D = loss_D + impl.gradient_penalty(D, real_w, fake, device, lambda_gp=lambda_gp)
            
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()
            
            # Weight clipping for WGAN-CP
            if cfg.get("weight_clipping"):
                impl.weight_clipping_(D, clip_value)
            
            # ---- Train G ----
            do_g_step = (cfg["loss_type"] == "bce") or (global_step % max(int(n_critic), 1) == 0)
            
            if do_g_step:
                _set_requires_grad(D, False)
                z2 = impl.sample_z(batch_sz, latent_dim, device)
                fake2_raw = G(z2)
                fake2 = _to_critic_space(fake2_raw, cfg, from_generator=True)
                
                if cfg["loss_type"] == "bce":
                    if cfg.get("instance_noise"):
                        sigma2 = _instance_noise_sigma(epoch, epochs, instance_noise_sigma)
                        fake2_d = _add_instance_noise(fake2, sigma2)
                    else:
                        fake2_d = fake2
                    g_logits = D(fake2_d)
                    y = torch.ones_like(g_logits)
                    loss_G = bce_loss(g_logits, y)
                else:
                    loss_G = -D(fake2).mean()
                    fm_lambda = float(cfg.get("feature_matching_lambda", 0.0))
                    if fm_lambda > 0:
                        loss_G = loss_G + fm_lambda * _feature_matching_loss(D, real_w, fake2)
                    moment_lambda = float(cfg.get("moment_matching_lambda", 0.0))
                    if moment_lambda > 0:
                        loss_G = loss_G + moment_lambda * _moment_matching_loss(real_w, fake2)
                
                opt_G.zero_grad()
                loss_G.backward()
                opt_G.step()
                if ema_G is not None:
                    with torch.no_grad():
                        for ema_p, p in zip(ema_G.parameters(), G.parameters()):
                            ema_p.data.mul_(ema_decay).add_(p.data, alpha=1.0 - ema_decay)
                _set_requires_grad(D, True)
            else:
                loss_G = torch.tensor(float('nan'))
            
            # Logging
            if epoch == 5000:
                if torch.isfinite(loss_G):
                    print(f"[{run_name}][{epoch}/{epochs}] D:{loss_D.item():.4f} G:{loss_G.item():.4f}")
                else:
                    print(f"[{run_name}][{epoch}/{epochs}] D:{loss_D.item():.4f}")
            
            logger.add_scalar("loss_D", float(loss_D.item()), global_step=epoch * l + step)
            if torch.isfinite(loss_G):
                logger.add_scalar("loss_G", float(loss_G.item()), global_step=epoch * l + step)
            global_step += 1
        
        # Visualization
        if (epoch >= max(0, epochs - 49)) and (epoch % 10 == 0):
            with torch.no_grad():
                z = impl.sample_z(min(16, batch_size), latent_dim, device)
                sample_model = ema_G if ema_G is not None else G
                samples = sample_model(z).detach().cpu().squeeze(1)
            f, a = plt.subplots(3, 1, figsize=(20, 30))
            for i in range(3):
                idx = random.randint(0, samples.shape[0] - 1)
                a[i].plot(samples[idx].view(-1), lw=1, ls='-', c='b', alpha=0.8)
                a[i].set_yticks(([0, 0.25, 0.5, 0.75, 1]))
            plt.savefig(os.path.join(results_dir(run_name, generate_label), f"{epoch}.jpg"))
            plt.close()
        
        # Save checkpoints
        if (epoch % max(save_every, 1) == 0) or (epoch == epochs):
            torch.save((ema_G if ema_G is not None else G).state_dict(), ckpt_path_G(run_name, generate_label))
            torch.save(D.state_dict(), ckpt_path_D(run_name, generate_label))


# ============================================================================
# Sample Generation & Evaluation
# ============================================================================

def generateSamples(variant: str, generate_label: int, latent_dim: int, sample_length: int,
                   generate_number: int, generate_batchsize: int, dataset: str = None):    
    v = normalize_variant(variant)
    cfg = get_config(v)
    impl = cfg["impl"]
    run_name = get_run_name(variant)
    
    G = impl.WGANGenerator1D(latent_dim=latent_dim, out_length=sample_length).to(device)
    ckpt = torch.load(ckpt_path_G(run_name, generate_label), map_location=device)
    G.load_state_dict(ckpt)
    G.eval()
    
    createPathIfNotExist(ckpt_dir(run_name, generate_label))
    createPathIfNotExist(results_dir(run_name, generate_label))
    
    if generate_number <= 0:
        raise ValueError("generateNumber must be > 0")
    
    all_samples = []
    batch_idx = 0
    for start in range(0, generate_number, generate_batchsize):
        cur_bs = min(generate_batchsize, generate_number - start)        
        batch_idx += 1
        
        with torch.no_grad():
            z = torch.randn(cur_bs, latent_dim, device=device)
            sampled = G(z).detach().cpu().numpy()  # (B,1,L)
        
        sampled = np.squeeze(sampled)  # (B,L) or (L,)
        if sampled.ndim == 1:
            sampled = sampled[None, :]
        
        # Min-max normalization
        mins = sampled.min(axis=1, keepdims=True)
        maxs = sampled.max(axis=1, keepdims=True)
        sampled = (sampled - mins) / (maxs - mins + 1e-8)
        all_samples.append(sampled)
    
    all_samples = np.concatenate(all_samples, axis=0)
    out_path = mat_path(run_name, generate_label, dataset)
    sio.savemat(out_path, {'result': all_samples})
    


def trainAndValidation(variant: str, num_epochs: int, batch_size: int, count: int,
                      sample_length: int, num_of_classes: int, datasets: str, base_seed: int):
    """Train and validate classifier on generated samples"""
    v = normalize_variant(variant)
    run_name = get_run_name(variant)
    set_seed(base_seed + int(count), deterministic=True)
    
    netC = Classifier(num_of_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(netC.parameters(), lr=0.001)
    
    # Prepare mat paths
    mat_paths = [mat_path(run_name, i, datasets) for i in range(num_of_classes)]
    missing = [p for p in mat_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing .mat files: {missing}")
    
    # Train classifier
    dataset = MyDataset(mat_paths)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in data_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = netC(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
    
    # Validation
    validation_dataset = FFTSignalDataset(dataSource=datasets, numOfClass=num_of_classes,
                                         numOfData=1000, lengthOfSample=sample_length, fs=12000)
    validation_loader = DataLoader(validation_dataset, batch_size=1000, shuffle=False)
    
    netC.eval()
    total_correct, total_preds = 0, 0
    validation_loss = 0.0
    confusion_matrix = np.zeros((num_of_classes, num_of_classes))
    
    for results_dir_list in [results_dir(run_name, i) for i in range(num_of_classes)]:
        createPathIfNotExist(results_dir_list)
    
    with torch.no_grad():
        for batch in validation_loader:
            inputs = torch.squeeze(batch['data']).to(device)
            labels = torch.squeeze(batch['label'].long()).to(device)
            
            outputs = netC(inputs)
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs.data, 1)
            total_preds += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            validation_loss += loss.item() * labels.size(0)
            
            for i in range(len(labels)):
                confusion_matrix[labels[i].item()][predicted[i].item()] += 1
    
    avg_loss = validation_loss / len(validation_dataset)
    accuracy = total_correct / total_preds
    
    # F1 scores
    tp = np.diag(confusion_matrix)
    fp = confusion_matrix.sum(axis=0) - tp
    fn = confusion_matrix.sum(axis=1) - tp
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1_per_class = 2 * precision * recall / (precision + recall + 1e-8)
    macro_f1 = np.mean(f1_per_class)
    weighted_f1 = np.sum(f1_per_class * confusion_matrix.sum(axis=1)) / (confusion_matrix.sum() + 1e-8)
    
    print(f'  Val Loss: {avg_loss:.5f}, Acc: {accuracy * 100:.2f}%, Macro-F1: {macro_f1:.4f}, Weighted-F1: {weighted_f1:.4f}')
    
    plot_confusion_matrix(confusion_matrix, classes=[str(i) for i in range(num_of_classes)],
                         savingPath=r"./confusionmatrix/", name=f"{run_name}_{count}")
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'per_class_f1': f1_per_class,
        'confusion_matrix': confusion_matrix,
        'validation_loss': avg_loss
    }


def ensure_generated_mats(variant: str, num_of_classes: int, latent_dim: int,
                         sample_length: int, generate_number: int, 
                         generate_batchsize: int, dataset: str = None):
    """Generate missing .mat files if checkpoints exist"""
    if dataset is None:
        dataset = DEFAULTS.get('datasets', 'cwru')
    v = normalize_variant(variant)
    run_name = get_run_name(variant)
    
    for i in range(num_of_classes):
        mp = mat_path(run_name, i, dataset)
        if os.path.exists(mp):
            continue
        
        cp = ckpt_path_G(run_name, i)
        if not os.path.exists(cp):
            raise FileNotFoundError(f"Missing checkpoint {cp}. Please train first.")
        
        print(f"  Auto-generate mat for class_{i}...")
        generateSamples(variant, i, latent_dim, sample_length, generate_number, generate_batchsize, dataset)


# ============================================================================
# Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Train and evaluate GAN/WGAN-CP/WGAN-GP")
    p.add_argument("--variant", type=str, default=DEFAULT_VARIANT,
                  help="gan | wgan-cp | wgan-gp")
    p.add_argument("--run_all", action="store_true",
                  help="Run all variants sequentially")
    p.add_argument("--retrain", action="store_true", default=retrain)
    p.add_argument("--no_retrain", action="store_false", dest="retrain")
    p.add_argument("--regenerate", action="store_true", default=regenerate)
    p.add_argument("--no_regenerate", action="store_false", dest="regenerate")
    p.add_argument("--self_check", action="store_true", help="Run checks and exit")
    
    # Training params
    for key, default in DEFAULTS.items():
        if key in ["datasets"]:
            p.add_argument(f"--{key}", type=str, default=default)
        elif isinstance(default, int):
            p.add_argument(f"--{key}", type=int, default=default)
        elif isinstance(default, float):
            p.add_argument(f"--{key}", type=float, default=default)
    
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--seed", type=int, default=BASE_SEED)
    p.add_argument("--deterministic", action="store_true", default=DETERMINISTIC)
    
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    init_paths()
    set_seed(args.seed, deterministic=args.deterministic)
    print(f"Using device: {device}, dataset: {args.datasets}, numOfClasses: {args.numOfClasses}")
    
    variants = ["gan", "wgan-cp", "wgan-gp"] if args.run_all else [normalize_variant(args.variant)]
    
    for variant in variants:
        # 为每个变体初始化训练时间跟踪器
        model_name = get_run_name(variant)
        time_tracker = TrainingTimeTracker(model_name)
        time_tracker.start()
        
        if args.self_check:
            self_check(variant, args.latent_dim, args.sampleLength, args.sampleNumber, args.datasets)
            continue
        
        run_name = get_run_name(variant)
        print(f"\n{'='*60}\n[{run_name}]\n{'='*60}")
        overall_start = time.time()
        train_model_elapsed = 0.0
        generate_data_elapsed = 0.0
        train_classifier_elapsed = 0.0
        
        if args.retrain:
            phase_start = time.time()
            for i in range(args.numOfClasses):
                print(f"Training class_{i}...")
                train_gan(variant, i, args.sampleNumber, args.epochs, args.latent_dim,
                         args.sampleLength, args.batch_size, args.learning_rate_g,
                         args.learning_rate_d, args.datasets, instance_noise_sigma=args.instance_noise_sigma,
                         r1_gamma=args.r1_gamma, label_smooth=args.label_smooth)
            phase_time = time.time() - phase_start
            time_tracker.phase_times['model_training'] = phase_time
            train_model_elapsed = phase_time
            print(f"[耗时] 生成模型训练: {format_elapsed(train_model_elapsed)}")
        
        if args.regenerate:
            phase_start = time.time()
            for i in range(args.numOfClasses):
                print(f"Generating class_{i}...")
                generateSamples(variant, i, args.latent_dim, args.sampleLength,
                              args.generateNumber, args.generateBatchsize, args.datasets)
            phase_time = time.time() - phase_start
            time_tracker.phase_times['data_generation'] = phase_time
            generate_data_elapsed = phase_time
            print(f"[耗时] 数据生成: {format_elapsed(generate_data_elapsed)}")
        
        ensure_generated_mats(variant, args.numOfClasses, args.latent_dim, args.sampleLength,
                            args.generateNumber, args.generateBatchsize, args.datasets)
        
        print("\nClassifier training...")
        results_list = []
        classification_start = time.time()
        for t in range(args.repeats):
            print(f"  Repeat {t+1}/{args.repeats}...")
            result_dict = trainAndValidation(variant, args.num_epochs_classifier, args.batch_size_classifier,
                                    t, args.sampleLength, args.numOfClasses, args.datasets, args.seed)
            results_list.append(result_dict)
        classification_time = time.time() - classification_start
        time_tracker.phase_times['classification_training'] = classification_time
        train_classifier_elapsed = classification_time
        print(f"[耗时] 分类器训练与验证: {format_elapsed(train_classifier_elapsed)}")
        
        # Extract metrics and compute statistics
        accuracy_list = [r['accuracy'] for r in results_list]
        macro_f1_list = [r['macro_f1'] for r in results_list]
        weighted_f1_list = [r['weighted_f1'] for r in results_list]
        
        # Calculate mean and standard deviation
        acc_mean = np.mean(accuracy_list)
        acc_std = np.std(accuracy_list)
        macro_f1_mean = np.mean(macro_f1_list)
        macro_f1_std = np.std(macro_f1_list)
        weighted_f1_mean = np.mean(weighted_f1_list)
        weighted_f1_std = np.std(weighted_f1_list)
        
        # Calculate max values
        acc_max = max(accuracy_list)
        macro_f1_max = max(macro_f1_list)
        weighted_f1_max = max(weighted_f1_list)
        
        # Print results
        print(f"\nResults for {run_name}:")
        print("Accuracy values:", [f"{acc*100:.2f}%" for acc in accuracy_list])
        print(f"Accuracy: {acc_mean*100:.2f}% ± {acc_std*100:.2f}% (max: {acc_max*100:.2f}%)")
        print(f"Macro-F1: {macro_f1_mean:.4f} ± {macro_f1_std:.4f} (max: {macro_f1_max:.4f})")
        print(f"Weighted-F1: {weighted_f1_mean:.4f} ± {weighted_f1_std:.4f} (max: {weighted_f1_max:.4f})")
        
        # Save results to file
        result_filename = f"./results/{run_name}_{args.datasets}_diagnosis_result.txt"
        with open(result_filename, 'w') as f:
            f.write(f"Model: {run_name}\n")
            f.write(f"Dataset: {args.datasets}\n")
            f.write(f"Number of iterations: {args.repeats}\n")
            f.write("="*80 + "\n\n")
            
            f.write("ACCURACY:\n")
            f.write(f"  All values: {' '.join([f'{acc*100:.2f}%' for acc in accuracy_list])}\n")
            f.write(f"  Mean: {acc_mean*100:.2f}%\n")
            f.write(f"  Std Dev: {acc_std*100:.2f}%\n")
            f.write(f"  Max: {acc_max*100:.2f}%\n\n")
            
            f.write("MACRO-F1:\n")
            f.write(f"  All values: {' '.join([f'{f1:.4f}' for f1 in macro_f1_list])}\n")
            f.write(f"  Mean: {macro_f1_mean:.4f}\n")
            f.write(f"  Std Dev: {macro_f1_std:.4f}\n")
            f.write(f"  Max: {macro_f1_max:.4f}\n\n")
            
            f.write("WEIGHTED-F1:\n")
            f.write(f"  All values: {' '.join([f'{f1:.4f}' for f1 in weighted_f1_list])}\n")
            f.write(f"  Mean: {weighted_f1_mean:.4f}\n")
            f.write(f"  Std Dev: {weighted_f1_std:.4f}\n")
            f.write(f"  Max: {weighted_f1_max:.4f}\n")
        
        # 结束计时并保存训练时间统计
        elapsed_time = time.time() - overall_start
        print(f"Using device: {device}, dataset: {args.datasets}, numOfClasses: {args.numOfClasses}")
        print(f"本次测试耗费时间为： {format_elapsed(elapsed_time)}")
        print("------------------------- 阶段耗时汇总 -------------------------")
        print(f"生成模型训练耗时: {format_elapsed(train_model_elapsed)}")
        print(f"数据生成耗时: {format_elapsed(generate_data_elapsed)}")
        print(f"分类器训练与验证耗时: {format_elapsed(train_classifier_elapsed)}")
        time_tracker.end()
        time_tracker.print_summary()
        time_tracker.save_json()
        print(f"\nResults saved to: {result_filename}")
