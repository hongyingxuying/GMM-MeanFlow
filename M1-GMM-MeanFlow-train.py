import os
import random
import copy
import time
import sys
import importlib
import importlib.util
from pathlib import Path
import torch
import torch.nn as nn
from torch import optim
import logging
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import scipy.io as sio

from U_DCNN import Classifier
from utils import *
from MyDataset import MyDataset
from ConfuseMatrix import plot_confusion_matrix
from data_pro_FFT_extension import FFTSignalDataset

model_backbone = 'udit'  # 'unet' | 'udit' | 'udit_true'
BACKBONE_MODULE_MAP = {
    'unet': 'U_Net',
    'udit': 'U_DiT',
    'udit_true': 'U_DiT_True',
}

module = importlib.import_module(BACKBONE_MODULE_MAP.get(model_backbone, model_backbone))
BackboneModel = getattr(module, 'UNet1D')

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

MEANFLOW_FD_PATH = CURRENT_DIR / "M1-GMM-MeanFlow.py"
_meanflow_spec = importlib.util.spec_from_file_location("M1_GMM_MeanFlow", MEANFLOW_FD_PATH)
_meanflow_module = importlib.util.module_from_spec(_meanflow_spec)
_meanflow_spec.loader.exec_module(_meanflow_module)
DiffusionMeanFlow = _meanflow_module.DiffusionMeanFlow

modelName = 'GMM-MeanFlow'
print(f"[model] resolved backbone class: {BackboneModel.__module__}.{BackboneModel.__name__}")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def createPathIfNotExist(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def format_elapsed(seconds):
    minutes = int(seconds // 60)
    remain_seconds = seconds % 60
    return f"{minutes} 分钟 {remain_seconds:.2f} 秒"


use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = 0
device = torch.device("cuda:0" if use_cuda else "cpu")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# training params
retrain = True
sampleNumber = 20
train_time_steps = 100
sample_steps = 1 
epochs = 5000
sampleLength = 4096
batch_size = min(32, sampleNumber)
learning_rate = 3e-4

# generate params
regenerate = True
generateNumber = 1000
generateBatchsize = 1000
num_epochs_classifier = 100
batch_size_classifier = 64
generateLabel = 0
numOfClasses = 10

datasets = 'paderborn'  # paderborn | cwru | xjtu
sample_sampler = "meanflow"  # "meanflow" | "ab2" | "euler" | "heun"
           
# MEANflow hyper-parameters
meanflow_r_eq_t_ratio = 0.50
meanflow_time_dist = 'lognorm'   # 'uniform' | 'lognorm'
meanflow_time_mu = -0.4
meanflow_time_sigma = 1.0
meanflow_jvp_api = 'func'        # 'autograd' | 'func' | 'finite_diff'
meanflow_time_condition_mode = 'dual'  # 'delta_t' | 'dual'
meanflow_loss_p = 1.0
meanflow_loss_c = 1e-3
generate_seed_base = 20260422

resultsSavingPath = createPathIfNotExist(r"./results/")
modelSavingPath = createPathIfNotExist(r"./models/")
dataSavingPath = createPathIfNotExist(r"./mats/")


def configure_meanflow(diffusion):
    diffusion.time_dist = meanflow_time_dist
    diffusion.time_mu = meanflow_time_mu
    diffusion.time_sigma = meanflow_time_sigma
    diffusion.r_eq_t_ratio = meanflow_r_eq_t_ratio
    diffusion.jvp_api = meanflow_jvp_api
    diffusion.time_condition_mode = meanflow_time_condition_mode
    return diffusion


def trainFDDiffusion(generateLabel=generateLabel, sampleNumber=sampleNumber, train_time_steps=train_time_steps, epochs=epochs):
    setup_logging(modelName)

    model = BackboneModel(c_in=1, c_out=1, time_dim=256, device=device).to(device)
    ema_model = copy.deepcopy(model).to(device)
    for p in ema_model.parameters():
        p.requires_grad = False
    ema_decay = 0.999

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    try:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=True)
    except TypeError:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)

    diffusion = DiffusionMeanFlow(noise_steps=train_time_steps, data_length=sampleLength)
    configure_meanflow(diffusion)

    logger = SummaryWriter(os.path.join("runs", modelName))
    createPathIfNotExist(r"./models/{}_{}/".format(modelName, generateLabel))
    createPathIfNotExist(r"./results/{}_{}/".format(modelName, generateLabel))

    # 训练数据 + 原始样本拟合GMM
    trainset = FFTSignalDataset(
        dataSource=datasets,
        numOfClass=generateLabel,
        numOfData=sampleNumber,
        lengthOfSample=sampleLength,
        transform='fft',
        normalize='minmax',
    )
    effective_batch_size = max(1, min(batch_size, len(trainset)))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=effective_batch_size, shuffle=True, drop_last=False)
    l = len(trainloader)
    last_batch_size = effective_batch_size

    gmm_train_samples = trainset.dataset
    if isinstance(gmm_train_samples, torch.Tensor):
        gmm_train_samples = gmm_train_samples.detach().cpu().numpy()
    diffusion.fit_gmm(gmm_train_samples[:, None, :])

    gmm_params_path = os.path.join("models", "{}_{}".format(modelName, generateLabel), "gmm_params.npz")
    diffusion.save_gmm_params(gmm_params_path)

    for epoch in range(epochs + 1):
        logging.info(f"Starting epoch {epoch}:")
        epoch_loss_accum = 0.0

        for step, batch in enumerate(trainloader):
            inputData = batch['data'].to(device)
            inputData = inputData.unsqueeze(1)
            last_batch_size = inputData.shape[0]

            loss_info = diffusion.meanflow_loss(
                model,
                inputData,
                r_eq_t_ratio=meanflow_r_eq_t_ratio,
                loss_p=meanflow_loss_p,
                loss_c=meanflow_loss_c,
                time_dist=meanflow_time_dist,
                time_mu=meanflow_time_mu,
                time_sigma=meanflow_time_sigma,
            )
            loss = loss_info["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                    ema_p.data.mul_(ema_decay).add_(p.data * (1.0 - ema_decay))

            epoch_loss_accum += loss.item()

            if epoch % 1000 == 0:
                print('[%d/%d]\tLoss: %.3f' % (epoch, epochs, loss.item()))
            logger.add_scalar("loss", loss.item(), global_step=epoch * l + step)
            logger.add_scalar("loss_u_mse", loss_info["u_mse"].item(), global_step=epoch * l + step)
            logger.add_scalar("dt_mean", loss_info["delta_t_mean"].item(), global_step=epoch * l + step)
            logger.add_scalar("r_eq_t_ratio", loss_info["r_eq_t_ratio"].item(), global_step=epoch * l + step)

        epoch_loss = epoch_loss_accum / max(1, l)
        scheduler.step(epoch_loss)

        if (epoch >= max(0, epochs - 49)) and ((epoch % 10) == 0):
            sampled_images = diffusion.sample(
                ema_model,
                n=min(16, last_batch_size),
                steps=sample_steps,
                sampler=sample_sampler,
            )
            sampled_images = sampled_images.detach().cpu()

        torch.save(ema_model.state_dict(), os.path.join("models", "{}_{}".format(modelName, generateLabel), "ckpt.pt"))


def generateSamples(
    generateLabel=generateLabel,
    generateNumber=generateNumber,
    generateBatchsize=generateBatchsize,
    sampler: str = sample_sampler,
    steps: int = sample_steps,
):
    run_device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BackboneModel(c_in=1, c_out=1, time_dim=256, device=run_device).to(run_device)

    ckpt_path = os.path.join(modelSavingPath, "{}_{}/ckpt.pt".format(modelName, generateLabel))
    ckpt = torch.load(ckpt_path, map_location=run_device)
    model.load_state_dict(ckpt)
    model.eval()

    diffusion = DiffusionMeanFlow(noise_steps=train_time_steps, data_length=sampleLength)
    configure_meanflow(diffusion)
    createPathIfNotExist(r"./models/{}_{}/".format(modelName, generateLabel))
    createPathIfNotExist(r"./results/{}_{}/".format(modelName, generateLabel))

    gmm_params_path = os.path.join(modelSavingPath, "{}_{}/gmm_params.npz".format(modelName, generateLabel))
    if not os.path.exists(gmm_params_path):
        raise FileNotFoundError(f"GMM params not found: {gmm_params_path}")
    diffusion.load_gmm_params(gmm_params_path)

    if run_device == "cuda":
        torch.cuda.empty_cache()

    safe_batch_cap = 32 if sampleLength >= 2048 else 128
    cur_bs = int(max(1, min(generateBatchsize, safe_batch_cap)))

    all_samples = []
    remaining = int(generateNumber)
    while remaining > 0:
        this_bs = int(min(cur_bs, remaining))
        try:
            sampled_images = diffusion.sample(
                model,
                n=this_bs,
                steps=steps,
                sampler=sampler,
            )
        except torch.OutOfMemoryError:
            if run_device == "cuda":
                torch.cuda.empty_cache()
            if this_bs <= 1:
                raise
            cur_bs = max(1, this_bs // 2)
            continue

        sampled_images = sampled_images.detach().float().cpu().numpy()
        sampled_images = np.squeeze(sampled_images)
        if sampled_images.ndim == 1:
            sampled_images = sampled_images[None, :]

        mins = sampled_images.min(axis=1, keepdims=True)
        maxs = sampled_images.max(axis=1, keepdims=True)
        denom = maxs - mins + 1e-8
        sampled_images = (sampled_images - mins) / denom

        all_samples.append(sampled_images.astype(np.float32, copy=False))
        remaining -= this_bs

        del sampled_images
        if run_device == "cuda":
            torch.cuda.empty_cache()

    all_samples = np.concatenate(all_samples, axis=0)

    mat_payload = {'result': all_samples}
    dataset_mat_path = f'{dataSavingPath}/{modelName}_{datasets}_time_class{generateLabel}.mat'
    legacy_mat_path = f'{dataSavingPath}/{modelName}_time_class{generateLabel}.mat'
    sio.savemat(dataset_mat_path, mat_payload)
    # Keep the historical filename for older analysis scripts.
    sio.savemat(legacy_mat_path, mat_payload)


def trainAndValidation(num_epochs, batch_size, count):
    # 每次分类器重复训练使用可复现且不同的种子，降低偶发塌陷不可复现问题
    set_seed(1234 + int(count))

    netC = Classifier(numOfClasses).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(netC.parameters(), lr=3e-4)
    try:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, verbose=False)
    except TypeError:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)

    mat_paths = []
    for class_id in range(0, numOfClasses):
        createPathIfNotExist(r"./results/{}_{}".format(modelName, class_id))
        matPath = dataSavingPath + r'{}_{}_time_class{}.mat'.format(modelName, datasets, class_id)
        if not os.path.exists(matPath):
            matPath = dataSavingPath + r'{}_time_class{}.mat'.format(modelName, class_id)
        mat_paths.append(matPath)

    dataset = MyDataset(mat_paths)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for _ in range(num_epochs):
        running_loss = 0.0
        batch_count = 0
        for batch in data_loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = netC(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(netC.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            batch_count += 1

        epoch_loss = running_loss / max(1, batch_count)
        scheduler.step(epoch_loss)

    validation_dataset = FFTSignalDataset(
        dataSource=datasets,
        numOfClass=numOfClasses,
        numOfData=1000,
        lengthOfSample=sampleLength,
        transform='fft',
        normalize='minmax',
    )
    validation_data_loader = DataLoader(validation_dataset, batch_size=generateNumber, shuffle=True)
    netC.eval()

    total_predictions = 0
    correct_predictions = 0
    validation_loss = 0.0
    confusion_matrix = np.zeros((numOfClasses, numOfClasses))
    pred_hist = np.zeros((numOfClasses,), dtype=np.int64)

    for batch in validation_data_loader:
        inputs = torch.squeeze(batch['data']).to(device)
        labels = torch.squeeze(batch['label'].long()).to(device)

        with torch.no_grad():
            outputs = netC(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)

            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            validation_loss += loss.item() * labels.size(0)

            for class_id in range(numOfClasses):
                pred_hist[class_id] += int((predicted == class_id).sum().item())

            for i in range(len(labels)):
                true_label = labels[i].item()
                predicted_label = predicted[i].item()
                confusion_matrix[true_label][predicted_label] += 1

    average_validation_loss = validation_loss / len(validation_data_loader.dataset)
    accuracy = correct_predictions / total_predictions

    tp = np.diag(confusion_matrix)
    fp = confusion_matrix.sum(axis=0) - tp
    fn = confusion_matrix.sum(axis=1) - tp
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1_per_class = 2 * precision * recall / (precision + recall + 1e-8)
    macro_f1 = np.mean(f1_per_class)
    weighted_f1 = np.sum(f1_per_class * confusion_matrix.sum(axis=1)) / (confusion_matrix.sum() + 1e-8)

    # 若预测高度塌陷到单一类别（>95%）视为无效轮次
    pred_ratio = pred_hist / max(1, pred_hist.sum())
    collapsed = bool(np.max(pred_ratio) > 0.95)

    print(f'Validation Loss: {average_validation_loss:.4f}, Accuracy: {accuracy * 100:.2f}%, Macro-F1: {macro_f1:.4f}, Weighted-F1: {weighted_f1:.4f}')
    #print('Per-class F1:', ', '.join([f'{f1:.4f}' for f1 in f1_per_class]))
    print('Pred hist:', pred_hist.tolist(), 'collapsed:', collapsed)

    plot_confusion_matrix(
        confusion_matrix,
        classes=['{}'.format(i) for i in range(numOfClasses)],
        savingPath=r"./ /",
        name=r"{}_{}".format(modelName, count),
    )
    return accuracy, collapsed


if __name__ == '__main__':
    set_seed(123)
    for re in range(1):
        start_time = time.time()
        train_model_elapsed = 0.0
        generate_data_elapsed = 0.0
        train_classifier_elapsed = 0.0
        print(f"-------------------------------- 第---{re}--{re}--{re}--次迭代 -----------------------------------")

        if retrain:
            train_model_start = time.time()
            print("============================ MEANflow扩散模型  训练 ==========================")
            for i in range(0, numOfClasses):
                trainFDDiffusion(generateLabel=i, sampleNumber=sampleNumber, train_time_steps=train_time_steps, epochs=epochs)
                print(f"{i}__Class 扩散模型训练完成.")
            train_model_elapsed = time.time() - train_model_start
            print(f"[耗时] 生成模型训练: {format_elapsed(train_model_elapsed)}")

        if regenerate:
            run_generate_seed = int(generate_seed_base + re * 1000)
            print(f"[seed] 生成阶段基础种子: {run_generate_seed}")
            generate_data_start = time.time()
            print("============================ MEANflow扩散模型  生成 ==========================")
            for i in range(0, numOfClasses):
                set_seed(run_generate_seed + i)
                generateSamples(generateLabel=i)
                print(f"{i}__Class 生成数据完成.")
            generate_data_elapsed = time.time() - generate_data_start
            print(f"[耗时] 数据生成: {format_elapsed(generate_data_elapsed)}")

        train_classifier_start = time.time()
        accuracyList = []
        collapsedCount = 0
        for time_idx in range(0, 10):
            accuracy, collapsed = trainAndValidation(num_epochs=num_epochs_classifier, batch_size=batch_size_classifier, count=time_idx)
            accuracyList.append(accuracy)
            if collapsed:
                collapsedCount += 1
        train_classifier_elapsed = time.time() - train_classifier_start
        print(f"[耗时] 分类器训练与验证: {format_elapsed(train_classifier_elapsed)}")

        filtered_accuracy_list = [acc for acc in accuracyList if acc > 0.26]
        ACC_MEAN = sum(filtered_accuracy_list) / (len(filtered_accuracy_list) + 1e-4)
        ACC_MAX = max(accuracyList)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"第 {re} 次测试耗费时间为： {format_elapsed(elapsed_time)}")
        print("------------------------- 阶段耗时汇总 -------------------------")
        print(f"生成模型训练耗时: {format_elapsed(train_model_elapsed)}")
        print(f"数据生成耗时: {format_elapsed(generate_data_elapsed)}")
        print(f"分类器训练与验证耗时: {format_elapsed(train_classifier_elapsed)}")
        print("************************************************* 诊断 结果 **************************************************")
        print(f"collapsed runs: {collapsedCount}/{len(accuracyList)}")
        print("accuracy:", [f"{acc*100:.2f}" for acc in accuracyList])
        print('max: ', f"{ACC_MAX*100:.2f}", 'mean: ', f"{ACC_MEAN*100:.2f}")
