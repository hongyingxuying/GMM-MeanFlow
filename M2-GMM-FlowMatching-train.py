import os
from pickle import FALSE
import random
import sys
import io
import copy
import torch
import torch.nn as nn
from torch import optim
import logging
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import scipy.io as sio
import time  # 在文件顶部添加

from U_DCNN import Classifier
from MyDataset import MyDataset
from ConfuseMatrix import plot_confusion_matrix
from data_pro_FFT_extension import FFTSignalDataset
from method_loader import load_attr_from_file
Diffusion = load_attr_from_file("M2_GMM_FlowMatching", "M2-GMM-FlowMatching.py", "Diffusion")
from utils import *
from U_Net import UNet1D
modelName = 'GMM-FlowMatching'



# 固定随机种子以确保结果可重复
def set_seed(seed):
    import os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def createPathIfNotExist(path):#创建存储路径
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def format_elapsed(seconds):
    minutes = int(seconds // 60)
    remain_seconds = seconds % 60
    return f"{minutes} 分钟 {remain_seconds:.2f} 秒"

def get_num_classes(dataset_name):
    if dataset_name == 'paderborn':
        return 10
    if dataset_name == 'cwru':
        return 10
    if dataset_name == 'xjtu':
        return 10
    raise ValueError("datasets must be 'paderborn' or 'cwru' or 'xjtu'")

def huber_loss(predictions, targets, delta=1.0):
    errors = torch.abs(predictions - targets)
    condition = errors > delta
    loss = torch.where(condition, 0.5 * errors**2, delta * (errors - 0.5 * delta))
    return torch.mean(loss)

use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = 0
device = torch.device("cuda:0" if use_cuda else "cpu")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# training params
retrain = True
sampleNumber = 20        
noise_steps = 100         #100
epochs = 5000             #5000    
sampleLength = 4096  
batch_size = min(32, sampleNumber)
learning_rate = 3e-4

# generate params
regenerate = True
generateNumber = 1000 
generateBatchsize = 1000 
num_epochs_classifier = 100    
batch_size_classifier = 32   
generateLabel = 0

datasets= 'paderborn'  # paderborn | cwru | xjtu

numOfClasses = get_num_classes(datasets)
print(f"Using device: {device}, dataset: {datasets}, numOfClasses: {numOfClasses}")

# sampling/integration params (speed vs quality)
sample_sampler = "ab2"   # "euler" | "heun" | "midpoint" | "ab2" | "rk4" |"hybrid"
sample_steps = 100        # <= noise_steps; try 50 for faster sampling

# GMM params (ablation: only GMM innovation)
gmm_max_components = 8
gmm_reg_covar = 1e-5

resultsSavingPath = createPathIfNotExist(r"./results/")#JPG图片
modelSavingPath = createPathIfNotExist(r"./models/")#模型保存位置
dataSavingPath = createPathIfNotExist(r"./mats/")#生成的数据：1000*1024的mat数据

def trainFDDiffusion(generateLabel=generateLabel, sampleNumber=sampleNumber, noise_steps=noise_steps, epochs=epochs):
    
    setup_logging(modelName)
    # model
    model = UNet1D(c_in=1, c_out=1, time_dim=256, device=device).to(device)
    # EMA model (用于稳定采样与保存)
    ema_model = copy.deepcopy(model).to(device)
    for p in ema_model.parameters():
        p.requires_grad = False
    ema_decay = 0.999

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    # LR 调度：根据训练 loss 降低 LR（兼容不同 PyTorch 版本）
    try:
        # 新版本支持 verbose 参数
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=True)
    except TypeError:
        # 旧版本没有 verbose 参数
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)
    mse = nn.MSELoss()
    sl1 = nn.SmoothL1Loss()
    diffusion = Diffusion(noise_steps=noise_steps,  data_length=sampleLength)
    diffusion.gmm_max_components = int(gmm_max_components)
    diffusion.gmm_reg_covar = float(gmm_reg_covar)
    logger = SummaryWriter(os.path.join("runs", modelName))
    createPathIfNotExist(r"./models/{}_{}/".format(modelName, generateLabel))
    createPathIfNotExist(r"./results/{}_{}/".format(modelName, generateLabel))

    # training dataset  paderborn cwru - 使用时域直接训练
    trainset = FFTSignalDataset(dataSource=datasets, numOfClass=generateLabel, numOfData=sampleNumber, lengthOfSample=sampleLength, transform='fft', normalize='minmax')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
    l = len(trainloader)

    gmm_train_samples = trainset.dataset
    if isinstance(gmm_train_samples, torch.Tensor):
        gmm_train_samples = gmm_train_samples.detach().cpu().numpy()
    gmm_train_samples = gmm_train_samples[:, None, :]
    diffusion.fit_gmm(gmm_train_samples)
    gmm_params_path = os.path.join("models", "{}_{}".format(modelName, generateLabel), "gmm_params.npz")
    diffusion.save_gmm_params(gmm_params_path)
    # plot some sample picture

  
    for epoch in range(epochs + 1):
        logging.info(f"Starting epoch {epoch}:")
        epoch_loss_accum = 0.0
        for step, batch in enumerate(trainloader):
            inputData = batch['data'].to(device)
            inputData = inputData.unsqueeze(1)
            t = torch.rand(inputData.shape[0], device=device) * float(max(1, diffusion.noise_steps - 1))            
            #t为一个batch的随机时间步长，取值范围在0到noise_steps-1之间的浮点数
            t_cont = (t / float(max(1, diffusion.noise_steps - 1))).to(device)#归一化到0-1之间
            x_t, v_target = diffusion.noise_data(inputData, t_cont)
            pre_v_target = model(x_t, t_cont)#预测的向量场

            loss_l1 = sl1(pre_v_target, v_target)
            loss_mse = mse(pre_v_target, v_target)
            loss = 0.01 * loss_l1 + 0.99 * loss_mse

            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪，防止剧烈震荡
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # EMA 更新
            with torch.no_grad():
                for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                    ema_p.data.mul_(ema_decay).add_(p.data * (1.0 - ema_decay))

            epoch_loss_accum += loss.item()

            if epoch % 1000 == 0:
                print('[%d/%d]\tLoss: %.3f' % (epoch, epochs,  loss.item()))
            logger.add_scalar("loss", loss.item(), global_step=epoch * l + step)

        # 学习率调度使用 epoch 平均损失作为指标
        epoch_loss = epoch_loss_accum / max(1, l)
        scheduler.step(epoch_loss)

        # 每若干 epoch 用 EMA 模型采样保存图像并保存 EMA ckpt（EMA 权重更稳定）
        if (epoch >= max(0, epochs-49)) and ((epoch % 10) == 0):
            sampled_images = diffusion.sample(ema_model, n=min(16, inputData.shape[0]))
            sampled_images = sampled_images.detach().cpu()
            '''
            f, a = plt.subplots(3, 1, figsize=(20, 30))
            for i in range(3):
                index = random.randint(0, len(sampled_images) - 1)
                a[i].plot(sampled_images[index].view(-1), lw=1, ls='-', c='b', alpha=0.8)
                a[i].set_yticks(([0, 0.25, 0.5, 0.75, 1]))
            plt.savefig(os.path.join("results", "{}_{}".format(modelName, generateLabel), f"{epoch}.jpg"))
            plt.close()'''

        # 保存 EMA 权重（覆盖）
        torch.save(ema_model.state_dict(), os.path.join("models", "{}_{}".format(modelName, generateLabel), "ckpt.pt"))

def generateSamples(generateLabel=generateLabel, generateNumber=generateNumber, generateBatchsize=generateBatchsize,
                    sampler: str = sample_sampler, steps: int = sample_steps):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet1D(c_in=1, c_out=1, time_dim=256, device=device).to(device)
    # 这里我们期望 ckpt 是 EMA 权重（训练时保存为 EMA）
    ckpt_path = os.path.join(modelSavingPath, "{}_{}/ckpt.pt".format(modelName, generateLabel))
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    diffusion = Diffusion(noise_steps=noise_steps, data_length=sampleLength)
    diffusion.gmm_max_components = int(gmm_max_components)
    diffusion.gmm_reg_covar = float(gmm_reg_covar)
    createPathIfNotExist(r"./models/{}_{}/".format(modelName, generateLabel))
    createPathIfNotExist(r"./results/{}_{}/".format(modelName, generateLabel))

    gmm_params_path = os.path.join(modelSavingPath, "{}_{}/gmm_params.npz".format(modelName, generateLabel))
    if not os.path.exists(gmm_params_path):
        raise FileNotFoundError(f"GMM params not found: {gmm_params_path}")
    diffusion.load_gmm_params(gmm_params_path)

    if device == "cuda":
        torch.cuda.empty_cache()

    # Avoid huge batches for long signals (Paderborn 4096) + attention.
    safe_batch_cap = 32 if sampleLength >= 2048 else 128
    cur_bs = int(max(1, min(generateBatchsize, safe_batch_cap)))

    all_samples = []
    remaining = int(generateNumber)
    while remaining > 0:
        this_bs = int(min(cur_bs, remaining))
        try:
            sampled_images = diffusion.sample(model, n=this_bs, steps=steps, sampler=sampler)
        except torch.OutOfMemoryError:
            if device == "cuda":
                torch.cuda.empty_cache()
            if this_bs <= 1:
                raise
            cur_bs = max(1, this_bs // 2)
            continue

        sampled_images = sampled_images.detach().float().cpu().numpy()
        sampled_images = np.squeeze(sampled_images)
        if sampled_images.ndim == 1:
            sampled_images = sampled_images[None, :]

        # 时域训练：保持与原始数据相同的归一化方式（逐样本 min-max 归一化）
        # 直接对生成的数据进行逐样本min-max归一化
        mins = sampled_images.min(axis=1, keepdims=True)
        maxs = sampled_images.max(axis=1, keepdims=True)
        denom = maxs - mins + 1e-8
        sampled_images = (sampled_images - mins) / denom  # 归一化到[0, 1]
        
        all_samples.append(sampled_images.astype(np.float32, copy=False))

        remaining -= this_bs
        del sampled_images
        if device == "cuda":
            torch.cuda.empty_cache()

    all_samples = np.concatenate(all_samples, axis=0)
    
    # 保存生成的样本（与原始数据格式完全相同，都是归一化的[0,1]）
    mat_payload = {'result': all_samples}
    dataset_mat_path = f'{dataSavingPath}/{modelName}_{datasets}_time_class{generateLabel}.mat'
    legacy_mat_path = f'{dataSavingPath}/{modelName}_time_class{generateLabel}.mat'
    sio.savemat(dataset_mat_path, mat_payload)
    # Keep the historical filename for older analysis scripts.
    #sio.savemat(legacy_mat_path, mat_payload)
    #print(f'Saving mat: {dataSavingPath}/{modelName}_time_class{generateLabel}.mat')

def trainAndValidation(num_epochs, batch_size, count):
    #set_seed(123)
    netC = Classifier(numOfClasses).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(netC.parameters(), lr=0.001)

    # load mat data (时域生成的数据)
    mat_paths = []
    for generateLabel in range(0, numOfClasses):
        createPathIfNotExist(r"./results/{}_{}".format(modelName, generateLabel))
        matPath = dataSavingPath + r'{}_{}_time_class{}.mat'.format(modelName, datasets, generateLabel)
        if not os.path.exists(matPath):
            matPath = dataSavingPath + r'{}_time_class{}.mat'.format(modelName, generateLabel)
        mat_paths.append(matPath)

    # initial dataset
    dataset = MyDataset(mat_paths)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # train classifier
    for epoch in range(num_epochs):
        running_loss = 0.0

        # Iterate over the data loader
        for batch in data_loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = netC(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    # validation data - 使用时域直接训练
    validation_dataset = FFTSignalDataset(dataSource=datasets, numOfClass=numOfClasses, numOfData=1000, lengthOfSample=sampleLength, transform='fft', normalize='minmax')
    validation_data_loader = DataLoader(validation_dataset, batch_size=generateNumber, shuffle=True)
    netC.eval()

    # Initialize variables for tracking accuracy and loss
    total_predictions = 0
    correct_predictions = 0
    validation_loss = 0.0

    # set class confusion matrix
    confusion_matrix = np.zeros((numOfClasses, numOfClasses))

    # Iterate over the validation data loader
    for batch in validation_data_loader:
        inputs = torch.squeeze(batch['data']).to(device)
        labels = torch.squeeze(batch['label'].long()).to(device)

        # Disable gradient calculation to speed up inference
        with torch.no_grad():            
            outputs = netC(inputs)
            loss = criterion(outputs, labels)#
            _, predicted = torch.max(outputs.data, 1)

            # Update accuracy statistics
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            # Accumulate the validation loss
            validation_loss += loss.item() * labels.size(0)

            # Update confusion matrix
            for i in range(len(labels)):
                true_label = labels[i].item()
                predicted_label = predicted[i].item()
                confusion_matrix[true_label][predicted_label] += 1

    # Calculate average validation loss and accuracy
    average_validation_loss = validation_loss / len(validation_data_loader.dataset)
    accuracy = correct_predictions / total_predictions

    # F1 scores
    tp = np.diag(confusion_matrix)
    fp = confusion_matrix.sum(axis=0) - tp
    fn = confusion_matrix.sum(axis=1) - tp
    precision = tp / (tp + fp + 1e-8)#这句的作用是计算每个类别的精确率（Precision）。精确率定义为：TP / (TP + FP)，其中 TP 是真正例的数量，FP 是假正例的数量。为了避免除以零的情况，分母中添加了一个小的常数（1e-8）。
    recall = tp / (tp + fn + 1e-8)#这句的作用是计算每个类别的召回率（Recall）。召回率定义为：TP / (TP + FN)，其中 TP 是真正例的数量，FN 是假负例的数量。同样地，为了避免除以零的情况，分母中添加了一个小的常数（1e-8）。
    f1_per_class = 2 * precision * recall / (precision + recall + 1e-8)#这句的作用是计算每个类别的 F1 分数。F1 分数是精确率和召回率的调和平均数，定义为：2 * (Precision * Recall) / (Precision + Recall)。为了避免除以零的情况，分母中添加了一个小的常数（1e-8）。最终得到的是一个数组，其中每个元素对应一个类别的 F1 分数。
    macro_f1 = np.mean(f1_per_class)#这句的作用是计算宏平均 F1 分数（Macro-F1）。宏平均 F1 分数是对所有类别的 F1 分数取平均值，定义为：Macro-F1 = (F1_class1 + F1_class2 + ... + F1_classN) / N，其中 N 是类别的数量。通过计算宏平均 F1 分数，可以得到一个整体的性能指标，反映模型在所有类别上的表现。
    weighted_f1 = np.sum(f1_per_class * confusion_matrix.sum(axis=1)) / (confusion_matrix.sum() + 1e-8)#这句的作用是计算加权平均 F1 分数（Weighted-F1）。加权平均 F1 分数是对每个类别的 F1 分数进行加权平均，权重通常是每个类别的样本数量。定义为：Weighted-F1 = (F1_class1 * N_class1 + F1_class2 * N_class2 + ... + F1_classN * N_classN) / (N_class1 + N_class2 + ... + N_classN)，其中 F1_classi 是第 i 个类别的 F1 分数，N_classi 是第 i 个类别的样本数量。通过计算加权平均 F1 分数，可以得到一个整体的性能指标，反映模型在所有类别上的表现，同时考虑了类别不平衡的问题。

    print(f'Validation Loss: {average_validation_loss:.4f}, Accuracy: {accuracy * 100:.2f}%, Macro-F1: {macro_f1:.4f}, Weighted-F1: {weighted_f1:.4f}')
    print('Per-class F1:', ', '.join([f'{f1:.4f}' for f1 in f1_per_class]))

    plot_confusion_matrix(confusion_matrix, classes=['{}'.format(i) for i in range(numOfClasses)],
                          savingPath=r"./confusionmatrix/", name=r"{}_{}".format(modelName, count))
    return accuracy

if __name__ == '__main__':
    set_seed(123)
    for re in range(1):
        start_time = time.time()  # 记录开始时间
        train_model_elapsed = 0.0
        generate_data_elapsed = 0.0
        train_classifier_elapsed = 0.0
        print(f"-------------------------------- 第---{re}--{re}--{re}--次迭代 -----------------------------------")
        if retrain:
            train_model_start = time.time()
            print("============================ 扩散模型  训练 ==========================")
            for i in range(0, numOfClasses):
                trainFDDiffusion(generateLabel=i, sampleNumber=sampleNumber, noise_steps=noise_steps, epochs=epochs)
                print(f"{i}__Class 扩散模型训练完成.")
            train_model_elapsed = time.time() - train_model_start
            print(f"[耗时] 生成模型训练: {format_elapsed(train_model_elapsed)}")

        if regenerate:
            generate_data_start = time.time()
            print("============================ 扩散模型  生成 ==========================")
            for i in range(0, numOfClasses):
                generateSamples(generateLabel=i)
                print(f"{i}__Class 生成数据完成.")
            generate_data_elapsed = time.time() - generate_data_start
            print(f"[耗时] 数据生成: {format_elapsed(generate_data_elapsed)}")

        train_classifier_start = time.time()
        accuracyList = []
        for time_idx in range(0, 10):
            accuracy = trainAndValidation(num_epochs=num_epochs_classifier, batch_size=batch_size_classifier, count=time_idx)
            accuracyList.append(accuracy)
        train_classifier_elapsed = time.time() - train_classifier_start
        print(f"[耗时] 分类器训练与验证: {format_elapsed(train_classifier_elapsed)}")


        ACC_MEAN = sum(accuracyList) / (len(accuracyList)+0.0001)
        ACC_MAX = max(accuracyList)

        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算耗费时间

        print(f"第 {re} 次测试耗费时间为： {format_elapsed(elapsed_time)}")
        print("------------------------- 阶段耗时汇总 -------------------------")
        print(f"生成模型训练耗时: {format_elapsed(train_model_elapsed)}")
        print(f"数据生成耗时: {format_elapsed(generate_data_elapsed)}")
        print(f"分类器训练与验证耗时: {format_elapsed(train_classifier_elapsed)}")
        print("************************************************* 诊断 结果 **************************************************")
        print("accuracy:", [f"{acc*100:.2f}" for acc in accuracyList])
        print('max: ', f"{ACC_MAX*100:.2f}", 'mean: ', f"{ACC_MEAN*100:.2f}")
