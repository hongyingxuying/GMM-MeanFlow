from itertools import filterfalse
from pickle import FALSE
import random
import torch
import torch.nn as nn
from torch import optim
import logging
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import scipy.io as sio
import time

from U_DCNN import Classifier
from MyDataset import MyDataset
from ConfuseMatrix import plot_confusion_matrix
from data_pro_FFT_extension import FFTSignalDataset
from method_loader import load_attr_from_file
Diffusion = load_attr_from_file("M4_DDPM", "M4-DDPM.py", "Diffusion")
from utils import *
from training_time_tracker import TrainingTimeTracker  # 导入训练时间跟踪器

def createPathIfNotExist(path):#创建存储路径
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def format_elapsed(seconds):
    minutes = int(seconds // 60)
    remain_seconds = seconds % 60
    return f"{minutes} 分钟 {remain_seconds:.2f} 秒"


def huber_loss(predictions, targets, delta=1.0):
    errors = torch.abs(predictions - targets)
    condition = errors > delta
    loss = torch.where(condition, 0.5 * errors**2, delta * (errors - 0.5 * delta))
    return torch.mean(loss)

use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = 0
device = torch.device("cuda:0" if use_cuda else "cpu")

# choose a model# model 1,2,3有什么区别？
from U_Net import UNet1D
modelName = 'DDPM'


# training params
retrain = True 
sampleNumber = 20
noise_steps = 100
epochs = 5000
sampleLength = 4096
batch_size = min(32, sampleNumber)
learning_rate = 3e-4

# generate params
regenerate = True #True#
generateNumber = 1000  #生成1000个1024的样本
generateBatchsize = 1000
num_epochs_classifier = 100
batch_size_classfier = 32
generateLabel = 0
numOfClasses = 10

datasets= 'xjtu'  # paderborn | cwru | xjtu
print(f"Using device: {device}, dataset: {datasets}, numOfClasses: {numOfClasses}")

resultsSavingPath = createPathIfNotExist(r"./results/")#JPG图片
modelSavingPath = createPathIfNotExist(r"./models/")#模型保存位置
dataSavingPath = createPathIfNotExist(r"./mats/")#生成的数据：1000*1024的mat数据

def trainFDDiffusion(generateLabel=generateLabel, sampleNumber=sampleNumber, noise_steps=noise_steps, epochs=epochs):
    setup_logging(modelName)
    model = UNet1D(c_in=1, c_out=1, time_dim=256, device=device).to(device)#UNETforDiagnosis.py
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    mse = nn.MSELoss()
    sl1 = nn.SmoothL1Loss()
    diffusion = Diffusion(beta_start=1e-4, beta_end=0.02, data_length=sampleLength)#DIFFUSIONforDiagnosis.py
    logger = SummaryWriter(os.path.join("runs", modelName))#主日志
    createPathIfNotExist(r"./models/{}_{}/".format(modelName, generateLabel))#创建模型存储目录
    createPathIfNotExist(r"./results/{}_{}/".format(modelName, generateLabel))#JPG图片存储目录

    # training dataset
    trainset = FFTSignalDataset(dataSource=datasets, numOfClass=generateLabel, numOfData=sampleNumber, lengthOfSample=sampleLength, fs=12000)#NEWFFTPreprocessing.py
    #返回20个故障类型为generateLabel的长度为1024的样本
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
    l = len(trainloader)
    
    # plot some sample picture
    f, a = plt.subplots(3, 1, figsize=(20, 30))#f:整图，a：子图
    for i in range(3):
        index = random.randint(0, len(trainset.dataset) - 1)#训练集中随机抽取3个样本绘制图片
        a[i].plot(trainset.dataset[index].view(-1), lw=1, ls='-', c='b', alpha=0.8)
        a[i].set_yticks(([0, 0.25, 0.5, 0.75, 1]))
    plt.savefig(r"./results/{}_{}/".format(modelName, generateLabel) + r'samplePicture.png')
    plt.close()

    for epoch in range(epochs + 1):
        logging.info(f"Starting epoch {epoch}:")
        for step, batch in enumerate(trainloader):#step=1：数据集仅产生一个batchsize；1个batch大小：20*1024；
            inputData = batch['data'].to(device)#20*1024
            inputData = inputData.unsqueeze(1)#20*1*1024
            t = diffusion.sample_timesteps(inputData.shape[0]).to(device)#随机抽取出 n=20 [1:1000,20个]时间步
            x_t, noise = diffusion.noise_data(inputData, t)#20*1*1024，噪声20*1*1024
            predicted_noise = model(x_t, t)#20*1*1024

            loss = mse(noise, predicted_noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 1000 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f' % (epoch, epochs, step, len(trainloader), loss.item()))
            logger.add_scalar("loss", loss.item(), global_step=epoch * l + step)

        if (epoch >= epochs-49) and ((epoch % 10) == 0):
            sampled_images = diffusion.sample(model, n=inputData.shape[0])
            # save image
            sampled_images = sampled_images.detach().cpu()
            f, a = plt.subplots(3, 1, figsize=(20, 30))
            for i in range(3):
                index = random.randint(0, len(sampled_images) - 1)
                a[i].plot(sampled_images[index].view(-1), lw=1, ls='-', c='b', alpha=0.8)
                a[i].set_yticks(([0, 0.25, 0.5, 0.75, 1]))
            plt.savefig(r"./results/{}_{}/{}.jpg".format(modelName, generateLabel, epoch))
            plt.close()
        torch.save(model.state_dict(), r"./models/{}_{}/ckpt.pt".format(modelName, generateLabel))

def generateSamples(generateLabel=generateLabel, generateNumber=generateNumber, generateBatchsize=generateBatchsize):
    device = "cuda"
    model = UNet1D(c_in=1, c_out=1, time_dim=256, device=device).to(device)
    ckpt = torch.load(r"{}/{}_{}/ckpt.pt".format(modelSavingPath, modelName, generateLabel))
    model.load_state_dict(ckpt)
    diffusion = Diffusion(beta_start=1e-4, beta_end=0.02, data_length=sampleLength)
    createPathIfNotExist(r"./models/{}_{}/".format(modelName, generateLabel))
    createPathIfNotExist(r"./results/{}_{}/".format(modelName, generateLabel))

    all_samples = []  # 用于存储所有生成的样本
    for _ in range(generateNumber // generateBatchsize):
        print("class_{}_batch_{}_generating ......".format(generateLabel, _))
        sampled_images = diffusion.sample(model, n=generateBatchsize)
        sampled_images = sampled_images.cpu().numpy()  # 假设sampled_images是一个Tensor
        sampled_images = np.squeeze(sampled_images)  # 去除多余的维度
        all_samples.append(sampled_images)

    # 将所有批次的样本整合到一个NumPy数组中
    all_samples = np.concatenate(all_samples, axis=0)

    # 保存到.mat文件
    sio.savemat(f'{dataSavingPath}/{modelName}_{datasets}_class{generateLabel}.mat', {'result': all_samples})
    print(f'Saving mat: {dataSavingPath}/{modelName}_{datasets}_class{generateLabel}.mat')

def trainAndValidation(num_epochs, batch_size, count):
    # initial classifier
    netC = Classifier(numOfClasses).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(netC.parameters(), lr=0.001)

    # load mat data
    mat_paths = []
    for generateLabel in range(0, numOfClasses):
        createPathIfNotExist(r"./results/{}_{}".format(modelName, generateLabel))
        matPath = dataSavingPath + r'{}_{}_class{}.mat'.format(modelName, datasets, generateLabel)
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

        # Print the average loss for this epoch
        epoch_loss = running_loss / len(data_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    # validation data
    validation_dataset = FFTSignalDataset(dataSource=datasets, numOfClass=numOfClasses, numOfData=1000, lengthOfSample=sampleLength,fs=12000)
    validation_data_loader = DataLoader(validation_dataset, batch_size=generateNumber, shuffle=False)

    # Set the model to evaluation mode
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
            # Forward pass
            outputs = netC(inputs)
            loss = criterion(outputs, labels)

            # Calculate the predicted class
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
    print("{}/{}".format(correct_predictions, total_predictions))

    # F1 scores
    tp = np.diag(confusion_matrix)
    fp = confusion_matrix.sum(axis=0) - tp
    fn = confusion_matrix.sum(axis=1) - tp
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1_per_class = 2 * precision * recall / (precision + recall + 1e-8)
    macro_f1 = np.mean(f1_per_class)
    weighted_f1 = np.sum(f1_per_class * confusion_matrix.sum(axis=1)) / (confusion_matrix.sum() + 1e-8)

    print(f'Validation Loss: {average_validation_loss:.5f}, Accuracy: {accuracy * 100:.2f}%, Macro-F1: {macro_f1:.4f}, Weighted-F1: {weighted_f1:.4f}')

    plot_confusion_matrix(confusion_matrix, classes=['{}'.format(i) for i in range(numOfClasses)],
                          savingPath=r"./confusionmatrix/", name=r"{}_{}".format(modelName, count))

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'per_class_f1': f1_per_class,
        'confusion_matrix': confusion_matrix,
        'validation_loss': average_validation_loss
    }

if __name__ == '__main__':
    # 初始化训练时间跟踪器
    time_tracker = TrainingTimeTracker(modelName)
    time_tracker.start()
    overall_start = time.time()
    train_model_elapsed = 0.0
    generate_data_elapsed = 0.0
    train_classifier_elapsed = 0.0
    
    if retrain:
        phase_start = time.time()
        for i in range(0, numOfClasses):
            print("class_{} training UNet for noise prediction......".format(i))
            trainFDDiffusion(generateLabel=i, sampleNumber=sampleNumber, noise_steps=noise_steps, epochs=epochs)
        phase_time = time.time() - phase_start
        time_tracker.phase_times['model_training'] = phase_time
        train_model_elapsed = phase_time
        print(f"[耗时] 生成模型训练: {format_elapsed(train_model_elapsed)}")
        
    accuracyList = []

    if regenerate:
        phase_start = time.time()
        for i in range(0, numOfClasses):
            print("class_{} generating samples......".format(i))
            generateSamples(generateLabel=i)
        phase_time = time.time() - phase_start
        time_tracker.phase_times['data_generation'] = phase_time
        generate_data_elapsed = phase_time
        print(f"[耗时] 数据生成: {format_elapsed(generate_data_elapsed)}")

    # Collect results from 10 iterations
    results_list = []
    classification_start = time.time()
    for t in range(0, 10):
        print("============================ Classifier Training & Validation {} ===========================".format(t))
        result_dict = trainAndValidation(num_epochs=num_epochs_classifier, batch_size=batch_size_classfier, count=t)        
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
    print("============================ Final Results ==========================")
    print("Accuracy values:", [f"{acc*100:.2f}%" for acc in accuracy_list])
    print(f"Accuracy: {acc_mean*100:.2f}% ± {acc_std*100:.2f}% (max: {acc_max*100:.2f}%)")
    print(f"Macro-F1: {macro_f1_mean:.4f} ± {macro_f1_std:.4f} (max: {macro_f1_max:.4f})")
    print(f"Weighted-F1: {weighted_f1_mean:.4f} ± {weighted_f1_std:.4f} (max: {weighted_f1_max:.4f})")
    
    # Save results to file
    result_filename = f"./results/{modelName}_{datasets}_diagnosis_result.txt"
    with open(result_filename, 'w') as f:
        f.write(f"Model: {modelName}\n")
        f.write(f"Dataset: {datasets}\n")
        f.write(f"Number of iterations: 10\n")
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
    
    print(f"\nResults saved to: {result_filename}")

    elapsed_time = time.time() - overall_start
    print(f"本次测试耗费时间为： {format_elapsed(elapsed_time)}")
    print("------------------------- 阶段耗时汇总 -------------------------")
    print(f"生成模型训练耗时: {format_elapsed(train_model_elapsed)}")
    print(f"数据生成耗时: {format_elapsed(generate_data_elapsed)}")
    print(f"分类器训练与验证耗时: {format_elapsed(train_classifier_elapsed)}")
    
    # 结束总计时并保存训练时间统计
    time_tracker.end()
    time_tracker.print_summary()
    time_tracker.save_json()
