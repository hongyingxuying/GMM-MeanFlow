import torch
import numpy as np
import os
import random
import csv
import scipy.io as scio
from pylab import mpl
from numpy.fft import fft
from numpy import linspace, sin, pi, power, ceil, log2, arange
from matplotlib import pyplot as plt
import torch.nn.functional as F
from scipy.linalg import svd
mpl.rcParams['font.sans-serif'] = ['SimHei']

paderbornDataPath = r'.\datasets\Paderborn'
cwruDataPath = r'.\datasets\cwru\\'
xjtuDataPath = r'.\datasets\xjtu\\'

#paderbornDataPath = r'D:\Doctorate\Datasets\Paderborn'
#cwruDataPath = r'D:\Doctorate\Datasets\cwru\\'
#xjtuDataPath = r'D:\Doctorate\Datasets\xjtu\\'

class Dataset():
    def __init__(self, dataSource, numOfClass, numOfData, lengthOfSample):
        self.dataset = np.array([])
        self.labelset = np.array([])
        self.dataSource = dataSource
        self.numOfClass = numOfClass
        self.numOfData = numOfData
        self.lengthOfSample = lengthOfSample
        self.stepToPickSample = 10

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        step = self.dataset[idx, :]
        step = torch.unsqueeze(step, 0)
        target = self.labelset[idx]
        return step, target

    def pickSamples(self, numOfsample):
        length = len(self.dataset)
        num = int(numOfsample)
        if numOfsample > length:
            raise Exception('sample number is bigger than length of data !')
        index = random.sample(range(length), num)
        dataset = self.dataset[index]
        label = self.labelset[index]
        return dataset, label

    def build_dataset(self):
        '''get dataset of signal'''
        if self.dataSource == 'paderborn':
            self.loadPaderbornDataset()
            print('Paderborn dataset loaded !')
            return
        if self.dataSource == 'cwru':
            self.loadCWRUData()
            print('CWRU dataset loaded !')
            return

    def dealWithData(self, data, step, sampleLength):#振动数据，10,1024
        data = np.reshape(data, (-1))
        num = (len(data) - sampleLength) // step#每次滑动10，计算总供能提取多少个1024样本。
        data = data[0:num * step + sampleLength]#[0-(n*step+1024)],对原数据取整除1024的部分
        output = []
        for i in range(num):
            tmpData = data[i * step:i * step + sampleLength]
            output.append(tmpData)
        data = np.array(output)#提取num*1024个样本
        return data

    def sampleData(self, data, label, sampleNumber):
        length = len(data)
        num = int(sampleNumber)
        if sampleNumber > length:
            raise Exception('sample number is bigger than length of data !')
        index = random.sample(range(length), num)#随机抽取20个样本
        dataset = data[index]
        label = label[index]
        return dataset, label

    def _resolve_xjtu_bearing_dir(self, bearing_name: str):
        candidates = [
            os.path.join(xjtuDataPath, bearing_name),
        ]
        if bearing_name.startswith('Bearing') and ' ' not in bearing_name:
            spaced = 'Bearing ' + bearing_name[len('Bearing'):]
            candidates.append(os.path.join(xjtuDataPath, spaced))
        for path in candidates:
            if os.path.isdir(path):
                return path
        raise FileNotFoundError(f"XJTU bearing folder not found for {bearing_name} under {xjtuDataPath}")

    def _read_xjtu_csv_column(self, csv_path: str, column_name: str = 'Horizontal_vibration_signals'):
        values = []
        with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                use_col = column_name if column_name in reader.fieldnames else reader.fieldnames[0]
                for row in reader:
                    value = row.get(use_col, '')
                    if value is None or value == '':
                        continue
                    try:
                        values.append(float(value))
                    except ValueError:
                        continue
            else:
                f.seek(0)
                for line in f:
                    parts = line.strip().split(',')
                    if not parts:
                        continue
                    try:
                        values.append(float(parts[0]))
                    except ValueError:
                        continue
        return np.asarray(values, dtype=np.float32)

    def _load_xjtu_bearing_data(self, bearing_name: str, last_n: int = 10):
        bearing_dir = self._resolve_xjtu_bearing_dir(bearing_name)
        csv_files = [f for f in os.listdir(bearing_dir) if f.lower().endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {bearing_dir}")
        csv_files.sort()
        selected = csv_files[-last_n:]
        signals = []
        for fname in selected:
            csv_path = os.path.join(bearing_dir, fname)
            col_data = self._read_xjtu_csv_column(csv_path)
            if col_data.size > 0:
                signals.append(col_data)
        if not signals:
            raise ValueError(f"No valid signal data found in {bearing_dir}")
        return np.concatenate(signals, axis=0)

    def getXJTUClassData(self, bearing_name: str, max_samples: int = None, last_n: int = 10):
        raw = self._load_xjtu_bearing_data(bearing_name, last_n=last_n)
        segments = self.dealWithData(raw, self.stepToPickSample, self.lengthOfSample)
        if max_samples is not None and segments.shape[0] > max_samples:
            labels = np.zeros((segments.shape[0], 1))
            segments, _ = self.sampleData(segments, labels, max_samples)
        return segments

    def datasetToTorch(self):
        if not isinstance(self.dataset, torch.Tensor):
            self.dataset = torch.from_numpy(self.dataset).float()
        if not isinstance(self.labelset, torch.Tensor):
            self.labelset = torch.from_numpy(self.labelset).float()

    def minmax_normalize(self):
        '''return minmax normalize dataset'''
        for index in range(self.length):
            # self.dataset[:, index] = (self.dataset[:, index] - self.dataset[:, index].min()) / (
            #     self.dataset[:, index].max() - self.dataset[:, index].min())
            self.dataset[index, :] = (self.dataset[index, :] - self.dataset[index, :].min()) / (
                self.dataset[index, :].max() - self.dataset[index, :].min())

    '''
    ------------------------------------------------------Paderborn-------------------------------------------------------------------
    '''
    def openPaderbornData(self, FK, K, N, M, F, num):#读取Paderborn数据，num∈（1, 20）
        fileName = str(r'N{}_M{}_F{}_{}{}_{}'.format(N, M, F, FK, K, num))#N{N}_M{M}_F{F}_{FK}{K}_{num}
        path = paderbornDataPath + str(r'\\{}{}\\'.format(FK, K)) + str(fileName) + str(r'.mat')
        originData = scio.loadmat(path)
        data = originData[fileName]['Y'][0][0][0][6][2][0]

        #读取“Y”这个变量中的第7行第3列的第1个元素（从0开始），取出这个数组作为数据。
        data = np.array(data)
        #print('----> add data ' + fileName + ' with size: ', data.shape)
        return data

    def getPaderbornSeriesData(self, FK, K, N, M, F, target_samples=5000):#同一系列数据，num从1到20拼接在一起     
        chunks = []
        samples_count = 0
        for num in range(1, 21):
            tmp = self.openPaderbornData(FK, K, N, M, F, num)
            tmp = self.dealWithData(tmp, self.stepToPickSample, self.lengthOfSample)
            chunks.append(tmp)
            samples_count += len(tmp)
            # 一旦有足够样本，就停止加载更多数据以节省内存
            if samples_count >= target_samples:
                break

        dataList = np.vstack(chunks) if len(chunks) > 0 else np.empty((0, self.lengthOfSample))
        # 如果加载的样本超过目标数量，随机采样到目标数量
        if len(dataList) > target_samples:
            indices = np.random.choice(len(dataList), target_samples, replace=False)
            dataList = dataList[indices]
        # print('**** series data loaded ****', '\n', 'series data length: ', dataList.shape)
        return dataList

    '''
    ------------------------------------------------------CWRU-------------------------------------------------------------------
    '''
    def openCWRUData(self, bathPath, key_num):
        path = bathPath + str(key_num) + '.mat'
        str1 = "X" + "%03d" % key_num + "_DE_time"#生成变量名：X{key_num}_DE_time
        data = scio.loadmat(path)#加载数据
        data = np.array(data[str1])#加载数据  原始振动信号包括驱动端_DE,风扇端_FE,和基座端_BA,这里仅用DE
        return data
   

class FFTSignalDataset(Dataset):#继承于Dataset
    # dataSource = 'cwru', numOfClass = generateLabel, numOfData = sampleNumber 20, lengthOfSample = sampleLength 1024, fs = 12000
    # transform: 'fft' (default, magnitude spectrum) | 'time' (raw segment)
    # normalize: 'minmax' (default) | 'none'
    def __init__(self, dataSource, numOfClass, numOfData, lengthOfSample, fs=None, transform: str = 'fft', normalize: str = 'minmax'):
        super(FFTSignalDataset, self).__init__(dataSource, numOfClass, numOfData, lengthOfSample)
        self.transform = (transform or 'fft').strip().lower()
        if self.transform not in {'fft', 'time'}:
            raise ValueError("transform must be 'fft' or 'time'")
        self.normalize = (normalize or 'minmax').strip().lower()
        if self.normalize not in {'minmax', 'none'}:
            raise ValueError("normalize must be 'minmax' or 'none'")
        default_fs = None
        # 根据 dataSource 选择数据集
        if self.dataSource == 'paderborn':
            #print('Loading Paderborn dataset...')
            self.loadPaderbornDataset()
            default_fs = 12000
        
        elif self.dataSource == 'cwru':
            #print('Loading CWRU dataset...')
            self.cwru4Classes(outputClass=numOfClass)
            default_fs = 12000
        elif self.dataSource == 'xjtu':
            #print('Loading XJTU dataset...')
            self.loadxjtuDataset()
            default_fs = 25600
        else:
            raise ValueError(f"Unknown dataSource: {self.dataSource}")

        if fs is None:
            fs = default_fs
        self.fs = fs

        self.length = self.dataset.shape[0]

        if self.transform == 'fft':
            self.toFFTSignal(self.fs)

        if self.normalize == 'minmax':
            self.minmax_normalize()
        self.datasetToTorch()

    def __getitem__(self, idx):
        sample = {'data': self.dataset[idx], 'label': self.labelset[idx]}
        return sample

    def cwru4Classes(self, outputClass):
        faultDataPath = cwruDataPath + r'\12k Drive End Bearing Fault Data\\'
        normalDataPath = cwruDataPath + r'\12k Drive End Bearing Fault Data\\'
        hp = 0

        innerData7 = self.openCWRUData(faultDataPath, 105)
        innerData7 = self.dealWithData(innerData7, self.stepToPickSample, self.lengthOfSample)#；
        innerLabel7 = np.ones((innerData7.shape[0], 1)) * 0  #故障标签为0
        innerData7, innerLabel7 = self.sampleData(innerData7, innerLabel7, self.numOfData)#从num个样本中随机抽取20个样本

        innerData14 = self.openCWRUData(faultDataPath, 169)
        innerData14 = self.dealWithData(innerData14, self.stepToPickSample, self.lengthOfSample)#；
        innerLabel14 = np.ones((innerData14.shape[0], 1)) * 1 #故障标签为1
        innerData14, innerLabel14 = self.sampleData(innerData14, innerLabel14, self.numOfData)#从num个样本中随机抽取20个样本

        innerData21 = self.openCWRUData(faultDataPath, 209)
        innerData21 = self.dealWithData(innerData21, self.stepToPickSample, self.lengthOfSample)#；
        innerLabel21 = np.ones((innerData21.shape[0], 1)) * 2#故障标签为2
        innerData21, innerLabel21 = self.sampleData(innerData21, innerLabel21, self.numOfData)#从num个样本中随机抽取20个样本

        ballData7 = self.openCWRUData(faultDataPath, 118)
        ballData7 = self.dealWithData(ballData7, self.stepToPickSample, self.lengthOfSample)
        ballLabel7 = np.ones((ballData7.shape[0], 1)) * 3#故障标签为3
        ballData7, ballLabel7 = self.sampleData(ballData7, ballLabel7, self.numOfData)

        ballData14 = self.openCWRUData(faultDataPath, 185)
        ballData14 = self.dealWithData(ballData14, self.stepToPickSample, self.lengthOfSample)
        ballLabel14 = np.ones((ballData14.shape[0], 1)) * 4#故障标签为4
        ballData14, ballLabel14 = self.sampleData(ballData14, ballLabel14, self.numOfData)

        ballData21 = self.openCWRUData(faultDataPath, 222)
        ballData21 = self.dealWithData(ballData21, self.stepToPickSample, self.lengthOfSample)
        ballLabel21 = np.ones((ballData21.shape[0], 1)) * 5#故障标签为5
        ballData21, ballLabel21 = self.sampleData(ballData21, ballLabel21, self.numOfData)

        outerData7 = self.openCWRUData(faultDataPath, 130)
        outerData7 = self.dealWithData(outerData7, self.stepToPickSample, self.lengthOfSample)
        outerLabel7 = np.ones((outerData7.shape[0], 1)) * 6#故障标签为6
        outerData7, outerLabel7 = self.sampleData(outerData7, outerLabel7, self.numOfData)

        outerData14 = self.openCWRUData(faultDataPath, 197)
        outerData14 = self.dealWithData(outerData14, self.stepToPickSample, self.lengthOfSample)
        outerLabel14 = np.ones((outerData14.shape[0], 1)) * 7#故障标签为7
        outerData14, outerLabel14 = self.sampleData(outerData14, outerLabel14, self.numOfData)

        outerData21 = self.openCWRUData(faultDataPath, 234)
        outerData21 = self.dealWithData(outerData21, self.stepToPickSample, self.lengthOfSample)
        outerLabel21 = np.ones((outerData21.shape[0], 1)) * 8#故障标签为8
        outerData21, outerLabel21 = self.sampleData(outerData21, outerLabel21, self.numOfData)

        normalData = self.openCWRUData(normalDataPath, 97)
        normalData = self.dealWithData(normalData, self.stepToPickSample, self.lengthOfSample)
        normalLabel = np.ones((normalData.shape[0], 1)) * 9#故障标签为9
        normalData, normalLabel = self.sampleData(normalData, normalLabel, self.numOfData)

        if outputClass == 0:
            self.dataset = innerData7
            self.labelset = innerLabel7
        elif outputClass == 1:
            self.dataset = innerData14
            self.labelset = innerLabel14
        elif outputClass == 2:
            self.dataset = innerData21
            self.labelset = innerLabel21
        elif outputClass == 3:
            self.dataset = ballData7
            self.labelset = ballLabel7
        elif outputClass == 4:
            self.dataset = ballData14
            self.labelset = ballLabel14
        elif outputClass == 5:
            self.dataset = ballData21
            self.labelset = ballLabel21
        elif outputClass == 6:
            self.dataset = outerData7
            self.labelset = outerLabel7
        elif outputClass == 7:
            self.dataset = outerData14
            self.labelset = outerLabel14
        elif outputClass == 8:
            self.dataset = outerData21
            self.labelset = outerLabel21        
        elif outputClass == 9:
            self.dataset = normalData
            self.labelset = normalLabel
        else:
            self.dataset = np.vstack((innerData7,innerData14,innerData21, ballData7, ballData14, ballData21, outerData7, outerData14, outerData21, normalData))
            self.labelset = np.vstack((innerLabel7, innerLabel14, innerLabel21, ballLabel7, ballLabel14, ballLabel21, outerLabel7, outerLabel14, outerLabel21, normalLabel))

        return

    def loadPaderbornDataset(self):
        #正常数据
        K001_N15_M07_F10_data = self.getPaderbornSeriesData("K", "001", "15", "07", "10")#k健康数据
        #"K""001"_N"15"_M"07"_F"10" K_损失部位，001_编号，N_轴承转速（15,9），M_负载（07,01），F_径向力（10,04）
        labelK001 = np.ones((K001_N15_M07_F10_data.shape[0], 1)) * 0
        K001_N15_M07_F10_data, labelK001 = self.sampleData(K001_N15_M07_F10_data, labelK001, self.numOfData)     

        #outer ring
        KA01_N15_M07_F10_data = self.getPaderbornSeriesData("KA", "01", "15", "07", "10")#EDM
        labelKA01 = np.ones((KA01_N15_M07_F10_data.shape[0], 1)) * 1
        KA01_N15_M07_F10_data, labelKA01 = self.sampleData(KA01_N15_M07_F10_data, labelKA01, self.numOfData)

        KA03_N15_M07_F10_data = self.getPaderbornSeriesData("KA", "03", "15", "07", "10")#Electric engraver
        labelKA03 = np.ones((KA03_N15_M07_F10_data.shape[0], 1)) * 2
        KA03_N15_M07_F10_data, labelKA03 = self.sampleData(KA03_N15_M07_F10_data, labelKA03, self.numOfData)

        KA04_N15_M07_F10_data = self.getPaderbornSeriesData("KA", "04", "15", "07", "10")#Fatigue pitting
        labelKA04 = np.ones((KA04_N15_M07_F10_data.shape[0], 1)) * 3
        KA04_N15_M07_F10_data, labelKA04 = self.sampleData(KA04_N15_M07_F10_data, labelKA04, self.numOfData)


        #inner ring
        KI01_N15_M07_F10_data = self.getPaderbornSeriesData("KI", "01", "15", "07", "10")#EDM
        labelKI01 = np.ones((KI01_N15_M07_F10_data.shape[0], 1)) * 4
        KI01_N15_M07_F10_data, labelKI01 = self.sampleData(KI01_N15_M07_F10_data, labelKI01, self.numOfData)

        KI07_N15_M07_F10_data = self.getPaderbornSeriesData("KI", "07", "15", "07", "10")#Electric engraver
        labelKI07 = np.ones((KI07_N15_M07_F10_data.shape[0], 1)) * 5
        KI07_N15_M07_F10_data, labelKI07 = self.sampleData(KI07_N15_M07_F10_data, labelKI07, self.numOfData)

        KI21_N15_M07_F10_data = self.getPaderbornSeriesData("KI", "21", "15", "07", "10")#Fatigue pitting        
        labelKI21 = np.ones((KI21_N15_M07_F10_data.shape[0], 1)) * 6
        KI21_N15_M07_F10_data, labelKI21 = self.sampleData(KI21_N15_M07_F10_data, labelKI21, self.numOfData)

        
        #hybrid fault
        KB23_N15_M07_F10_data = self.getPaderbornSeriesData("KB", "23", "15", "07", "10")#Fatigue pitting
        labelKB23 = np.ones((KB23_N15_M07_F10_data.shape[0], 1)) * 7
        KB23_N15_M07_F10_data, labelKB23 = self.sampleData(KB23_N15_M07_F10_data, labelKB23, self.numOfData)

        KB24_N15_M07_F10_data = self.getPaderbornSeriesData("KB", "24", "15", "07", "10")#Fatigue pitting
        labelKB24 = np.ones((KB24_N15_M07_F10_data.shape[0], 1)) * 8
        KB24_N15_M07_F10_data, labelKB24 = self.sampleData(KB24_N15_M07_F10_data, labelKB24, self.numOfData)

        KB27_N15_M07_F10_data = self.getPaderbornSeriesData("KB", "27", "15", "07", "10")#Plastic deform：indentations
        labelKB27 = np.ones((KB27_N15_M07_F10_data.shape[0], 1)) * 9
        KB27_N15_M07_F10_data, labelKB27 = self.sampleData(KB27_N15_M07_F10_data, labelKB27, self.numOfData)


        if self.numOfClass == 0:
            self.dataset = K001_N15_M07_F10_data
            self.labelset = labelK001

        # outer ring fault   
        elif self.numOfClass == 1:
            self.dataset = KA01_N15_M07_F10_data
            self.labelset = labelKA01
        elif self.numOfClass == 2:
            self.dataset = KA03_N15_M07_F10_data
            self.labelset = labelKA03
        elif self.numOfClass == 3:
            self.dataset = KA04_N15_M07_F10_data
            self.labelset = labelKA04

        # inner ring fault
        elif self.numOfClass == 4:
            self.dataset = KI01_N15_M07_F10_data
            self.labelset = labelKI01
        elif self.numOfClass == 5:
            self.dataset = KI07_N15_M07_F10_data
            self.labelset = labelKI07
        elif self.numOfClass == 6:
            self.dataset = KI21_N15_M07_F10_data
            self.labelset = labelKI21

        # hybrid fault
        elif self.numOfClass == 7:
            self.dataset = KB23_N15_M07_F10_data
            self.labelset = labelKB23
        elif self.numOfClass == 8:
            self.dataset = KB24_N15_M07_F10_data
            self.labelset = labelKB24
        elif self.numOfClass == 9:
            self.dataset = KB27_N15_M07_F10_data
            self.labelset = labelKB27
        
        else:
            self.dataset = np.vstack((K001_N15_M07_F10_data, KA01_N15_M07_F10_data,KA03_N15_M07_F10_data,KA04_N15_M07_F10_data, KI01_N15_M07_F10_data, 
                                      KI07_N15_M07_F10_data, KI21_N15_M07_F10_data, KB23_N15_M07_F10_data, KB24_N15_M07_F10_data, KB27_N15_M07_F10_data))
            self.labelset = np.vstack((labelK001, labelKA01, labelKA03, labelKA04, labelKI01, labelKI07, labelKI21, labelKB23, labelKB24, labelKB27))
        return

    def loadxjtuDataset(self):
        xjtu_bearings = [
            ('Bearing1_1', 0),
            ('Bearing1_2', 1),
            ('Bearing1_3', 2),
            ('Bearing2_1', 3),
            ('Bearing2_2', 4),
            ('Bearing2_3', 5),
            ('Bearing2_4', 6),
            ('Bearing3_1', 7),
            ('Bearing3_2', 8),
            ('Bearing3_3', 9),
        ]

        data_list = []
        label_list = []
        for bearing_name, label in xjtu_bearings:
            data = self.getXJTUClassData(bearing_name, max_samples=self.numOfData, last_n=10)
            labels = np.ones((data.shape[0], 1)) * label
            data_list.append(data)
            label_list.append(labels)
            print(f"{bearing_name} class data sampled, shape: {data.shape}")

        if 0 <= self.numOfClass <= 9:
            self.dataset = data_list[self.numOfClass]
            self.labelset = label_list[self.numOfClass]
        else:
            self.dataset = np.vstack(data_list)
            self.labelset = np.vstack(label_list)
        return
    

    def datasetMultiply(self, multi):
        tempset = self.dataset
        tempset = tempset * multi
        self.dataset = tempset
        return
    
    def toFFTSignal(self, fs):
        for index, rawSignal in enumerate(self.dataset):
            fre, fft = self.FFT(fs, rawSignal)
            self.dataset[index] = fft
        return

    def FFT(self, fs, data):
        len_ = len(data)
        # n = int(power(2, ceil(log2(len_))))
        n = len_ * 2 + 2
        data = np.squeeze(data)
        fft_y_ = (fft(data, n)) / len_ * 2
        fre_ = arange(int(n / 2)) * fs / n
        fft_y_ = fft_y_[range(int(n / 2))]
        return fre_[1:], abs(fft_y_[1:])


