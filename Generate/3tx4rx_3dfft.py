from MMGenerateFunc import MMGenerateFunctions, MMPlot
import numpy as np

# 雷达参数配置
numSamples = 80
numChirps = 1
numFrames = 64000
numTx = 3
numRx = 4

# 滤波器参数
lowcut = 90  # 下限频率 (Hz)
highcut = 300.0  # 上限频率 (Hz)
fs = 1600.0  # 采样频率 (Hz)

# STFT参数
window_size = 96
overlap = 0.7

# 原始数据路径
bin_path = r"/home/mambauser/MMCode/data/rawData/0_0_Raw_0.bin"

# 获取bin文件数据
data_raw = MMGenerateFunctions.getRawData_multiTx(
    bin_path, numSamples, numChirps, numFrames, numTx, numRx
)  # numRx, numFrames, numSamples

data_1dfft_dyn_all = np.zeros(data_raw[0].shape, dtype=complex)

tx = 0
for rx in range(4):
    # 1D-FFT
    data_1dfft = MMGenerateFunctions.rangeFFT(data_raw[tx][rx], numFrames, numSamples)
    # 去除静态杂波
    data_1dfft_dyn_all[rx, :, :] = MMGenerateFunctions.removeStatic(
        data_1dfft, numFrames, numSamples
    )
    # data_1dfft_dyn = data_1dfft

numAngles = 90
# 3D-FFT
data_3dfft = MMGenerateFunctions.angleFFT(
    data_1dfft_dyn_all, numFrames, numSamples, numAngles
)

range_index, angle_index = MMGenerateFunctions.getRangeAngleIndex(data_3dfft)

# 提取相位并解缠绕
range_angle, angle_diff = MMGenerateFunctions.getUnwrapAngle(
    data_3dfft[:, range_index, angle_index]
)

# 对相位差进行带通滤波
# angle_diff_filtered = MMGenerateFunctions.filteredAngle(angle_diff, lowcut, highcut, fs)
angle_diff_filtered = angle_diff

# 计算STFT
f, t, Zxx = MMGenerateFunctions.getSTFT(angle_diff_filtered, window_size, overlap, fs)

# 绘制结果
MMPlot.plotSTFT(t, f, Zxx)
# MMPlot.plotMain(data_1dfft, data_3dfft[:, :, 44], range_angle, angle_diff,
#                 angle_diff_filtered, [t, f, Zxx])
