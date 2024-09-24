import numpy as np
from mmwave.dataloader import DCA1000
from scipy.signal import butter, filtfilt, stft


# 读取原始数据
def getRawData(bin_path, numSamples, numChirps, numFrames, numTx, numRx):
    numChirpsPerFrame = numTx * numChirps
    data_raw = np.fromfile(bin_path, dtype=np.int16)
    data_raw = data_raw.reshape(numFrames, -1)
    data_raw = np.apply_along_axis(
        DCA1000.organize,
        1,
        data_raw,
        num_chirps=numChirpsPerFrame,
        num_rx=numRx,
        num_samples=numSamples)  # numFrames, numChirps, numRx, numSamples

    if (numChirps == 1):
        data_raw = np.squeeze(data_raw)  # numFrames, numRx, numSamples
        data_raw = np.transpose(data_raw,
                                (1, 0, 2))  # numRx, numFrames, numSamples

    else:
        data_raw = np.transpose(
            data_raw, (2, 0, 1, 3))  # numRx, numFrames, numChirps, numSamples
        data_raw = np.reshape(data_raw,
                              (numRx, numFrames * numChirps, numSamples))

    return data_raw


def getRawData_multiTx(bin_path, numSamples, numChirps, numFrames, numTx,
                       numRx):
    numChirpsPerFrame = numTx * numChirps
    data_raw = np.fromfile(bin_path, dtype=np.int16)
    data_raw = data_raw.reshape(numFrames, -1)
    data_raw = np.apply_along_axis(
        DCA1000.organize,
        1,
        data_raw,
        num_chirps=numChirpsPerFrame,
        num_rx=numRx,
        num_samples=numSamples)  # numFrames, numChirps, numRx, numSamples
    data_raw_lst = [
        np.transpose(np.squeeze(data_raw[:, i, :, :]), (1, 0, 2))
        for i in range(numTx)
    ]
    return data_raw_lst


def getRawData_2243(bin_path, numSamples, numChirps, numFrames, numTx, numRx):
    frames = np.fromfile(bin_path, dtype=np.int16)
    if (numChirps == 1):
        frames = frames.reshape(numRx, 2, numSamples, 1, numFrames)
        frames = frames.reshape(numFrames, 1, numSamples, 2, numRx)
        frames = np.transpose(frames)
        data_raw = np.squeeze(
            frames[:, 0, :, :, :] +
            1j * frames[:, 1, :, :, :])  # numRx, numSamples, numFrames
        data_raw = np.transpose(data_raw, (0, 2, 1))

    return data_raw


# 1D-FFT
def rangeFFT(data_raw_rx0, numFrames, numSamples):
    data_1dfft = np.zeros((numFrames, numSamples), dtype=complex)
    # window = np.hanning(numSamples)
    window = np.hamming(numSamples)
    for frame in range(numFrames):
        # 对每个样本进行FFT
        data_1dfft[frame, :] = np.fft.fft(data_raw_rx0[frame, :] * window)
        # data_1dfft[frame, :] =  np.fft.fftshift(np.fft.fft(data_raw_rx0[frame, :] * window))
    return data_1dfft


# 3D-FFT?
def angleFFT(data_raw, numFrames, numSamples, numAngles):
    data_3dfft = np.zeros((numFrames, numSamples, numAngles), dtype=complex)
    for frame in range(numFrames):
        for sample in range(numSamples):
            data_3dfft[frame, sample, :] = np.fft.fftshift(
                np.fft.fft(data_raw[:, frame, sample], n=numAngles))
    return data_3dfft


# 去除静态杂波
def removeStatic(data_1dfft, numFrames, numSamples):
    data_1dfft_dyn = np.zeros((numFrames, numSamples), dtype=complex)
    data_1dfft_avg = np.mean(data_1dfft, axis=0)
    for frame in range(numFrames):
        data_1dfft_dyn[frame, :] = data_1dfft[frame, :] - data_1dfft_avg
    return data_1dfft_dyn


def getRangeIndex(data_1dfft_dyn):
    range_index = np.argmax(np.sum(np.abs(data_1dfft_dyn), axis=0))
    return range_index

def getRangeAngleIndex(data_3dfft):
    # 对第一个维度求和，得到一个形状为 (80, 90) 的数组
    summed_data = np.sum(np.abs(data_3dfft), axis=0)
    # 找到最大值的索引
    max_index = np.argmax(summed_data)
    # 将一维索引转换为二维索引
    range_index, angle_index = np.unravel_index(max_index, summed_data.shape)
    return range_index, angle_index

    

# 提取相位并解缠绕
def getUnwrapAngle(data_1dfft_dyn_select):
    # 找出绝对值总和最大的列索引
    range_angle = np.unwrap(np.angle(data_1dfft_dyn_select))
    angle_diff = np.diff(range_angle)
    return range_angle, angle_diff


# 对相位差进行带通滤波
def filteredAngle(angle_diff, lowcut, highcut, fs, order=4):
    # 对相位差进行带通滤波
    def bandpass_filter(data, lowcut, highcut, fs, order):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='bandpass')
        y = filtfilt(b, a, data)
        return y

    angle_diff_filtered = bandpass_filter(angle_diff, lowcut, highcut, fs,
                                          order)
    return angle_diff_filtered


# STFT
def getSTFT(angle_diff_filtered, window_size, overlap, fs):
    nperseg = window_size
    noverlap = int(window_size * overlap)

    f, t, Zxx = stft(angle_diff_filtered,
                     fs,
                     nperseg=nperseg,
                     noverlap=noverlap)
    return f, t, Zxx
