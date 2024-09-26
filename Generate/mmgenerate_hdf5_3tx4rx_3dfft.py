import os
import re
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import h5py
import multiprocessing
from MMGenerateFunc import MMGenerateFunctions, MMPlot
import numpy as np

# HDF5 文件路径
hdf5_path = r"/home/mambauser/MMCode/data/stft_3d.h5"

# 定义全局锁，防止多个进程同时写入
lock = multiprocessing.Lock()


def save_to_hdf5(hdf5_path, key, Zxx):
    # 使用锁，确保只有一个进程在写 HDF5 文件
    with lock:
        # 每个进程独立打开 HDF5 文件
        with h5py.File(hdf5_path, "a") as hdf5_file:  # 使用 "a" 模式，表示追加写入
            # 保存 Zxx 到 HDF5 文件中
            hdf5_file.create_dataset(key, data=Zxx)


def process_bin_file_hdf5(bin_path, wordIndex, fileIndex, personIndex, numTx, numRx):
    # 获取bin文件数据
    data_raw = MMGenerateFunctions.getRawData_multiTx(
        bin_path, numSamples, numChirps, numFrames, numTx, numRx
    )  # numRx, numFrames, numSamples

    # 获取bin文件数据
    data_raw = MMGenerateFunctions.getRawData_multiTx(
        bin_path, numSamples, numChirps, numFrames, numTx, numRx
    )  # numRx, numFrames, numSamples

    data_1dfft_dyn_all = np.zeros(data_raw[0].shape, dtype=complex)

    for tx in range(numTx):
        for rx in range(numRx):
            # 1D-FFT
            data_1dfft = MMGenerateFunctions.rangeFFT(
                data_raw[tx][rx], numFrames, numSamples
            )
            # 去除静态杂波
            data_1dfft_dyn_all[rx, :, :] = MMGenerateFunctions.removeStatic(
                data_1dfft, numFrames, numSamples
            )

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
        f, t, Zxx = MMGenerateFunctions.getSTFT(
            angle_diff_filtered, window_size, overlap, fs
        )

        # 存储STFT结果到HDF5文件
        num_subplots = 20
        t_intervals = np.linspace(t.min(), t.max(), num_subplots + 1)
        for i in range(num_subplots):
            start_time = t_intervals[i]
            end_time = t_intervals[i + 1]
            indices = (t >= start_time) & (t < end_time)
            key = f"{wordIndex}_{fileIndex*num_subplots+i}_{personIndex}_{tx}"
            save_to_hdf5(hdf5_path, key, Zxx[:, indices])


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

# 路径和文件夹列表
base_path = r"/home/mambauser/MMCode/data/rawData"

personIndex = 0


# 使用进程池并行处理文件
def main():
    bin_files = [f for f in os.listdir(base_path) if f.endswith(".bin")]
    total_files = len(bin_files)
    tasks = []

    with ProcessPoolExecutor(max_workers=8) as executor:
        # 创建进度条
        with tqdm(total=total_files) as pbar:
            for file_name in bin_files:
                # 使用正则表达式匹配文件名
                match = re.match(rf"(\d+)_(\d+)_Raw_?0\.bin", file_name)
                if match:
                    # 提取字母和数字
                    letter_part = int(match.group(1))
                    number_part = int(match.group(2))

                    bin_path = os.path.join(base_path, file_name)

                    letterIndex = letter_part
                    fileIndex = number_part

                    # 提交任务到进程池
                    future = executor.submit(
                        process_bin_file_hdf5,
                        bin_path,
                        letterIndex,
                        fileIndex,
                        personIndex,
                        numTx,
                        numRx,
                    )
                    tasks.append(future)

            # 等待所有任务完成并更新进度条
            for future in as_completed(tasks):
                try:
                    future.result()  # 获取结果并检查是否有异常
                except Exception as e:
                    print(f"任务失败: {e}")
                pbar.update(1)  # 更新进度条


if __name__ == "__main__":
    main()