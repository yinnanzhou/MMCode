import os
from PIL import Image
import h5py
import numpy as np


def get_data(
    folder_path,
    in_channels=1,
    wordIndex=list(range(50)),
    fileIndex=list(range(200)),
    personIndex=list(range(50)),
    txIndex=list(range(200)),
):
    samples = []
    labels = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            parts = filename.split("_")
            A = int(parts[0])
            B = int(parts[1])
            C = int(parts[2])
            D = int(parts[3].split(".")[0])
            if A in wordIndex and B in fileIndex and C in personIndex and D in txIndex:
                im_dir = os.path.join(folder_path, filename)
                image = Image.open(im_dir).convert("L" if in_channels == 1 else "RGB")

                # # 图片裁剪
                # width, height = image.size
                # left = 0  # 裁剪区域的左边界为图片宽度的一半
                # top = height * 39 / 40  # 裁剪区域的上边界为图片顶部
                # right = width  # 裁剪区域的右边界为图片宽度
                # bottom = height  # 裁剪区域的下边界为图片高度
                # image = image.crop((left, top, right, bottom))

                samples.append(image)
                labels.append(A)

    # 使用 set 获取列表中的唯一值，并使用 sorted 对唯一值进行排序
    unique_labels = sorted(set(labels))
    # 创建映射字典
    label_map = {label: index for index, label in enumerate(unique_labels)}

    # 使用映射字典将原始列表映射为新的列表
    mapped_labels = [label_map[label] for label in labels]

    return samples, mapped_labels


def get_data_hdf5(
    h5_file_path,
    in_channels=1,
    wordIndex=list(range(50)),
    fileIndex=list(range(200)),
    personIndex=list(range(3)),
    txIndex=list(range(12)),
):
    samples = []
    labels = []

    # 打开HDF5文件
    with h5py.File(h5_file_path, "r") as f:
        # 遍历HDF5文件的所有key，key的格式为"wordIndex_fileIndex_personIndex_txIndex"
        for key in f.keys():
            parts = key.split("_")
            A = int(parts[0])  # wordIndex
            B = int(parts[1])  # fileIndex
            C = int(parts[2])  # personIndex
            D = int(parts[3])  # txIndex, 去掉' tx'前缀

            # 如果满足指定的索引范围条件
            if A in wordIndex and B in fileIndex and C in personIndex and D in txIndex:
                # 读取Zxx数据
                if f[key][:].shape[1] == 111:
                    Zxx = f[key][:, :-1]
                else:
                    Zxx = f[key][:, :]

                # 对Zxx数据进行处理：10 * np.log10(np.abs(Zxx))
                Zxx_processed = 10 * np.log10(np.abs(Zxx))

                # 如果是单通道数据，则需要将Zxx数据维度扩展
                if in_channels == 1:
                    Zxx_processed = np.expand_dims(Zxx_processed, axis=0)

                # 将处理后的数据加入samples
                samples.append(Zxx_processed)

                # 将wordIndex作为标签
                labels.append(A)

    # 生成标签映射
    unique_labels = sorted(set(labels))
    label_map = {label: index for index, label in enumerate(unique_labels)}
    mapped_labels = [label_map[label] for label in labels]

    return samples, mapped_labels


def get_data_hdf5_nolog(
    h5_file_path,
    in_channels=1,
    wordIndex=list(range(50)),
    fileIndex=list(range(200)),
    personIndex=list(range(3)),
    txIndex=list(range(12)),
):
    samples = []
    labels = []

    # 打开HDF5文件
    with h5py.File(h5_file_path, "r") as f:
        # 遍历HDF5文件的所有key，key的格式为"wordIndex_fileIndex_personIndex_txIndex"
        for key in f.keys():
            parts = key.split("_")
            A = int(parts[0])  # wordIndex
            B = int(parts[1])  # fileIndex
            C = int(parts[2])  # personIndex
            D = int(parts[3])  # txIndex, 去掉' tx'前缀

            # 如果满足指定的索引范围条件
            if A in wordIndex and B in fileIndex and C in personIndex and D in txIndex:
                # 读取Zxx数据
                if f[key][:].shape[1] == 111:
                    Zxx = f[key][:, :-1]
                else:
                    Zxx = f[key][:, :]

                # 对Zxx数据进行处理：10 * np.log10(np.abs(Zxx))
                Zxx_processed = np.abs(Zxx)

                # 如果是单通道数据，则需要将Zxx数据维度扩展
                if in_channels == 1:
                    Zxx_processed = np.expand_dims(Zxx_processed, axis=0)

                # 将处理后的数据加入samples
                samples.append(Zxx_processed)

                # 将wordIndex作为标签
                labels.append(A)

    # 生成标签映射
    unique_labels = sorted(set(labels))
    label_map = {label: index for index, label in enumerate(unique_labels)}
    mapped_labels = [label_map[label] for label in labels]

    return samples, mapped_labels
