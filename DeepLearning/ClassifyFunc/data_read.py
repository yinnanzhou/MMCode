import os
from PIL import Image


def get_data(folder_path,
             in_channels=1,
             wordIndex=list(range(50)),
             fileIndex=list(range(200)),
             personIndex=list(range(3)),
             txIndex=list(range(12))):
    samples = []
    labels = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            parts = filename.split('_')
            A = int(parts[0])
            B = int(parts[1])
            C = int(parts[2])
            D = int(parts[3].split('.')[0])
            if A in wordIndex and B in fileIndex and C in personIndex and D in txIndex:
                im_dir = os.path.join(folder_path, filename)
                image = Image.open(im_dir).convert('L' if in_channels ==
                                                   1 else 'RGB')

                # # 图片裁剪
                # width, height = image.size
                # left = 0  # 裁剪区域的左边界为图片宽度的一半
                # top = height * 49 / 50  # 裁剪区域的上边界为图片顶部
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
