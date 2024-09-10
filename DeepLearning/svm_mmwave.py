import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from ClassifyFunc.visualization import visualize_predict

# 图像目录
image_dir = r'/home/mambauser/YinnanTest'

# 获取图像文件列表
image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

# 初始化数据和标签列表
data = []
labels = []

# 遍历文件，使用tqdm显示进度条
for file in tqdm(image_files, desc="Processing Images"):
    # 提取标签
    parts = file.split('_')
    A = int(parts[0])
    B = int(parts[1])
    C = int(parts[2])
    D = int(parts[3].split('.')[0])
    if D in [0,4]:
        labels.append(A)
        # 读取图像并转换为灰度
        image_path = os.path.join(image_dir, file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # 调整图像大小以便于处理（假设大小为64x64）
        image = cv2.resize(image, (256, 256))
        
        # 将图像展平并添加到数据列表
        data.append(image.flatten())

# 转换为numpy数组
data = np.array(data)
labels = np.array(labels)

# 划分训练和测试集
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# 创建和训练SVM模型
svm = SVC(kernel='linear')  # 你可以尝试其他核函数，比如 'rbf'
svm.fit(X_train, y_train)

# 进行预测
train_predictions = svm.predict(X_train)
test_predictions = svm.predict(X_test)

# 计算准确率
train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)


print(f"训练集准确率: {train_accuracy:.2f}")
print(f"测试集准确率: {test_accuracy:.2f}")
visualize_predict(y_test, test_predictions)