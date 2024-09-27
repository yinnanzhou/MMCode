import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from MMClassifyFunc.train import Trainer
from MMClassifyFunc.models import CustomResNet
from MMClassifyFunc.data_read import get_data
from MMClassifyFunc.visualization import visualize_results


from tqdm import tqdm
from MMClassifyFunc.data_preprocess import get_loader, get_loader_all
from MMClassifyFunc.visualization import visualize_results, visualize_predict

from sklearn.metrics import confusion_matrix


confusion_matrices = []

folder_path = r'/home/mambauser/MMCode/data/processed1D'
in_channels = 3

samples_train, labels_train = get_data(
    folder_path=folder_path,
    in_channels=in_channels,
    # wordIndex=list(range(5)),
    fileIndex=list(range(0,10))+list(range(12,30))+list(range(32,40)),
    # fileIndex=list(range(0,10))+list(range(30,40)),
    # fileIndex=list(range(0,40)),
    # personIndex=[1],
    txIndex=[0,1],
)

print("len(samples_train): {}".format(len(samples_train)))
print("len(set(labels_train)): {}".format(len(set(labels_train))))



samples_predict, labels_predict = get_data(
    folder_path=folder_path,
    in_channels=in_channels,
    # wordIndex=list(range(5)),
    fileIndex=[10,11,30,31],
    # personIndex=[1],
    # txIndex=[0,4,8],
)

print("len(samples_predict): {}".format(len(samples_predict)))
print("len(set(labels_train)): {}".format(len(set(labels_train))))


for t in range(30):

    trainloader, testloader = get_loader(
        samples=samples_train, labels=labels_train
    )

    # classifier
    classifier = CustomResNet(
        in_channels=in_channels,
        num_classes=len(set(labels_train)),
        weights=models.ResNet18_Weights.DEFAULT,
        model="resnet18",
    )

    # optimizers
    lr = 1e-3
    betas = (0.5, 0.99)
    optimizer = optim.Adam(classifier.parameters(), lr=lr, betas=betas)
    criterion = nn.CrossEntropyLoss()

    # train model
    NUM_INPUTS = 1
    epochs = 30

    trainer = Trainer(
        num_inputs=NUM_INPUTS,
        classifier=classifier,
        optimizer=optimizer,
        criterion=criterion,
        print_every=1,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        use_cuda=torch.cuda.is_available(),
        use_scheduler=False,
    )

    trainer.train(trainloader=trainloader, testloader=testloader, epochs=epochs)

    # visualize_results(trainer=trainer)

    # Create dataset and dataloader
    dataloader = get_loader_all(samples_predict, labels_predict)

    # classifier
    trainer.classifier.eval()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # classifier.to(device)
    trainer.classifier.to(device)

    # Prepare for evaluation
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Processing batches"):
            images, labels = images.to(device), labels.to(device)
            outputs = trainer.classifier(images)
            _, predicted = torch.max(outputs.data, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    confusion_matrices.append(cm)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sum_matrix = np.zeros((48, 48), dtype=int)

# 遍历每个矩阵并进行相加
for matrix in confusion_matrices:
    sum_matrix += matrix

plt.figure(figsize=(30, 20))

sns.heatmap(
    sum_matrix.astype("float") / sum_matrix.sum(axis=1)[:, np.newaxis],
    annot=True,
    fmt=".1f",
    cmap="Blues",
    cbar=False,
)
accuracy = np.trace(sum_matrix) / sum_matrix.sum()
plt.title("Predict accuracy: {:.2%}".format(accuracy))
plt.xlabel("Predicted label")
plt.ylabel("True label")

plt.show()
