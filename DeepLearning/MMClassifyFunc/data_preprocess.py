import glob
import numpy as np
from matplotlib import image
from torchvision import transforms
from PIL import Image
import torch
from torch.utils.data import DataLoader, TensorDataset
import random
from collections import Counter


def get_loader(samples, labels):
    temp = list(zip(samples, labels))
    random.shuffle(temp)
    samples_loader, labels_loader = zip(*temp)

    samples_loader = torch.stack(samples_loader, dim=0)
    labels_loader = torch.Tensor(labels_loader).long()

    # split train test data
    SPLIT_RATE = 0.8
    train_num = int(labels_loader.size(0) * SPLIT_RATE)
    train_X, train_Y = samples_loader[0:train_num], labels_loader[0:train_num]
    test_X, test_Y = samples_loader[train_num:], labels_loader[train_num:]

    # print labels
    print("train_Y")
    print(
        ", ".join(
            [
                f"Number {num}: {count}"
                for num, count in Counter(train_Y.tolist()).items()
            ]
        )
    )
    print("test_Y")
    print(
        ", ".join(
            [
                f"Number {num}: {count}"
                for num, count in Counter(test_Y.tolist()).items()
            ]
        )
    )

    trainset = TensorDataset(train_X, train_Y)
    testset = TensorDataset(test_X, test_Y)
    trainloader = DataLoader(trainset, batch_size=256, shuffle=True)
    testloader = DataLoader(testset, batch_size=256, shuffle=True)
    return trainloader, testloader


def get_loader_all(samples, labels):
    samples_loader = torch.stack(samples, dim=0)
    labels_loader = torch.Tensor(labels).long()

    dataset = TensorDataset(samples_loader, labels_loader)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    return dataloader
