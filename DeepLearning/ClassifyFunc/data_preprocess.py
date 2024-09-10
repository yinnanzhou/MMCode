import glob
import numpy as np
from matplotlib import image
from torchvision import transforms
from PIL import Image
import torch
from torch.utils.data import DataLoader, TensorDataset
import random
from collections import Counter

preprocess = transforms.Compose(
    [transforms.Resize((256, 256)),
     transforms.ToTensor()])


def get_loader(samples, labels):
    for i in range(len(samples)):
        samples[i] = preprocess(samples[i])

    temp = list(zip(samples, labels))
    random.shuffle(temp)
    samples, labels = zip(*temp)

    samples = torch.stack(samples, dim=0)
    labels = torch.Tensor(labels).long()

    # split train test data
    SPLIT_RATE = 0.8
    train_num = int(labels.size(0) * SPLIT_RATE)
    train_X, train_Y = samples[0:train_num], labels[0:train_num]
    test_X, test_Y = samples[train_num:], labels[train_num:]

    # print labels
    print("train_Y")
    print(", ".join([
        f"Number {num}: {count}"
        for num, count in Counter(train_Y.tolist()).items()
    ]))
    print("test_Y")
    print(", ".join([
        f"Number {num}: {count}"
        for num, count in Counter(test_Y.tolist()).items()
    ]))

    trainset = TensorDataset(train_X, train_Y)
    testset = TensorDataset(test_X, test_Y)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32, shuffle=True)
    return trainloader, testloader


def get_loader_all(samples, labels):
    for i in range(len(samples)):
        samples[i] = preprocess(samples[i])

    samples = torch.stack(samples, dim=0)
    labels = torch.Tensor(labels).long()

    dataset = TensorDataset(samples, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    return dataloader
