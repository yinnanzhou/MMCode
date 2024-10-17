import torch
import torch.nn as nn


class BasicConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 relu=True,
                 bn=True,
                 bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=groups,
                              bias=bias)
        self.bn = nn.BatchNorm2d(
            out_channels, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class CustomResNet(nn.Module):
    def __init__(self,
                 in_channels,
                 num_classes,
                 weights=None,
                 model='resnet18'):
        super(CustomResNet, self).__init__()

        __resnet_opt = ['resnet18', 'resnet34', 'resnet101', 'resnet152']
        if model not in __resnet_opt:
            raise Exception("Please use the model from " +
                            (', ').join(__resnet_opt))

        self.res = torch.hub.load('pytorch/vision:v0.10.0',
                                  model,
                                  weights=weights)

        if in_channels != 3:
            self.res.conv1 = nn.Conv2d(in_channels,
                                       64,
                                       kernel_size=(7, 7),
                                       stride=(2, 2),
                                       padding=(3, 3),
                                       bias=False)

        if num_classes != 1000:
            self.res.fc = nn.Linear(in_features=512,
                                    out_features=num_classes,
                                    bias=True)

    def forward(self, x):
        out = self.res(x)

        return out


class MultiInResNet(nn.Module):
    def __init__(self,
                 num_inputs,
                 num_classes,
                 num_in_convs: list,
                 in_channels: list,
                 out_channels: list,
                 weights=None,
                 model='resnet18'):
        super(MultiInResNet, self).__init__()

        self.num_inputs = num_inputs
        self.in_channels = in_channels

        if len(num_in_convs) != num_inputs or len(
                in_channels) != num_inputs or len(out_channels) != num_inputs:
            raise Exception(
                'The number of inputs should be consistent with the length of num_in_convs/in_channels/out_channels.'
            )

        __resnet_opt = ['resnet18', 'resnet34', 'resnet101', 'resnet152']
        if model not in __resnet_opt:
            raise Exception("Please use the model from " +
                            (', ').join(__resnet_opt))

        self.res = torch.hub.load('pytorch/vision:v0.10.0',
                                  model,
                                  weights=weights)

        if num_classes != 1000:
            self.res.fc = nn.Linear(in_features=512,
                                    out_features=num_classes,
                                    bias=True)

        if num_inputs == 1:
            self.res.conv1 = nn.Conv2d(in_channels[0],
                                       64,
                                       kernel_size=(7, 7),
                                       stride=(2, 2),
                                       padding=(3, 3),
                                       bias=False)
        elif num_inputs > 1:
            self.res.conv1 = nn.Conv2d(sum(out_channels),
                                       64,
                                       kernel_size=(7, 7),
                                       stride=(2, 2),
                                       padding=(3, 3),
                                       bias=False)

            self.ins = [[] for _ in range(num_inputs)]

            for i in range(len(self.ins)):
                for _ in range(num_in_convs[i]):
                    self.ins[i].append(
                        BasicConv(in_channels[i],
                                  out_channels[i],
                                  kernel_size=3))
                self.ins[i] = nn.Sequential(*self.ins[i])
        else:
            raise ValueError("Invalide num_inputs.")

    def forward(self, *x):
        if len(x) != self.num_inputs:
            raise Exception('The number of inputs error.')

        if self.num_inputs == 1:
            out = self.res(x[0])
        elif self.num_inputs > 1:
            channels = [chn.shape[1] for chn in x]
            if channels != self.in_channels:
                raise Exception(
                    'Please re-order the tensors based on the number of channels, should be: '
                    + self.in_channels)

            _ins = [[] for _ in range(self.num_inputs)]

            for idx, _x in enumerate(x):
                _ins[idx] = self.ins[idx](_x)

            out1 = torch.cat(_ins, dim=1)
            out = self.res(out1)
        else:
            raise ValueError("Invalide num_inputs.")

        return out
