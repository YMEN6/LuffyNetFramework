# -*- coding:utf8 -*-
"""
ResNet18 and ResNet50 as baseline
"""

import torch

from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms


class ResidualBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c_out)
        )
        # 将输入进行压缩后往下传，此时类似down sample
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(c_out)
            )
        # 残差连接，此时相当于经过了一个nn.Identity()
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        relu = nn.ReLU()
        out = self.left(x)
        out += self.shortcut(x)
        out = relu(out)
        return out


class ResidualBlockConcat(nn.Module):
    # 看看怎么处理，个人感觉参考ELN，先压缩再拼接
    """use concat instead of '+' for shortcut"""
    def __init__(self, c_in, c_out, stride=1):
        super(ResidualBlockConcat, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(c_in, c_out // 2, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(c_out // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out // 2, c_out // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c_out // 2)
        )
        # 将输入进行压缩后往下传，此时类似down sample
        self.shortcut = nn.Sequential(
            nn.Conv2d(c_in, c_out // 2, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(c_out // 2)
        )

    def forward(self, x):
        relu = nn.ReLU()
        left = self.left(x)
        right = self.shortcut(x)
        out = torch.cat([left, right], 1)
        out = relu(out)
        return out


class ResidualBlockLayer(nn.Module):
    """Replace the function::make_layer"""
    def __init__(self, in_channels, out_channels, num_blocks, stride, concat=False):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.concat = concat
        self.blocks = num_blocks

        block = ResidualBlockConcat if concat is True else ResidualBlock
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        last_channel = in_channels
        for stride in strides:
            layers.append(block(last_channel, out_channels, stride))
            last_channel = out_channels
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResidualFC(nn.Module):
    def __init__(self, in_channel, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=in_channel, out_features=num_classes)
        )

        self.input_shape = in_channel
        self.output_shape = num_classes

    def forward(self, x):
        return self.net(x)


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        self.flatten = nn.Flatten()

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)

        # my
        out = self.pool1(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool2(out)
        out = self.flatten(out)

        out = self.fc(out)
        return out


class DepthSepConv(nn.Module):
    """Depthwise Separable Conv"""

    def __init__(self, c_in, c_out, kernel_size, stride, padding=None):
        super().__init__()
        # auto padding
        if padding is None:
            padding = kernel_size // 2 if isinstance(kernel_size, int) else [x // 2 for x in kernel_size]
        self.depth_wise = nn.Conv2d(c_in, c_in, kernel_size=kernel_size, stride=stride, padding=padding, bias=False,
                                    groups=c_in)
        self.point_wise = nn.Conv2d(c_in, c_out, kernel_size=1, stride=1, padding=0, bias=True)
        self.net = nn.Sequential(
            self.depth_wise,
            self.point_wise
        )

    def forward(self, x):
        return self.net(x)


class DBR(nn.Module):
    """Use DepthSeparableConv to instead of normal conv in CBR"""
    def __init__(self, in_channel, out_channel, kernel, stride):
        super().__init__()
        self.net = nn.Sequential(
            DepthSepConv(in_channel, out_channel, kernel_size=kernel, stride=stride),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        self.in_channels = in_channel
        self.out_channels = out_channel
        self.stride = stride
        self.kernel_size = kernel

    def forward(self, x):
        return self.net(x)


class FlexibleResidualBlock(nn.Module):
    """Flexible Residual Block, to meet the requirement of changing about DepthWiseSeparableConv and kernel size"""
    def __init__(self, c_in, c_out, kernel=3, stride=1, conv=nn.Conv2d):
        super().__init__()
        padding = kernel // 2
        self.left = nn.Sequential(
            conv(c_in, c_out, kernel_size=kernel, stride=stride, padding=padding),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            conv(c_out, c_out, kernel_size=kernel, stride=1, padding=padding),
            nn.BatchNorm2d(c_out)
        )
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(
                conv(c_in, c_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(c_out)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        relu = nn.ReLU()
        out = self.left(x)
        out += self.shortcut(x)
        return relu(out)


class FlexibleResidualBlockerLayer(nn.Module):
    CON = {
        "normal": nn.Conv2d,
        "dp": DepthSepConv
    }

    def __init__(self, in_channels, out_channels, num_blocks, kernel, stride, conv="normal"):
        super().__init__()
        assert conv in FlexibleResidualBlockerLayer.CON.keys()
        self.kernel = kernel
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = conv
        self.blocks = num_blocks

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        last_channel = in_channels
        for stride in strides:
            layers.append(FlexibleResidualBlock(
                last_channel, out_channels, kernel, stride, FlexibleResidualBlockerLayer.CON.get(conv)
            ))
            last_channel = out_channels
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FlexibleResNet(nn.Module):
    def __init__(self, kernel, channels, num_classes=10):
        super().__init__()
        conv = "dp"
        self.first_layer = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=kernel, stride=1, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.layer1 = FlexibleResidualBlockerLayer(channels, channels, 2, kernel, 1, conv)
        self.layer2 = FlexibleResidualBlockerLayer(channels, 2 * channels, 2, kernel, 2, conv)
        self.layer3 = FlexibleResidualBlockerLayer(channels * 2, channels * 4, 2, kernel, 2, conv)
        self.layer4 = FlexibleResidualBlockerLayer(channels * 4, channels * 8, 2, kernel, 2, conv)
        self.fc = nn.Linear(channels * 8, num_classes, bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        self.flatten = nn.Flatten()

    def forward(self, x):
        out = self.first_layer(x)
        out = self.pool1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool2(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out


# 该模块暂时没放进去测试，感觉测太多了
class BottleneckResidualBlock(nn.Module):
    """Bottleneck Residual Block: input -> 1*1 -> 3*3 -> 1*1"""
    def __init__(self, c_in, c_out, kernel=3, stride=1, conv=nn.Conv2d):
        super().__init__()
        padding = kernel // 2
        middle_channel = c_in
        self.left = nn.Sequential(
            conv(c_in, middle_channel, kernel_size=1, stride=1),
            nn.BatchNorm2d(middle_channel),
            nn.ReLU(inplace=True),

            conv(middle_channel, middle_channel, kernel_size=kernel, stride=stride, padding=padding),
            nn.BatchNorm2d(middle_channel),
            nn.ReLU(inplace=True),

            conv(middle_channel, c_out, kernel_size=1, stride=1),
            nn.BatchNorm2d(c_out),
        )
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(
                conv(c_in, c_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(c_out)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        relu = nn.ReLU()
        out = self.left(x)
        out += self.shortcut(x)
        return relu(out)


if __name__ == '__main__':
    torch.cuda.empty_cache()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # build model
    # net = ResNet(ResidualBlockConcat)
    net = FlexibleResNet(3, 8)
    net = net.to(device)
    # build dataset
    tf = transforms.Compose([
        transforms.Resize(96),
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.49139968, 0.48215827, 0.44653124], std=[0.24703233, 0.24348505, 0.26158768])
    ])
    dataset = CIFAR10(root="E:\\", train=True, transform=tf)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True, pin_memory=True)
    # just have a try
    criterion = nn.CrossEntropyLoss()
    for data, label in data_loader:
        datas = data.to(device)
        labels = label.to(device)
        outputs = net(datas)
        loss = criterion(outputs, labels)
        print(f"Loss: {loss.item():.4f}")
