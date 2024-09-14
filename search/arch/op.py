# -*- coding:utf8 -*_
import torch
import torch.nn as nn


# ===================================== FB-NET AND DARTS =======================================
class Identity(nn.Module):
    def __init__(self, c_in, c_out):
        super(Identity, self).__init__()
        self.net = nn.Conv2d(c_in, c_out, kernel_size=1)

        self.in_channels = c_in
        self.out_channels = c_out

    def forward(self, x):
        return self.net(x)


class FactorizedReduce(nn.Module):
    """
    通道变为c_out，大小
    减少特征图的宽度和高度，同时减少参数量和计算复杂度
    """

    def __init__(self, c_in, c_out):
        super(FactorizedReduce, self).__init__()
        self.conv1 = nn.Conv2d(c_in, c_out // 2, kernel_size=1, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv2d(c_in, c_out // 2, kernel_size=1, stride=2, padding=0, bias=False)
        self.final = nn.Sequential(
            nn.BatchNorm2d(c_out, affine=True),
            nn.ReLU(inplace=False)
        )

        self.in_channels = c_in
        self.out_channels = c_out

    def forward(self, x):
        concat = torch.cat([
            self.conv1(x),
            self.conv2(x[:, :, 1:, 1:])
        ], dim=1)
        return self.final(concat)


class ConvBNReLu(nn.Module):
    """
    structure: conv2d => bn => relu
    leaky relu可以缓解relu的死神经问题
    """

    def __init__(self, c_in, c_out, kernel, stride, padding=0, bias=False, lk=False):
        super(ConvBNReLu, self).__init__()
        self.net = nn.Sequential()
        if padding == -1:
            self.net.add_module("conv", nn.Conv2d(c_in, c_out, kernel_size=kernel, padding="same", bias=bias))
        else:
            self.net.add_module("conv",
                                nn.Conv2d(c_in, c_out, kernel_size=kernel, stride=stride, padding=padding, bias=bias))
        self.net.add_module("bn", nn.BatchNorm2d(c_out, affine=True))
        self.net.add_module("relu", nn.ReLU(inplace=False) if lk is False else nn.LeakyReLU(inplace=False))

        self.in_channels = c_in
        self.out_channels = c_out
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel

    def forward(self, x):
        return self.net(x)


class FinalFC(nn.Module):
    def __init__(self, in_feature, out_feature, use_bias=True):
        """

        :param in_feature: 这里实际上是最后一层卷积的output_channel，因为先AvgPool了，输出为1，所以最后进入fc的是 1 * channel展开的一维向量
        :param out_feature:
        :param use_bias:
        """
        super(FinalFC, self).__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features=in_feature, out_features=out_feature, bias=use_bias)
        )

        self.input_shape = in_feature
        self.output_shape = out_feature

    def forward(self, x):
        return self.net(x)

# ======================================= FB-NET AND DARTS END=====================================================


class ResnetBlock(nn.Module):
    """
    通道保持不变, 大小不变
    """
    def __init__(self, channel, block_nums=1):
        """
        残差块
        :param channel: 输入通道和输出通道保持一致
        :param block_nums: 残差块堆叠次数
        """
        super(ResnetBlock, self).__init__()
        self.net = nn.ModuleList()
        for _ in range(block_nums):
            self.net.append(nn.Sequential(
                ConvBNReLu(channel, channel // 2, 1, 1, 0, lk=True),
                ConvBNReLu(channel // 2, channel, 3, 1, 1, lk=True),
            ))

    def forward(self, x):
        # 向量各个元素相加: x + operator(x)
        for operator in self.net:
            x = operator(x) + x
        return x


class Darknet53(nn.Module):
    def __init__(self, num_class=10):
        """

        :param num_class: 分类器种类数
        """
        super(Darknet53, self).__init__()
        self.layer1 = nn.Sequential(
            ConvBNReLu(3, 32, 3, 1, 1, lk=True),
            ConvBNReLu(32, 64, 3, 2, 1, lk=True),
            ResnetBlock(64, 1)
        )
        self.layer2 = nn.Sequential(
            ConvBNReLu(64, 128, 3, 2, 1, lk=True),
            ResnetBlock(128, 2)
        )
        self.layer3 = nn.Sequential(
            ConvBNReLu(128, 256, 3, 2, 1, lk=True),
            ResnetBlock(256, 8)
        )
        self.layer4 = nn.Sequential(
            ConvBNReLu(256, 512, 3, 2, 1, lk=True),
            ResnetBlock(512, 8)
        )
        self.layer5 = nn.Sequential(
            ConvBNReLu(512, 1024, 3, 2, 1, lk=True),
            ResnetBlock(1024, 4)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_class)

        self.net = nn.Sequential(
            self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.pool, self.fc
        )

    def forward(self, x):
        return self.net(x)
