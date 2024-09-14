# -*- coding:utf8 -*-
"""
Stores all currently available candidate network structures,
with the following infrastructure reference (can be seen in conjunction with MetaBuilder):

basic_args = [
            {
                "input_size": 224,
                "classes": 1000,
            },
            {
                "type": single or mix,
                "blocks": [
                    ["conv", channel_in, channel_out, kernel, stride, padding, use_bias]
                    ["max_pool", kernel]
                    ["avg_pool", kernel]
                    ["skip", channel_in, channel_out],
                    ["final", cls_num, use_bias]
                ],
                "name": layer or block name
            },

            {...}
        ]


input_shape, output_shape should be considered with Data Transform
"""

SINGLE = 1
MIX = 2


# 确保名字都是小写
SUPER_NET_DICT = dict()


# CIFAR10输入太小了，32*32很容易丢失
FB_NET_CIFAR10 = [
    {
        "input_size": 32,
        "classes": 10
    },

    # 第一层，一般是固定的卷积处理
    {
        "name": "Stage0::ConvK3",
        "type": SINGLE,
        "blocks": [
            ["conv", 3, 16, 3, 2, 0, False]
        ]
    },
    {
        "name": "Stage1::skip",
        "type": SINGLE,
        "blocks": [
            ["skip", 16, 16],
        ]
    },
    # 这里可以考虑保留skip，然后修改为same
    {
        "name": "Stage2::MixConv",
        "type": MIX,
        "blocks": [
            ["conv", 16, 24, 3, 2, -1, False],
            ["conv", 16, 24, 3, 2, -1, True],
            ["conv", 16, 24, 5, 2, -1, False],
            ["conv", 16, 24, 5, 2, -1, True],
            ["skip", 16, 24]
        ]
    },
    {
        "name": "Stage3::MixConv",
        "type": MIX,
        "blocks": [
            ["conv", 24, 32, 3, 2, 0, False],
            ["conv", 24, 32, 3, 2, 0, True],
            ["conv", 24, 32, 5, 2, 1, False],
            ["conv", 24, 32, 5, 2, 1, True],
        ]
    },
    # 这一层，将padding改为了same，不然太小了，下一层卷积会挂掉
    {
        "name": "Stage4::MixConv",
        "type": MIX,
        "blocks": [
            ["conv", 32, 64, 3, 2, 0, False],
            ["conv", 32, 64, 3, 2, 0, True],
            ["conv", 32, 64, 5, 2, 1, False],
            ["conv", 32, 64, 5, 2, 1, True],
        ]
    },
    # 这一层，将padding改为了same，不然太小了，下一层卷积会挂掉
    {
        "name": "Stage5::MixConv",
        "type": MIX,
        "blocks": [
            ["conv", 64, 184, 5, 2, -1, False],
            ["conv", 64, 184, 5, 2, -1, True],
        ]
    },

    {
        "name": "Stage6::MixConv",
        "type": MIX,
        "blocks": [
            ["max_pool", 3],
            ["avg_pool", 3],
        ]
    },
    # CIFAR-10就是10种数据
    {
        "name": "Stage7::Single",
        "type": SINGLE,
        "blocks": [
            ["final", 184, 1000, True]
        ]
    }
]

FB_NET_IMAGENET = [
    {
        "input_size": 224,
        "classes": 100
    },

    # 第一层，一般是固定的卷积处理
    {
        "name": "Stage0::ConvK3",
        "type": SINGLE,
        "blocks": [
            ["conv", 3, 16, 3, 2, 0, False]
        ]
    },
    {
        "name": "Stage1::skip",
        "type": SINGLE,
        "blocks": [
            ["skip", 16, 16],
        ]
    },
    # 这里可以考虑保留skip，然后修改为same
    {
        "name": "Stage2::MixConv",
        "type": MIX,
        "blocks": [
            ["conv", 16, 24, 3, 2, -1, False],
            ["conv", 16, 24, 3, 2, -1, True],
            ["conv", 16, 24, 5, 2, -1, False],
            ["conv", 16, 24, 5, 2, -1, True],
            ["skip", 16, 24]
        ]
    },
    {
        "name": "Stage3::MixConv",
        "type": MIX,
        "blocks": [
            ["conv", 24, 32, 3, 2, 0, False],
            ["conv", 24, 32, 3, 2, 0, True],
            ["conv", 24, 32, 5, 2, 1, False],
            ["conv", 24, 32, 5, 2, 1, True],
        ]
    },
    # 这一层，将padding改为了same，不然太小了，下一层卷积会挂掉
    {
        "name": "Stage4::MixConv",
        "type": MIX,
        "blocks": [
            ["conv", 32, 64, 3, 2, 0, False],
            ["conv", 32, 64, 3, 2, 0, True],
            ["conv", 32, 64, 5, 2, 1, False],
            ["conv", 32, 64, 5, 2, 1, True],
        ]
    },
    # 这一层，将padding改为了same，不然太小了，下一层卷积会挂掉
    {
        "name": "Stage5::MixConv",
        "type": MIX,
        "blocks": [
            ["conv", 64, 184, 5, 2, -1, False],
            ["conv", 64, 184, 5, 2, -1, True],
        ]
    },

    {
        "name": "Stage6::MixConv",
        "type": MIX,
        "blocks": [
            ["max_pool", 3],
            ["avg_pool", 3],
        ]
    },
    # CIFAR-10就是10种数据
    {
        "name": "Stage7::Single",
        "type": SINGLE,
        "blocks": [
            ["final", 184, 1000, True]
        ]
    }
]


SUPER_NET_DICT["fbnet"] = FB_NET_CIFAR10
SUPER_NET_DICT["fbnet_img"] = FB_NET_IMAGENET


# ResNet18
RESNET18 = [
    {
        "input_size": 32,
        "classes": 10
    },

    {
        "name": "Stage0::Conv0",
        "type": SINGLE,
        "blocks": [
            ["conv", 3, 64, 3, 1, 1, False]
        ]
    },

    {
        "name": "Stage1::Pool",
        "type": MIX,
        "blocks": [
            ["max_pool", 3, 2, 1],
            ["avg_pool", 3, 2, 1]
        ]
    },

    {
        "name": "Stage3::Layer1",
        "type": MIX,
        "blocks": [
            # False: 矩阵加, True: 矩阵拼接, 目标：搜索不同架构和不同超参数
            ["residual_layer", 64, 64, 2, 1, False],
            ["residual_layer", 64, 64, 3, 1, False],
            ["residual_layer", 64, 64, 4, 1, False],
            ["residual_layer", 64, 64, 2, 1, True],
            ["residual_layer", 64, 64, 3, 1, True],
            ["residual_layer", 64, 64, 4, 1, True],
            # 使用单卷积代替复杂的残差块, 目标：搜索不同架构
            ["conv", 64, 64, 3, 1, 1, False],
            ["dbr", 64, 64, 3, 1],
            # 在可以接受的地方直接跳过, 目标：搜索不同架构
            ["skip", 64, 64],
            # 修改卷积核大小1, 5, 7, 目标：搜索超参数
            ["flexible_residual_layer", 64, 64, 2, 1, 1, "normal"],
            ["flexible_residual_layer", 64, 64, 2, 5, 1, "normal"],
            ["flexible_residual_layer", 64, 64, 2, 7, 1, "normal"],
            # 使用深度卷积替换传统卷积, 目标：搜索不同架构
            ["flexible_residual_layer", 64, 64, 2, 3, 1, "dp"]

        ]
    },

    {
        "name": "Stage4::Layer2",
        "type": MIX,
        "blocks": [
            ["residual_layer", 64, 128, 2, 2, False],
            ["residual_layer", 64, 128, 3, 2, False],
            ["residual_layer", 64, 128, 4, 2, False],
            ["residual_layer", 64, 128, 2, 2, True],
            ["residual_layer", 64, 128, 3, 2, True],
            ["residual_layer", 64, 128, 4, 2, True],
            ["conv", 64, 128, 3, 2, 1, False],
            ["dbr", 64, 128, 3, 2],
            ["flexible_residual_layer", 64, 128, 2, 1, 2, "normal"],
            ["flexible_residual_layer", 64, 128, 2, 5, 2, "normal"],
            ["flexible_residual_layer", 64, 128, 2, 7, 2, "normal"],
            ["flexible_residual_layer", 64, 128, 2, 3, 2, "dp"]
        ]
    },

    {
        "name": "Stage5::Layer3",
        "type": MIX,
        "blocks": [
            ["residual_layer", 128, 256, 2, 2, False],
            ["residual_layer", 128, 256, 3, 2, False],
            ["residual_layer", 128, 256, 4, 2, False],
            ["residual_layer", 128, 256, 2, 2, True],
            ["residual_layer", 128, 256, 3, 2, True],
            ["residual_layer", 128, 256, 4, 2, True],
            ["conv", 128, 256, 3, 2, 1, False],
            ["dbr", 128, 256, 3, 2],
            ["flexible_residual_layer", 128, 256, 2, 1, 2, "normal"],
            ["flexible_residual_layer", 128, 256, 2, 5, 2, "normal"],
            ["flexible_residual_layer", 128, 256, 2, 7, 2, "normal"],
            ["flexible_residual_layer", 128, 256, 2, 3, 2, "dp"]
        ]
    },

    {
        "name": "Stage6::Layer4",
        "type": MIX,
        "blocks": [
            ["residual_layer", 256, 512, 2, 2, False],
            ["residual_layer", 256, 512, 3, 2, False],
            ["residual_layer", 256, 512, 4, 2, False],
            ["residual_layer", 256, 512, 2, 2, True],
            ["residual_layer", 256, 512, 3, 2, True],
            ["residual_layer", 256, 512, 4, 2, True],
            ["conv", 256, 512, 3, 2, 1, False],
            ["dbr", 256, 512, 3, 2],
            ["flexible_residual_layer", 256, 512, 2, 1, 2, "normal"],
            ["flexible_residual_layer", 256, 512, 2, 5, 2, "normal"],
            ["flexible_residual_layer", 256, 512, 2, 7, 2, "normal"],
            ["flexible_residual_layer", 256, 512, 2, 3, 2, "dp"]
        ]
    },

    {
        "name": "Stage7::Pool",
        "type": MIX,
        "blocks": [
            ["max_pool", 2],
            ["avg_pool", 2]
        ]
    },

    {
        "name": "Stage8::FC",
        "type": SINGLE,
        "blocks": [
            ["residual_fc", 512, 10]
        ]
    }
]

RESNET18_100 = [
    {
        "input_size": 32,
        "classes": 100
    },

    {
        "name": "Stage0::Conv0",
        "type": SINGLE,
        "blocks": [
            ["conv", 3, 64, 3, 1, 1, False]
        ]
    },

    {
        "name": "Stage1::Pool",
        "type": MIX,
        "blocks": [
            ["max_pool", 3, 2, 1],
            ["avg_pool", 3, 2, 1]
        ]
    },

    {
        "name": "Stage3::Layer1",
        "type": MIX,
        "blocks": [
            # False: 矩阵加, True: 矩阵拼接, 目标：搜索不同架构和不同超参数
            ["residual_layer", 64, 64, 2, 1, False],
            ["residual_layer", 64, 64, 3, 1, False],
            ["residual_layer", 64, 64, 4, 1, False],
            ["residual_layer", 64, 64, 2, 1, True],
            ["residual_layer", 64, 64, 3, 1, True],
            ["residual_layer", 64, 64, 4, 1, True],
            # 使用单卷积代替复杂的残差块, 目标：搜索不同架构
            ["conv", 64, 64, 3, 1, 1, False],
            ["dbr", 64, 64, 3, 1],
            # 在可以接受的地方直接跳过, 目标：搜索不同架构
            ["skip", 64, 64],
            # 修改卷积核大小1, 5, 7, 目标：搜索超参数
            ["flexible_residual_layer", 64, 64, 2, 1, 1, "normal"],
            ["flexible_residual_layer", 64, 64, 2, 5, 1, "normal"],
            ["flexible_residual_layer", 64, 64, 2, 7, 1, "normal"],
            # 使用深度卷积替换传统卷积, 目标：搜索不同架构
            ["flexible_residual_layer", 64, 64, 2, 3, 1, "dp"]

        ]
    },

    {
        "name": "Stage4::Layer2",
        "type": MIX,
        "blocks": [
            ["residual_layer", 64, 128, 2, 2, False],
            ["residual_layer", 64, 128, 3, 2, False],
            ["residual_layer", 64, 128, 4, 2, False],
            ["residual_layer", 64, 128, 2, 2, True],
            ["residual_layer", 64, 128, 3, 2, True],
            ["residual_layer", 64, 128, 4, 2, True],
            ["conv", 64, 128, 3, 2, 1, False],
            ["dbr", 64, 128, 3, 2],
            ["flexible_residual_layer", 64, 128, 2, 1, 2, "normal"],
            ["flexible_residual_layer", 64, 128, 2, 5, 2, "normal"],
            ["flexible_residual_layer", 64, 128, 2, 7, 2, "normal"],
            ["flexible_residual_layer", 64, 128, 2, 3, 2, "dp"]
        ]
    },

    {
        "name": "Stage5::Layer3",
        "type": MIX,
        "blocks": [
            ["residual_layer", 128, 256, 2, 2, False],
            ["residual_layer", 128, 256, 3, 2, False],
            ["residual_layer", 128, 256, 4, 2, False],
            ["residual_layer", 128, 256, 2, 2, True],
            ["residual_layer", 128, 256, 3, 2, True],
            ["residual_layer", 128, 256, 4, 2, True],
            ["conv", 128, 256, 3, 2, 1, False],
            ["dbr", 128, 256, 3, 2],
            ["flexible_residual_layer", 128, 256, 2, 1, 2, "normal"],
            ["flexible_residual_layer", 128, 256, 2, 5, 2, "normal"],
            ["flexible_residual_layer", 128, 256, 2, 7, 2, "normal"],
            ["flexible_residual_layer", 128, 256, 2, 3, 2, "dp"]
        ]
    },

    {
        "name": "Stage6::Layer4",
        "type": MIX,
        "blocks": [
            ["residual_layer", 256, 512, 2, 2, False],
            ["residual_layer", 256, 512, 3, 2, False],
            ["residual_layer", 256, 512, 4, 2, False],
            ["residual_layer", 256, 512, 2, 2, True],
            ["residual_layer", 256, 512, 3, 2, True],
            ["residual_layer", 256, 512, 4, 2, True],
            ["conv", 256, 512, 3, 2, 1, False],
            ["dbr", 256, 512, 3, 2],
            ["flexible_residual_layer", 256, 512, 2, 1, 2, "normal"],
            ["flexible_residual_layer", 256, 512, 2, 5, 2, "normal"],
            ["flexible_residual_layer", 256, 512, 2, 7, 2, "normal"],
            ["flexible_residual_layer", 256, 512, 2, 3, 2, "dp"]
        ]
    },

    {
        "name": "Stage7::Pool",
        "type": MIX,
        "blocks": [
            ["max_pool", 2],
            ["avg_pool", 2]
        ]
    },

    {
        "name": "Stage8::FC",
        "type": SINGLE,
        "blocks": [
            ["residual_fc", 512, 100]
        ]
    }
]

SUPER_NET_DICT["resnet18"] = RESNET18
SUPER_NET_DICT["resnet18_100"] = RESNET18_100
