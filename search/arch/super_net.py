# -*- coding:utf8 -*-

import copy
import torch
from torch import nn
from collections import OrderedDict

from search.arch.layer import SingleLayer, MixLayer
from search.arch.builder import MetaBuilder, SINGLE, MIX
from search.arch.arch_config import SUPER_NET_DICT

from utils import logger
from utils import lut_generator


class SuperNet(nn.Module):
    def __init__(self, config_name, input_size):
        super().__init__()
        arch_config = SUPER_NET_DICT.get(config_name, None)
        if arch_config is None:
            logger.NAS_LOGGER.error(f"Invalid arch config name, expected: {SUPER_NET_DICT.keys()}")
            exit(-1)
        builder = MetaBuilder(arch_config)
        self.net = builder.net
        self.config_name = config_name
        self.input_size = input_size

    def get_params(self):
        """
        返回的结构： List of [layer_type, layer.get_params()]
        :return:
        """
        information = list()
        assert isinstance(self.net, nn.Sequential)
        for name, layer in self.net.named_children():
            assert any([isinstance(layer, SingleLayer), isinstance(layer, MixLayer)])
            conditions = layer.get_params(self.input_size)
            layer_type = SINGLE if isinstance(layer, SingleLayer) else MIX
            information.append([layer_type, conditions])
        return information

    def forward(self, input_data):
        return self.net(input_data)

    def __str__(self):
        net_info = "\n".join([
                "Name:{}\tLayer:{}".format(name, layer) for name, layer in self.super_net.named_children()
            ])
        info = "ConfigName:{config}\tNetInfo:\n{net_info}".format(
            config=self.config_name, net_info=net_info
        )
        return info

    def generate_final_net(self):
        """
        提取生成最终网络的OrderedDict
        :return:
        """
        order_dict = OrderedDict()
        for name, layer in self.net.named_children():
            if isinstance(layer, SingleLayer):
                order_dict[name] = layer.operator
            elif isinstance(layer, MixLayer):
                index, _ = layer.chosen_index()
                order_dict[name] = layer.candidate_operators[index]
            else:
                logger.NAS_LOGGER.error("Invalid layer type at SuperNet. Generate final net failed!")
                raise Exception("Invalid layer type at SuperNet")
        return order_dict

    def show(self):
        """
        show the structure of super net
        :return: str
        """
        word_list = list()
        for name, layer in self.net.named_children():
            if isinstance(layer, SingleLayer):
                word_list.append("Name:{}\tLayer:{}".format(name, layer))
            elif isinstance(layer, MixLayer):
                word_list.append(
                    "Name:{}\tLayer:\n".format(name) + "\n".join([
                        f"Theta:{layer.path_theta[index]:.3f}{' 3'*4}Operator:{operator}" for index, operator in enumerate(layer.candidate_operators)
                    ])
                )
        return "\n".join(word_list)


class FinalNet(nn.Module):
    def __init__(self, ordered_dict, lat, mem):
        super().__init__()
        assert isinstance(ordered_dict, OrderedDict)
        self.net = nn.Sequential(ordered_dict)
        self.latency = lat
        self.memory = mem

    def forward(self, input_data):
        return self.net(input_data)

    def __str__(self):
        net_info = "\n".join([
            "Name:{}\tLayer:{}".format(name, layer) for name, layer in self.net.named_children()
        ])
        return net_info


