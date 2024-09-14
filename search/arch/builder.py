# -*- coding:utf8 -*-
import os
from collections import OrderedDict

from search.arch.op import *
from search.arch.resnet.resnet import *
from search.arch.layer import MixLayer, SingleLayer
from search.arch.arch_config import SINGLE, MIX
from search.arch.lut import mk_lut_name

# set environment "CHECK=1" to enable check mode
check = os.environ.get("CHECK", None)


class MetaBuilder(object):
    def __init__(self, basic_args):
        assert isinstance(basic_args, list) and all(map(lambda x: isinstance(x, dict), basic_args))
        self._convert_map = {
            # torch operator
            "avg_pool": nn.AvgPool2d,
            "max_pool": nn.MaxPool2d,

            # fb-net operator
            "conv": ConvBNReLu,
            "skip": Identity,
            "final": FinalFC,

            # resnet18
            "residual_layer": ResidualBlockLayer,
            "residual_fc": ResidualFC,
            "flexible_residual_layer": FlexibleResidualBlockerLayer,
            "dbr": DBR,

        }
        net_dict = OrderedDict()

        header = basic_args.pop(0)
        self.input_size = header["input_size"]

        for detail in basic_args:
            layer_type = detail["type"]
            name = detail["name"]
            if layer_type == SINGLE:
                operator = self._build_single_layer(detail)
            elif layer_type == MIX:
                operator = self._build_mix_layer(detail)
            else:
                raise Exception(f"Invalid layer_type<{layer_type}> at MetaBuilder!")
            net_dict[name] = operator
        self._super_net = nn.Sequential(net_dict)

    def _build_single_layer(self, detail):
        """
        Build SingleLayer from config;
        when set environment var "CHECK", use check mode for LUT_RECORDER
        :param detail:
        :return:
        """
        operator = self._build_operator(detail["blocks"][0])
        if check is None:
            return SingleLayer(operator)
        else:
            name = mk_lut_name(detail["blocks"][0], self.input_size)
            return SingleLayer(operator, name)

    def _build_mix_layer(self, detail):
        blocks = detail["blocks"]
        candidate_list = list()
        info_list = list()
        for info in blocks:
            if len(info) <= 0:
                continue
            candidate_list.append(self._build_operator(info))
            info_list.append(mk_lut_name(info, self.input_size))
        return MixLayer(candidate_list, None if check is None else info_list)

    def _build_operator(self, info):
        operator = self._convert_map.get(info[0], None)
        if operator is None:
            raise Exception(f"Invalid config key name: {info[0]}")

        if info[0] in self._convert_map.keys():
            ret = self._convert_map[info[0]](*info[1:])
        else:
            raise Exception(f"Invalid config type<{info[0]}>!")
        return ret

    @property
    def net(self):
        return self._super_net


if __name__ == '__main__':
    from arch_config import FB_NET_CIFAR10, RESNET18
    builder = MetaBuilder(RESNET18)
    net = builder.net
    net.eval()
