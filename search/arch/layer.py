# -*- coding:utf8 -*-
import os
import torch
import numpy as np

from search.arch.lut import LutItem
from search.arch.op import *
from search.arch.resnet.resnet import *
from utils import lut_generator


def _build_condition_dict(operator):
    condition_dict = dict()
    if isinstance(operator, ConvBNReLu):
        name = "conv"
        tp = LutItem.CONV
        condition_dict["input_channel"] = operator.in_channels
        condition_dict["output_channel"] = operator.out_channels
        condition_dict["stride"] = operator.stride
        condition_dict["padding"] = operator.padding
        condition_dict["kernel"] = operator.kernel_size

    elif isinstance(operator, nn.AvgPool2d):
        name = "avg_pool"
        tp = LutItem.AVG_POOL
        condition_dict["kernel"] = operator.kernel_size

    elif isinstance(operator, nn.MaxPool2d):
        name = "max_pool"
        tp = LutItem.MAX_POOL
        condition_dict["kernel"] = operator.kernel_size

    elif isinstance(operator, Identity):
        name = "skip"
        tp = LutItem.SKIP
        condition_dict["input_channel"] = operator.in_channels
        condition_dict["output_channel"] = operator.out_channels

    elif isinstance(operator, FinalFC):
        name = "final"
        tp = LutItem.FINAL
        condition_dict["input_shape"] = operator.input_shape
        condition_dict["output_shape"] = operator.output_shape

    elif isinstance(operator, ResidualBlockLayer):
        name = "residual_layer"
        tp = LutItem.RESIDUAL_LAYER
        condition_dict["input_channel"] = operator.in_channels
        condition_dict["output_channel"] = operator.out_channels
        condition_dict["blocks"] = operator.blocks
        condition_dict["stride"] = operator.stride
        condition_dict["concat"] = operator.concat

    elif isinstance(operator, ResidualFC):
        name = "residual_fc"
        tp = LutItem.RESIDUAL_FC
        condition_dict["input_shape"] = operator.input_shape
        condition_dict["output_shape"] = operator.output_shape

    elif isinstance(operator, FlexibleResidualBlockerLayer):
        name = "flexible_residual_layer"
        tp = LutItem.FLEXIBLE_RESIDUAL_LAYER
        condition_dict["input_channel"] = operator.in_channels
        condition_dict["output_channel"] = operator.out_channels
        condition_dict["blocks"] = operator.blocks
        condition_dict["stride"] = operator.stride
        condition_dict["kernel"] = operator.kernel
        condition_dict["conv"] = operator.conv

    elif isinstance(operator, DBR):
        name = "dbr"
        tp = LutItem.DBR
        condition_dict["input_channel"] = operator.in_channels
        condition_dict["output_channel"] = operator.out_channels
        condition_dict["stride"] = operator.stride
        condition_dict["kernel"] = operator.kernel_size

    else:
        raise Exception(f"Unexpected operator type<{type(operator)}>")

    return name, tp, condition_dict


class MixLayer(nn.Module):
    def __init__(self, candidates, info_list=None):
        """
        并列层的输入、输出 size是应该保持一致的
        并行层，需要记录每一条路，以及每条路各自对应的比例系数
        需要更新的参数：每个operator的w和b，以及架构参数θ（path_theta)
        :param candidates: List of Operator(Layer)
        :param info_list: List of information about each operator(Layer)
        """
        super().__init__()
        self.info_list = info_list
        self.candidate_operators = nn.ModuleList(candidates)
        self.path_theta = nn.Parameter(torch.Tensor(len(self.candidate_operators)), requires_grad=True)

    def __str__(self):
        """
        structure:    {layer_info}::{theta}
        :return:
        """
        info = "\t".join([
            f"{layer}::{self.path_theta[index]}" for index, layer in enumerate(self.candidate_operators)
        ])
        return info

    @property
    def probability(self):
        # update, change from softmax to gumbel softmax
        return torch.nn.functional.gumbel_softmax(self.path_theta, dim=0)
        # return torch.softmax(self.path_theta, dim=0)

    def chosen_index(self):
        """
        return index and the ratio of the max choice
        :return:
        """
        probability = self.probability.data.cpu().numpy()
        index = int(np.argmax(probability))
        return index, probability[index]

    def entropy(self, eps=1e-8):
        """
        这个待定，目前还不知道有什么用
        Entropy = -Σ p log p
        :param eps:
        :return:
        """
        probability = self.probability
        log_pro = torch.log(probability + eps)
        entropy = - torch.sum(torch.mul(probability, log_pro))
        return entropy

    def forward(self, input_data):
        if self.info_list is None:
            return self._forward(input_data)
        else:
            return self._forward_fake(input_data)

    def _forward(self, input_data):
        output_record = list()
        prob = self.probability
        torch.cuda.empty_cache()
        for index, operator in enumerate(self.candidate_operators):
            try:
                tempt = operator(input_data)
                # Single return value
                if not isinstance(tempt, tuple) and not isinstance(tempt, list):
                    output_record.append([tempt * prob[index]])
                # more than one return value
                else:
                    output_record.append([
                        tp * prob[index] for tp in tempt
                    ])
            except Exception as exc:
                raise exc

        ret = list()
        for i in range(len(output_record[0])):
            convert = torch.Tensor([item[i].cpu().detach().numpy() for item in output_record]).cuda()
            ret.append(torch.sum(convert, dim=0))

        return ret[0] if len(ret) == 1 else ret

    def _forward_fake(self, input_data):
        start_timer, end_timer = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        output_record = list()
        prob = self.probability
        for index, operator in enumerate(self.candidate_operators):
            try:
                # record time
                # front = torch.cuda.memory_allocated()
                start_timer.record()
                tempt = operator(input_data)
                end_timer.record()
                torch.cuda.synchronize()
                # rear = torch.cuda.memory_allocated()

                if not isinstance(tempt, tuple) and not isinstance(tempt, list):
                    output_record.append([tempt * prob[index]])
                else:
                    output_record.append([tp * prob[index] for tp in tempt])

                # record to lut
                t_cost = start_timer.elapsed_time(end_timer)

                if os.environ.get("RECORD_MEMORY", None):
                    # record MB = total number * 4B per parameter / (1024 ** 2) at the last time
                    # m_cost = (rear - front) / 1024.0
                    m_cost = sum(p.numel() for p in operator.parameters()) * 4 / (1024 ** 2)
                    print(f"update LutItem: time({t_cost}ms)  memory({m_cost}MB)")
                else:
                    m_cost = None
                    print(f"update LutItem: time({t_cost}ms)")

                lut_generator.LUT_RECORDER.record(self.info_list[index], t_cost, m_cost)

            except Exception as e:
                print(f"LayerIndex::{index}")
                raise e

        ret = list()
        for i in range(len(output_record[0])):
            convert = torch.Tensor([item[i].cpu().detach().numpy() for item in output_record]).cuda()
            ret.append(torch.sum(convert, dim=0))

        return ret[0] if len(ret) == 1 else ret

    def get_params(self, input_size):
        """
        面向Estimator服务
        返回一个list，包含各个层的name、index、condition_dict
        :return:
        """
        params_list = list()
        prob = self.probability.data.cpu().numpy()
        for index, operator in enumerate(self.candidate_operators):
            name, tp, condition_dict = _build_condition_dict(operator)
            condition_dict["input_size"] = input_size
            params_list.append({
                "name": name,
                "type": tp,
                "index": index,
                "condition_dict": condition_dict,
                "theta": prob[index],
            })

        return params_list

    def final_params(self, input_size):
        """
        返回最大theta那个operator对应的信息，用于计算lat和mem
        :return:
        """
        index, prob = self.chosen_index()
        operator = self.candidate_operators[index]
        name, tp, condition_dict = _build_condition_dict(operator)
        condition_dict["input_size"] = input_size
        return {
            "name": name,
            "type": tp,
            "index": index,
            "condition_dict": condition_dict,
            "theta": prob
        }


class SingleLayer(nn.Module):
    def __init__(self, operator, info=None):
        super().__init__()
        self.operator = operator
        self.info = info

    def forward(self, input_data):
        if self.info is None:
            return self._forward(input_data)
        else:
            return self._forward_fake(input_data)

    def _forward(self, input_data):
        return self.operator(input_data)

    def _forward_fake(self, input_data):
        start_timer, end_timer = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start_timer.record()
        tempt = self.operator(input_data)
        end_timer.record()
        torch.cuda.synchronize()

        t_cost = start_timer.elapsed_time(end_timer)
        if os.environ.get("RECORD_MEMORY", None):
            m_cost = sum(p.numel() for p in self.operator.parameters()) * 4 / (1024 ** 2)
            print(f"update LutItem: time({t_cost}ms) memory({m_cost}MB)")
        else:
            m_cost = None
            print(f"update LutItem: time({t_cost}ms)")

        lut_generator.LUT_RECORDER.record(self.info, t_cost, m_cost)
        return tempt

    def get_params(self, input_size):
        name, tp, condition_dict = _build_condition_dict(self.operator)
        condition_dict["input_size"] = input_size
        return {
            "name": name,
            "type": tp,
            "condition_dict": condition_dict,
        }

    def __str__(self):
        return str(self.operator)
