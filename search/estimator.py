# -*- coding:utf8 -*-

import yaml
import torch

from torch.autograd.profiler_util import _format_memory
from datetime import datetime

from search.arch.lut import LutItem, get_lut_name
from search.arch.super_net import SuperNet, FinalNet
from search.arch.builder import SINGLE, MIX
from search.arch.layer import SingleLayer, MixLayer
from utils import logger
from utils.meter import AvgMeter


class Estimator(object):
    """
    Time unit: ms
    Memory unit: MB
    """
    def __init__(self, yaml_path, lat_baseline, lat_lambda, lat_length, memory_constrain, power):
        # for latency estimate
        with open(yaml_path, "r") as f:
            content = yaml.load(f, Loader=yaml.FullLoader)
        assert isinstance(content, dict)
        self.lut = {
            name: LutItem(name, raw_data) for name, raw_data in content.items()
        }
        self.lat_baseline = lat_baseline
        self.lat_lambda = lat_lambda
        self.lat_length = lat_length

        # for memory usage constrain
        self.memory_constrain = memory_constrain
        self.power = power

    def __str__(self):
        # 传入的是MB，但是这里的format_memory会把他当B来看
        # mem_info = f"HardwareConstrain = ( CurrentMemoryUsage / MemoryThreshold = {_format_memory(self.memory_constrain * 1024 ** 2)} ) ** γ"
        # lat_info = f"LatencyConstrain = ( (ΣΣ OperatorLatency × Probability - Baseline) / (λ × Baseline) ) ** γ"

        mem_info = f"HardwareConstrain = ( CurrentMemoryUsage / MemoryThreshold = {_format_memory(self.memory_constrain * 1024 ** 2)} ) ** {self.power}"
        lat_info = f"LatencyConstrain = ( (ΣΣ OperatorLatency × Probability - {self.lat_baseline}) / {self.lat_length} ) × {self.lat_lambda}"
        return "\t\t".join([lat_info, mem_info])

    def calculate_lat(self, latency):
        """
        Calculate the LAT for loss
        """
        # return ((latency - self.lat_baseline) / (self.lat_baseline * self.lat_lambda)) ** self.power
        return self.lat_lambda * (latency - self.lat_baseline) / self.lat_length

    def calculate_mem(self, memory):
        """
        Calculate
        """
        return (memory / self.memory_constrain) ** self.power

    def _get_lut_name(self, name, item_type, condition_dict):
        key_name = get_lut_name(name, item_type, condition_dict)
        if key_name not in self.lut.keys():
            raise Exception(f"Operator({key_name}) not in lut table!")
        else:
            return key_name

    def predict_time(self, net):
        """
        unit: ms
        total_time = ΣΣ operator_theta * time
        lat = α * (time - baseline)
        :param net:
        :return:
        """
        total_latency = 0
        for layer_type, params in net.get_params():
            # info structure: [layer_type, sub_layer.get_params()]
            if layer_type == SINGLE:
                detail_dict = params
                key_name = self._get_lut_name(
                    detail_dict["name"],
                    detail_dict["type"],
                    detail_dict["condition_dict"]
                )
                lat = self.lut.get(key_name).get_time()
                if lat is None:
                    information = f"Operator query failed! Name: {detail_dict['name']} Type: {LutItem.REVERSE[detail_dict['tp']]}"
                    logger.NAS_LOGGER.error(information)
                    raise Exception("Predicted Fail! Invalid operator!")
                else:
                    total_latency += lat

            elif layer_type == MIX:
                for detail_dict in params:
                    inner_index = detail_dict["index"]
                    theta = detail_dict["theta"]
                    key_name = self._get_lut_name(
                        detail_dict["name"],
                        detail_dict["type"],
                        detail_dict["condition_dict"]
                    )
                    lat = self.lut.get(key_name).get_time()
                    if lat is None:
                        information = f"Operator query failed! Block position::{inner_index}.  Name: {detail_dict['name']} Type: {LutItem.REVERSE[detail_dict['tp']]}"
                        logger.NAS_LOGGER.error(information)
                        raise Exception("Predicted Fail! Invalid operator!")
                    else:
                        total_latency += theta * lat

            else:
                raise Exception("Invalid layer type. Expected <SINGLE 1, MIX 2>, receive {}".format(layer_type))

        return total_latency

    def predict_memory(self, net):
        """
        memory unit: MB
        total memory = ΣΣ operator_theta * memory
        :param net:
        :return:
        """
        total_memory = 0
        for layer_type, params in net.get_params():
            # info structure: [layer_type, sub_layer.get_params()]
            if layer_type == SINGLE:
                detail_dict = params
                key_name = self._get_lut_name(
                    detail_dict["name"],
                    detail_dict["type"],
                    detail_dict["condition_dict"]
                )
                mem_usage = self.lut.get(key_name).get_memory()
                if mem_usage is None:
                    information = f"Operator query failed! Name: {detail_dict['name']} Type: {LutItem.REVERSE[detail_dict['tp']]}"
                    logger.NAS_LOGGER.error(information)
                    raise Exception("Predicted Fail! Invalid operator!")
                else:
                    total_memory += mem_usage

            elif layer_type == MIX:
                for detail_dict in params:
                    inner_index = detail_dict["index"]
                    theta = detail_dict["theta"]
                    key_name = self._get_lut_name(
                        detail_dict["name"],
                        detail_dict["type"],
                        detail_dict["condition_dict"]
                    )
                    mem_usage = self.lut.get(key_name).get_memory()
                    if mem_usage is None:
                        information = f"Operator query failed! Block position::{inner_index}. Name: {detail_dict['name']} Type: {LutItem.REVERSE[detail_dict['tp']]}"
                        logger.NAS_LOGGER.error(information)
                        raise Exception("Predicted Fail! Invalid operator!")
                    else:
                        total_memory += theta * mem_usage

            else:
                raise Exception("Invalid layer type. Expected <SINGLE 1, MIX 2>, receive {}".format(layer_type))

        return total_memory

    def predict_final(self, net):
        """

        :param net:
        :return: latency, hardware(memory usage percentage)
        """
        total_memory = 0
        total_latency = 0
        assert isinstance(net, SuperNet)
        for _, layer in net.net.named_children():
            detail_dict = layer.get_params(net.input_size) if isinstance(layer, SingleLayer) else layer.final_params(net.input_size)
            key_name = self._get_lut_name(
                detail_dict["name"],
                detail_dict["type"],
                detail_dict["condition_dict"]
            )
            lat = self.lut.get(key_name).get_time()
            mem = self.lut.get(key_name).get_memory()
            if lat is None or mem is None:
                information = f"Operator query failed!"
                logger.NAS_LOGGER.error(information)
                raise Exception("Predicted Fail! Invalid operator!")
            else:
                total_latency += lat
                total_memory += mem
        return total_latency, total_memory


def warm_up(net, data_loader, device):
    inputs, labels = next(iter(data_loader))
    logger.NAS_LOGGER.info("Start to warm up at {}".format(datetime.now()))
    random_input = torch.rand(inputs.shape)
    random_input = random_input.to(device)
    for i in range(10):
        logger.NAS_LOGGER.info("Warm up {}/10 at {}".format(i + 1, datetime.now()))
        net(random_input)


class ExternalEstimator(object):
    """
    对外来的神经网络进行同期对比，考核CE、推理时延以及占用的内存大小
    """
    def __init__(self, criterion, memory_constrain, power):
        self.calculate_fn = lambda x: (x / memory_constrain) ** power
        self.criterion = criterion

    def predict_time(self, net, data_loader, device):
        def _accuracy(outputs, target, topk=(1, 5)):
            maxk = max(topk)
            batch_size = target.size(0)
            _, pred = outputs.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()
            res = list()
            for k in topk:
                correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(1.0 / batch_size))

            return res

        infer_time = AvgMeter()
        losses = AvgMeter()
        top1 = AvgMeter()
        top5 = AvgMeter()

        start_timer, end_timer = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        assert isinstance(net, torch.nn.Module)
        net = net.to(device)
        net.eval()

        with torch.no_grad():
            for data, label in data_loader:
                data = data.to(device)
                label = label.to(device)
                start_timer.record()
                output = net(data)
                end_timer.record()
                torch.cuda.synchronize()
                infer_time.update(start_timer.elapsed_time(end_timer))

                losses.update(self.criterion(output, label))
                acc1, acc5 = _accuracy(output, label)
                top1.update(acc1[0], data.size(0))
                top5.update(acc5[0], data.size(0))
        # 这里我稍微优化了一下，各自返回了最优值
        return losses.min, infer_time.min, top1.avg, top5.avg

    def predict_memory(self, net):
        total_memory = sum(p.numel() for p in net.parameters()) * 4 / (1024 ** 2)
        return self.calculate_fn(total_memory), total_memory

    def estimate(self, net, data_loader, device):
        warm_up(net, data_loader, device)
        loss, it, top1, top5 = self.predict_time(net, data_loader, device)
        hc, mem = self.predict_memory(net)
        info = "Net: {tp}\tLoss: {ls:.4f}\tInfer time: {lat:.4f} ms\tHardware constrain: {hc:.4f}\tTop1: {top1:.4f}\tTop5: {top5:.4f}".format(
            tp=type(net), ls=loss, lat=it, hc=hc, top1=top1, top5=top5
        )
        logger.NAS_LOGGER.info(info)
        return {
            "loss": round(loss.item(), 4),
            "lat": round(it, 4),
            "memory": mem,
            "mem_constrain": hc,
            "top1": round(top1.item(), 4),
            "top5": round(top5.item(), 4),
        }


if __name__ == '__main__':
    estimator = Estimator("lut.yaml", 45, 6)
    print(estimator)
