# -*- coding:utf8 -*-
from search.arch.lut import get_lut_name, LutItem

"""
临时生成可用的lut.yaml文件

        conv__cin__cout__kernel__stride__padding
        final__cin__cout
        skip__cin__cout
        avg_pool__kernel
        max_pool__kernel

        LUT_DICT = {
            "name": { "time": mean time, "memory": mean memory usage},
        }
"""
import yaml


class LutRecorder(object):
    def __init__(self, yaml_path):
        self.path = yaml_path
        self.lut_dict = dict()

    def save(self):
        with open(self.path, "w") as f:
            yaml.dump(self.lut_dict, f)

    def record(self, name, t_cost, m_cost=None):
        """
        采用累积的方法来更新信息（参考Proxyless论文）
        time: ms; memory: MB; energy: To Be Determined
        :param name: 操作符的名字
        :param m_cost: 本次耗时
        :param t_cost: 本次内存占用
        :return:
        """
        if name not in self.lut_dict.keys():
            self.lut_dict[name] = {
                "time": 0.0,
                "memory": 0.0,
                "energy": 0,
                "count": 0
            }
        # update
        self.lut_dict[name]["count"] += 1
        self.lut_dict[name]["time"] = (self.lut_dict[name]["time"] * (self.lut_dict[name]["count"] - 1) + t_cost) / self.lut_dict[name]["count"]

        if m_cost is not None:
            # 因为更新后只会在最后一次记录参数量大小，所以不再使用count来求平均
            # self.lut_dict[name]["memory"] = (self.lut_dict[name]["memory"] * (self.lut_dict[name]["count"] - 1) + m_cost) / self.lut_dict[name]["count"]
            self.lut_dict[name]["memory"] = m_cost


LUT_RECORDER = LutRecorder("../search/new_lut.yaml")

