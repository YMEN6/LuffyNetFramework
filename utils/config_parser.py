# -*- coding:utf8 -*-

import configparser
import os.path


class NasConfig(object):
    STR = 1
    INT = 2
    FLOAT = 3
    BOOL = 4
    EVAL = 5

    def __init__(self, cfg_path):
        self.config = configparser.ConfigParser()
        self.config.read(cfg_path, encoding="utf8")
        self.value_dict = dict()

        self._init()

    def _get_value(self, section, key, tp):
        function = {
            NasConfig.STR: self.config.get,
            NasConfig.INT: self.config.getint,
            NasConfig.FLOAT: self.config.getfloat,
            NasConfig.BOOL: self.config.getboolean,
            NasConfig.EVAL: self.config.get,
        }
        assert tp in function.keys()
        return function[tp](section, key)

    def _init(self):
        query_dict = {
            "estimator": [
                ["config", NasConfig.STR],
                ["baseline", NasConfig.FLOAT],
                ["length", NasConfig.FLOAT],
                ["lambda", NasConfig.FLOAT],
                ["constrain", NasConfig.INT],
                ["power", NasConfig.INT],
            ],
            "train": [
                ["epoch", NasConfig.INT],
                ["batch_size", NasConfig.INT],
            ],
            "weight": [
                ["learning_rate", NasConfig.FLOAT],
                ["momentum", NasConfig.FLOAT],
                ["weight_decay", NasConfig.FLOAT],
            ],
            "theta": [
                ["learning_rate", NasConfig.FLOAT],
                ["weight_decay", NasConfig.FLOAT],
            ],
            "manager": [
                ["print_frequency", NasConfig.INT]
            ],
            "data": [
                ["reshape_size", NasConfig.INT],
                ["crop_size", NasConfig.INT]
            ],
            "attentive": [
                ["preprocess_epoch", NasConfig.INT],
                ["iter_numbers", NasConfig.INT],
                ["iter_epoch", NasConfig.INT],
                ["iter_lr", NasConfig.FLOAT]
            ]

        }

        for section, key_list in query_dict.items():
            self.value_dict[section] = dict()
            for line in key_list:
                key_name, tp = line
                value = self._get_value(section, key_name, tp)
                if tp == NasConfig.EVAL:
                    value = eval(value)
                self.value_dict[section][key_name] = value

    def get_estimator(self):
        return self.value_dict["estimator"]

    def get_train(self):
        return self.value_dict["train"]

    def get_valid(self):
        return self.value_dict["valid"]

    def get_manager(self):
        return self.value_dict["manager"]

    def get_data(self):
        return self.value_dict["data"]

    def get_theta(self):
        return self.value_dict["theta"]

    def get_weight(self):
        return self.value_dict["weight"]

    def get_attentive(self):
        return self.value_dict["attentive"]


ini_path = os.path.join(os.path.split(os.path.abspath("."))[0], "search", "nas.ini")
NAS_CONFIG = NasConfig(ini_path)
