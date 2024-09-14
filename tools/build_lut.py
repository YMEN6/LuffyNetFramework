# -*- coding:utf -*-
import os
import sys
import torch
import warnings

import yaml
from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
warnings.filterwarnings("ignore")

from search.estimator import Estimator
from search.data_set import build_dataset
from search.loss import ComprehensiveCriterion
from search.manager import SuperNetRunManager, FinalNetRunManager, get_path
from search.arch.super_net import SuperNet, FinalNet


def build_generator():
    from tools.test import simple_test
    from utils import lut_generator

    torch.cuda.empty_cache()
    simple_test(name="resnet18_100", dataset="cifar100")
    # simple_test(name="resnet18", dataset="cifar10")
    lut_generator.LUT_RECORDER.save()


def merge_lut(source_path, data_path):
    """
    update lut from data_path to source_path
    :param source_path:
    :param data_path:
    :return:
    """
    if os.path.exists(source_path):
        with open(source_path, "r") as f:
            source_dict = yaml.load(f, Loader=yaml.FullLoader)
    else:
        source_dict = dict()

    assert os.path.exists(data_path)
    with open(data_path, "r") as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)

    source_dict.update(data_dict)
    with open(source_path, "w") as f:
        yaml.dump(source_dict, f)


if __name__ == '__main__':
    source_path = "../search/lut.yaml"
    data_path = "../search/new_lut.yaml"
    merge_lut(source_path, data_path)

    # build_generator()



