# -*- coding:utf8 -*-

import os
import torch
import pickle
import numpy as np
import torchvision

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from utils import logger


DATASET_CHOICES = ["imagenet", "cifar10", "cifar100", "tiny100"]


def build_dataset(path, dataset=None):
    assert dataset in DATASET_CHOICES
    if dataset == "imagenet":
        data_provider = ImagenetDataProvider(path, 64, 64)
        t_dataset, v_dataset = data_provider.get_dataset()
        t_sampler, v_sampler = data_provider.get_sampler()
        return (t_dataset, t_sampler), (v_dataset, v_sampler)
    elif dataset == "cifar10":
        data_provider = CIFAR10DataProvider(path)
        t_dataset, v_dataset = data_provider.get_dataset()
        return (t_dataset, None), (v_dataset, None)
    elif dataset == "cifar100":
        data_provider = CIFAR100DataProvider(path)
        t_dataset, v_dataset = data_provider.get_dataset()
        return (t_dataset, None), (v_dataset, None)
    elif dataset == "tiny100":
        data_provider = TinyImageNet100DataProvider(path, 32)
        t_dataset, v_dataset = data_provider.get_dataset()
        return (t_dataset, None), (v_dataset, None)
    else:
        raise Exception(f"Unexpected dataset type<{dataset}>")


def build_special_dataset(path):
    with open(path, "rb") as f:
        dataset = pickle.load(f)
    return dataset


# ------------------------------ ImageNet -------------------------
def build_indices(train_dataset, length=100):
    """
    存cls2idx_list这个： [0, 68, xxx]代表某个分类的代码
    cls_index => data_index
    :param train_dataset:
    :param length: 采样CLASS的数目
    :return:
    """
    cls2idx_list = [train_dataset.class_to_idx[cls] for cls in np.random.choice(train_dataset.classes, length)]
    # save the indices
    with open(ImagenetDataProvider.DEFAULT_INDICES_PATH, "wb") as f:
        pickle.dump(cls2idx_list, f)


def load_indices(t_dataset, v_dataset):
    """
    将在cls_list中各个分类对应的数据下标都获取出来
    :param t_dataset:
    :param v_dataset:
    :return:
    """
    # read from pickle
    with open(ImagenetDataProvider.DEFAULT_INDICES_PATH, "rb") as f:
        cls_list = pickle.load(f)

    # get train indices and valid indices
    train_indices = [i for i, (_, cls) in enumerate(t_dataset.samples) if cls in cls_list]
    valid_indices = [i for i, (_, cls) in enumerate(v_dataset.samples) if cls in cls_list]
    return train_indices, valid_indices


class ImagenetDataProvider(object):
    # 我实在是懒得再写到配置里面了
    DEFAULT_IMAGENET_PATH = "E:\\imagenet"
    DEFAULT_INDICES_PATH = "class_indices.pkl"

    def __init__(self, data_path, r_size, c_size, re_sample=False, sample_num=100, mean=None, std=None):
        """

        :param data_path: ImageNet data directory path
        :param r_size: reshape_size for image input. it should be not less than c_size
        :param c_size: center crop size. it is the real input size for model
        :param re_sample: force to resample, which will generate classes indexes and override the olds
        """
        if os.path.exists(data_path):
            logger.NAS_LOGGER.info(f"ImagenetDataProvider::data_path set value ({data_path})")
            self.data_path = data_path
        else:
            self.data_path = ImagenetDataProvider.DEFAULT_IMAGENET_PATH
            logger.NAS_LOGGER.warning(
                f"DataPath not exist! ImagenetDataProvider::data_path use default value ({self.data_path}).")

        self.train_path = os.path.join(self.data_path, "train")
        self.valid_path = os.path.join(self.data_path, "val")
        self.mean = [0.485, 0.456, 0.406] if mean is None else mean
        self.std = [0.229, 0.224, 0.225] if std is None else std

        self.data_transform = transforms.Compose([
            transforms.Resize(r_size),
            transforms.CenterCrop(c_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        self.train_dataset = ImageFolder(self.train_path, self.data_transform)
        self.valid_dataset = ImageFolder(self.valid_path, self.data_transform)
        if re_sample is True or not os.path.exists(ImagenetDataProvider.DEFAULT_INDICES_PATH):
            build_indices(self.train_dataset, sample_num)
        t_indices, v_indices = load_indices(self.train_dataset, self.valid_dataset)

        self.train_sampler = torch.utils.data.SubsetRandomSampler(t_indices)
        self.valid_sampler = torch.utils.data.SubsetRandomSampler(v_indices)

    def get_dataset(self):
        return self.train_dataset, self.valid_dataset

    def get_sampler(self):
        return self.train_sampler, self.valid_sampler


# -----------------------------------CIFAR-10 ----------------------------------------
class CIFAR10DataProvider(object):
    DEFAULT_CIFAR10_PATH = "E:\\"

    def __init__(self, root):
        if os.path.exists(root):
            logger.NAS_LOGGER.info(f"CIFAR-10DataProvider::data_path set value ({root})")
            self.data_path = root
        else:
            self.data_path = CIFAR10DataProvider.DEFAULT_CIFAR10_PATH
            logger.NAS_LOGGER.warning(
                f"DataPath not exist! CIFAR-10DataProvider::data_path use default value ({self.data_path}).")

        self.mean = [0.49139968, 0.48215827, 0.44653124]
        self.std = [0.24703233, 0.24348505, 0.26158768]
        t_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
        v_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

        self.train_dataset = torchvision.datasets.CIFAR10(root=self.data_path, train=True, transform=t_transform, download=True)
        self.valid_dataset = torchvision.datasets.CIFAR10(root=self.data_path, train=False, transform=v_transform, download=True)

    def get_dataset(self):
        return self.train_dataset, self.valid_dataset


# -----------------------------------CIFAR-100 ---------------------------------------
class CIFAR100DataProvider(object):
    DEFAULT_CIFAR100_PATH = "E:\\"

    def __init__(self, root):
        if os.path.exists(root):
            logger.NAS_LOGGER.info(f"CIFAR-100DataProvider::data_path set value ({root})")
            self.data_path = root
        else:
            self.data_path = CIFAR100DataProvider.DEFAULT_CIFAR100_PATH
            logger.NAS_LOGGER.warning(
                f"DataPath <{root}> not exist! CIFAR-100DataProvider::data_path use default value ({self.data_path}).")

        self.mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
        self.std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

        t_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
        v_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

        self.train_dataset = torchvision.datasets.CIFAR100(root=self.data_path, train=True, transform=t_transform, download=True)
        self.valid_dataset = torchvision.datasets.CIFAR100(root=self.data_path, train=False, transform=v_transform, download=True)

    def get_dataset(self):
        return self.train_dataset, self.valid_dataset


# -----------------------------------Tiny-ImageNet-100 ---------------------------------------
class TinyImageNet100DataProvider(object):
    DEFAULT_PATH = "/home/why/WhyEnv/DataSet"

    def __init__(self, root, r_size=64):
        if os.path.exists(root):
            logger.NAS_LOGGER.info(f"TinyImageNet100DataProvider::data_path set value ({root})")
            self.data_path = root
        else:
            self.data_path = TinyImageNet100DataProvider.DEFAULT_PATH
            logger.NAS_LOGGER.warning(
                f"DataPath not exist! TinyImageNet100DataProvider::data_path use default value ({self.data_path}).")

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.train_path = os.path.join(self.data_path, "train")
        self.valid_path = os.path.join(self.data_path, "val")
        self.data_transform = transforms.Compose([
            transforms.Resize(r_size),
            # transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        self.train_dataset = ImageFolder(self.train_path, self.data_transform)
        self.valid_dataset = ImageFolder(self.valid_path, self.data_transform)

    def get_dataset(self):
        return self.train_dataset, self.valid_dataset

