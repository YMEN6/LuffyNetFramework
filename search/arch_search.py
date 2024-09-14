# -*- coding:utf8 -*-
"""
Main program
"""
import os
import sys
import warnings
import torch
import argparse
import numpy as np

if True:
    try:
        from utils import logger
    except ImportError:
        current_path = os.path.dirname(os.path.abspath(__file__))
        root_path = os.path.join(current_path, "..")
        sys.path.extend([current_path, root_path])

    warnings.filterwarnings("ignore")
    from utils import logger
    from utils.config_parser import NAS_CONFIG
    from search.manager import SuperNetRunManager, FinalNetRunManager, get_path, AttentiveRunManager
    from search.data_set import build_dataset, DATASET_CHOICES
    from search.estimator import Estimator
    from search.arch.super_net import SuperNet, SUPER_NET_DICT


# for global
parser = argparse.ArgumentParser()
parser.add_help = True
parser.add_argument("--save_path", type=str, required=True)
parser.add_argument("--manual_seed", type=int, default=0)

# for data
"""
--data_path:    存放数据集的外部目录地址，必须传入
--dataset:      所使用的数据集类型，目前只支持CIFAR10, CIFAR100, ImageNet
"""
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--dataset", type=str, choices=DATASET_CHOICES)
parser.add_argument("--device", type=str, default=None)

# set train model
"""
--train or -t: train. else valid
train:  1) target = "super_net":  SuperNetRunManager对SuperNet进行训练
        2) target = "final_net":  FinalNetRunManager对采样后的FinalNet进行训练，此时从SuperNetRunManager路径读入
        
valid:  1) target = "super_net":  SuperNetRunManager对刚采样后的FinalNet进行评估
        2) target = "final_net":  FinalNetRunManager对训练后的FinalNet进行评估，此时从FinalNetRunManager路径读入
"""
parser.add_argument("-t", "--train", action="store_true")
parser.add_argument("--target", type=str, default="super_net", choices=["super_net", "final_net"])

# set path
"""
--final_path: 当传入时，从指定路径读入final_net，要求使用完整的路径（而不是目录的路径）
--check_path: 当传入时，从指定目录读入check_path，只要求传入目录的路径 (历史遗留问题)
当不传入时，默认从 save_path的目录中寻找并读入（默认工作模式）
"""
parser.add_argument("--final_path", type=str, default=None)
parser.add_argument("--check_path", type=str, default=None)

# [super net]
parser.add_argument("--net_config", type=str, default="resnet18_100", choices=SUPER_NET_DICT.keys())

# some parameters read from parser will override those from ini config. default=None means that use config
# [train]
parser.add_argument("--epoch", type=int, default=NAS_CONFIG.get_train().get("epoch"))
parser.add_argument("--batch_size", type=int, default=NAS_CONFIG.get_train().get("batch_size"))

# [weight]
"""
weight parameters will be used as follows:
1) train::super_net, for weight of super net
2) train::final_net, for weight of final net
"""
parser.add_argument("--w_lr", type=float, default=NAS_CONFIG.get_weight().get("learning_rate"))
parser.add_argument("--w_momentum", type=float, default=NAS_CONFIG.get_weight().get("momentum"))
parser.add_argument("--w_decay", type=float, default=NAS_CONFIG.get_weight().get("weight_decay"))

# [theta]
parser.add_argument("--a_decay", type=float, default=NAS_CONFIG.get_theta().get("weight_decay"))
parser.add_argument("--a_lr", type=float, default=NAS_CONFIG.get_theta().get("learning_rate"))

# [estimator]
parser.add_argument("--lut_path", type=str, default=NAS_CONFIG.get_estimator().get("config"))
parser.add_argument("--lat_lambda", type=float, default=NAS_CONFIG.get_estimator().get("lambda", 2))
parser.add_argument("--lat_baseline", type=float, default=NAS_CONFIG.get_estimator().get("baseline", 6))
parser.add_argument("--lat_length", type=float, default=NAS_CONFIG.get_estimator().get("length", 18))

parser.add_argument("--constrain", type=int, default=NAS_CONFIG.get_estimator().get("constrain"))
parser.add_argument("--power", type=int, default=NAS_CONFIG.get_estimator().get("power", 10))

# [manager]
parser.add_argument("--print_frequency", type=int, default=NAS_CONFIG.get_manager().get("print_frequency"))


if __name__ == '__main__':
    args = parser.parse_args()

    # set manual seed
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)

    # -----------------------------build logger-------------------------------
    os.makedirs(args.save_path, exist_ok=True)

    if logger.NAS_LOGGER is None:
        log_path = os.path.join(args.save_path, "log")
        logger.NAS_LOGGER = logger.setup_logger(log_path)

    # -----------------------------build dataset-------------------------------
    data_config_dict = NAS_CONFIG.get_data()
    (train_dataset, train_sampler), (valid_dataset, valid_sampler) = build_dataset(
        args.data_path, args.dataset
    )

    # -----------------------------build super net-------------------------------
    is_dict = {
        "imagenet": 224,
        "cifar10": 32,
        "cifar100": 32
    }
    super_net = SuperNet(args.net_config, is_dict[args.dataset])

    # -----------------------------build final net-------------------------------
    if args.target == "final_net":
        # load final net from final_path
        if args.final_path is not None and os.path.exists(args.final_path):
            final_net = torch.load(args.final_path)
        # load final net from default path (ensure that you have run SuperNetRunManager to build final net before)
        else:
            # train: load final net from SPRM.path
            if args.train is True:
                _, _, final_path = get_path(os.path.join(args.save_path, "SuperNet"))
            # valid: load final net from FNRM.path
            else:
                _, final_path, _ = get_path(os.path.join(args.save_path, "FinalNet"))
            final_net = torch.load(final_path)
    else:
        final_net = None

    # -----------------------------build estimator-------------------------------
    power = args.power
    assert power > 1
    estimator = Estimator(
        yaml_path=args.lut_path,
        lat_baseline=args.lat_baseline, lat_lambda=args.lat_lambda, lat_length=args.lat_length,
        memory_constrain=args.constrain, power=args.power
    )

    # -----------------------------build run manager-------------------------------
    # train: weight and theta
    # valid: origin FinalNet
    if args.target == "super_net":
        # run manager without BO and WO
        # nas_manager = SuperNetRunManager(
        #     net=super_net, epoch=args.epoch,
        #     w_lr=args.w_lr, w_momentum=args.w_momentum, w_decay=args.w_decay,
        #     a_lr=args.a_lr, a_decay=args.a_decay,
        #     train_dataset=train_dataset, train_batch_size=args.batch_size, train_sampler=train_sampler,
        #     valid_dataset=valid_dataset, valid_batch_size=args.batch_size, valid_sampler=valid_sampler,
        #     estimator=estimator,
        #     directory_path=args.save_path, device=args.device,
        #     print_frequency=args.print_frequency
        # )

        # attentive run manager with BO and WO
        attentive_dict = NAS_CONFIG.get_attentive()
        p_epoch = attentive_dict["preprocess_epoch"]
        i_epoch = attentive_dict["iter_epoch"]
        i_numbers = attentive_dict["iter_numbers"]
        i_lr = attentive_dict["iter_lr"]
        nas_manager = AttentiveRunManager(
            net=super_net, preprocess_epoch=p_epoch, iter_epoch=i_epoch, iter_numbers=i_numbers,
            w_lr=args.w_lr, w_momentum=args.w_momentum, w_decay=args.w_decay,
            t_lr=args.a_lr, t_decay=args.a_decay,
            i_lr=i_lr, i_momentum=args.w_momentum, i_decay=args.w_decay,
            train_dataset=train_dataset, train_batch_size=args.batch_size,
            valid_dataset=valid_dataset, valid_batch_size=args.batch_size,
            estimator=estimator, save_directory=args.save_path,
            device=args.device, print_frequency=args.print_frequency
        )
    # train: original FinalNet
    # valid: learned FinalNet
    else:
        nas_manager = FinalNetRunManager(
            net=final_net, epoch=args.epoch,
            lr=args.w_lr, momentum=args.w_momentum, decay=args.w_decay,
            train_dataset=train_dataset, train_batch_size=args.batch_size, train_sampler=train_sampler,
            valid_dataset=valid_dataset, valid_batch_size=args.batch_size, valid_sampler=valid_sampler,
            directory_path=args.save_path, device=args.device,
            print_frequency=args.print_frequency
        )

    # load from check point, and continue to run
    if args.check_path is not None:
        assert os.path.exists(args.check_path)
        nas_manager.load_checkpoint(args.check_path)
    else:
        nas_manager.load_checkpoint()
    nas_manager.show_config()

    # -----------------------------------train----------------------------------------
    # train the super net
    torch.cuda.empty_cache()
    if args.train is True:
        # start to train
        # nas_manager.warm_up()
        nas_manager.train()

    # -----------------------------------eval----------------------------------------
    # show the structure and accuracy of the net
    else:
        nas_manager.analyze()
