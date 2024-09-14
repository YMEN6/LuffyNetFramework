# -*- coding:utf8 -*-
import os
import gc
import tqdm
import pickle

import numpy as np
import torch.cuda
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.autograd.profiler_util import _format_time as format_time
from thop import profile
from copy import copy, deepcopy
from datetime import datetime
from collections import OrderedDict

from search.estimator import Estimator, ExternalEstimator
from search.loss import ComprehensiveCriterion
from search.data_set import build_dataset
from search.arch.super_net import SuperNet, FinalNet
from search.arch.layer import MixLayer, SingleLayer
from utils import logger
from utils.meter import AvgMeter


def _accuracy(output, target, topk=(1,)):
    """
    直接抄的网上，和ProxyLess里面写的几乎一样了
    :param output:
    :param target:
    :param topk:
    :return:
    """
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()

    res = list()
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        # 这里取100.0好像也行，那么出来就直接是百分比了，如果取1那就是小数
        res.append(correct_k.mul_(1.0 / batch_size))

    return res


def _load(target, state_dict):
    """
    Load model from weight dict
    :param target:
    :param state_dict:
    :return:
    """
    assert isinstance(target, torch.nn.Module)
    model_dict = target.state_dict()
    temp_dict = dict()
    for k, v in state_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
    model_dict.update(temp_dict)
    target.load_state_dict(model_dict)
    if hasattr(target, "freeze"):
        target.freeze()


def get_path(dir_path):
    """
    考虑到SuperNetRunManager和FinalNetRunManager在存放刚生成的final net时存在义务交叉（FinalNet的需要去SuperNet的目录读）
    因此将这一层的处理交给最外层去解决，不在这个里面来处理这个问题
    :param dir_path:
    :return:
    """
    checkpoint_dir = os.path.join(dir_path, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "ck.pth")

    # init best model path for saving
    best_model_dir = os.path.join(dir_path, "best")
    os.makedirs(best_model_dir, exist_ok=True)
    best_model_path = os.path.join(best_model_dir, "best.pth")

    # init final net path
    final_net_path = os.path.join(dir_path, "final")
    os.makedirs(final_net_path, exist_ok=True)
    final_net_path = os.path.join(final_net_path, "final.pth")

    return checkpoint_path, best_model_path, final_net_path


def _get_timer():
    """
    用于在GPU上计时
    :return:
    """
    start_timer, end_timer = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    return start_timer, end_timer


class SuperNetRunManager(object):
    def __init__(self, net, epoch,
                 w_lr, w_momentum, w_decay,
                 a_lr, a_decay,
                 train_dataset, train_batch_size, train_sampler,
                 valid_dataset, valid_batch_size, valid_sampler,
                 estimator,
                 directory_path, device=None,
                 train_ratio=0.8,  # ratio of data for weight training
                 valid_frequency=1, print_frequency=10, save_frequency=1):
        # for train
        self.epoch = epoch
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.best_acc = 0.0

        # for DataLoader, include Training Data and Validation Data
        if train_sampler is None:
            self.train_dataloader = DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=True, pin_memory=True)
        else:
            self.train_dataloader = DataLoader(train_dataset, batch_size=self.train_batch_size, sampler=train_sampler, pin_memory=True)
        if valid_sampler is None:
            self.valid_dataloader = DataLoader(valid_dataset, batch_size=self.valid_batch_size, shuffle=True, pin_memory=True)
        else:
            self.valid_dataloader = DataLoader(valid_dataset, batch_size=self.valid_batch_size, sampler=valid_sampler, pin_memory=True)

        # init SuperNet and Optimizer for W
        self.device = device if device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.final_net = None
        self.super_net = copy.deepcopy(net)
        assert isinstance(self.super_net, SuperNet)
        self.super_net = self.super_net.to(self.device)

        # set training ratio to slice training on weight vs theta
        self.train_ratio = train_ratio

        sgd_params = list()
        theta_params = list()
        for layer in self.super_net.net:
            for name, param in layer.named_parameters():
                # record parameters for theta
                if name.startswith("path_theta") and isinstance(layer, MixLayer):
                    theta_params.append(param)
                # record parameters for weight
                elif param.requires_grad:
                    sgd_params.append(param)
                else:
                    continue

        # use cosine to model cosine annealing
        self.weight_optimizer = optim.SGD(sgd_params, lr=w_lr, momentum=w_momentum, weight_decay=w_decay)
        self.weight_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.weight_optimizer, self.epoch, eta_min=1e-6)

        # choose not to use scheduler
        self.theta_optimizer = optim.Adam(theta_params, lr=a_lr, weight_decay=a_decay, betas=(0.5, 0.99))
        # consider whether a scheduler is necessary for theta_optimizer here

        # init Estimator and Loss Function
        self.estimator = estimator
        self.criterion = ComprehensiveCriterion()

        # other parameter for debug and print
        self.valid_frequency = valid_frequency
        self.print_frequency = print_frequency
        self.save_frequency = save_frequency

        # self.path: directory for saving all file we need
        self.path = directory_path
        self._checkpoint_path, self._best_model_path, self._final_net_path = get_path(
            os.path.join(self.path, "SuperNet"))
        self._start_epoch = 0

        # for config show
        w_opt_info = "WeightLR:{wlr:.5f}\tWeightMomentum:{wmm}\tWeightDecay:{wdc:.5f}\t".format(wlr=w_lr,
                                                                                                wmm=w_momentum,
                                                                                                wdc=w_decay)
        a_opt_info = "ThetaLR:{alr:.5f}\tThetaDecay:{adc:.5f}\t".format(alr=a_lr, adc=a_decay)
        head_info = "Epoch:{epoch}\tRatio:{ratio}\t" \
                    "Estimator:{estimate}\nSavePath:{path}\n". \
            format(epoch=self.epoch, ratio=train_ratio, estimate=estimator, path=self.path)
        self.info = "\n".join([head_info, w_opt_info, a_opt_info])

    @property
    def config(self):
        config = {
            key: self.__dict__[key] for key in self.__dict__ if not key.startswith("_")
        }
        return config

    def load_checkpoint(self, path=None):
        """

        :param path: 这里改动为最外层目录的路径
        :return:
        """
        if path is None:
            ck_path = self._checkpoint_path
            fn_path = self._final_net_path
        else:
            path = os.path.join(path, "SuperNet")
            assert os.path.exists(path)
            ck_path, _, fn_path = get_path(path)

        # skip if file not exist(first run of program)
        if not os.path.exists(ck_path):
            return

        checkpoint_dict = torch.load(ck_path)
        self.super_net.load_state_dict(checkpoint_dict["state_dict"])
        self.weight_optimizer.load_state_dict(checkpoint_dict["weight_optimizer"])
        self.theta_optimizer.load_state_dict(checkpoint_dict["theta_optimizer"])
        self.best_acc = checkpoint_dict["best_acc"]

        # keep running or restart
        if checkpoint_dict["done"] is not True:
        # if checkpoint_dict["done"] is not True or checkpoint_dict["epoch"] < self.epoch - 1:
            self._start_epoch = checkpoint_dict["epoch"]

        if os.path.exists(fn_path):
            self.final_net = torch.load(fn_path)

    def predict_latency(self):
        """
        调用Estimator预估延迟，返回单位为ms
        :return:
        """
        if isinstance(self.estimator, Estimator):
            total_time = self.estimator.predict_time(self.super_net)
            lat = self.estimator.calculate_lat(total_time)
            logger.NAS_LOGGER.info(f"Predict time: {total_time:.4f} ms, LatConstrain: {lat:.4f}")
            return lat
        else:
            raise TypeError(f"Expected estimator_type<GPU, MOBILE>, receive {type(self.estimator)}")

    def predict_hardware(self):
        """
        类似上面，也是直接调用Estimator，这个返回是一个值（没有单位，被消掉了）
        :return:
        """
        if isinstance(self.estimator, Estimator):
            total_memory = self.estimator.predict_memory(self.super_net)
            hc = self.estimator.calculate_mem(total_memory)
            logger.NAS_LOGGER.info(f"Predict memory: {total_memory:.4f} MB, HardwareConstrain: {hc:.4f}")
            return hc
        else:
            raise TypeError(f"Expected estimator_type<GPU, MOBILE>, receive {type(self.estimator)}")

    def validate(self):
        """
        batch_time: ms
        :return:
        """
        self.super_net.eval()

        # 另外这里是validate，所以不再关注每一层具体跑多久，而是关注整体下来跑多久
        batch_time = AvgMeter()
        losses = AvgMeter()
        top1 = AvgMeter()
        top5 = AvgMeter()

        start_timer, end_timer = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            for i, (datas, labels) in enumerate(self.valid_dataloader):
                datas = datas.to(self.device)
                labels = labels.to(self.device)

                start_timer.record()
                outputs = self.super_net(datas)
                end_timer.record()
                torch.cuda.synchronize()
                batch_time.update(start_timer.elapsed_time(end_timer))

                # calculate lat, mem, loss and acc
                latency = self.predict_latency()
                memory_constrain = self.predict_hardware()
                loss = self.criterion(outputs, labels, latency, memory_constrain)
                acc1, acc5 = _accuracy(outputs, labels, (1, 5))

                losses.update(loss, datas.size(0))
                top1.update(acc1[0], datas.size(0))
                top5.update(acc5[0], datas.size(0))

                if i % self.print_frequency == 0 or i + 1 == len(self.valid_dataloader):
                    log_info = "Valid {}/{}\t " \
                               "Batch Time (avg:{batch_time.avg:.4f} ms)\t" \
                               "Loss (val:{losses.val:.4f})\t (avg:{losses.avg:.4f})\t" \
                               "Lat ({lat:.4f})\tHC ({mc:.4f})\t" \
                               "Top-1 acc (val:{top1.val:.4f})\t (avg:{top1.avg:.4f})\t" \
                               "Top-5 acc (val:{top5.val:.4f})\t (avg:{top5.avg:.4f})\n". \
                        format(i, len(self.valid_dataloader) - 1,
                               batch_time=batch_time, losses=losses, lat=latency, mc=memory_constrain, top1=top1, top5=top5)
                    logger.NAS_LOGGER.info(log_info)

        return losses.avg, top1.avg, top5.avg

    def _train_one_epoch(self, current_epoch):
        assert isinstance(self.super_net, SuperNet)
        self.super_net.train()

        batch_time = AvgMeter()
        losses = AvgMeter()
        top1 = AvgMeter()
        top5 = AvgMeter()

        milestone = (len(self.train_dataloader) * self.train_ratio)
        start_timer, end_timer = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        epoch_iterator = tqdm.tqdm(self.train_dataloader, desc=f"Training (Epoch {current_epoch + 1}/{self.epoch})")
        for i, (datas, labels) in enumerate(epoch_iterator):
            datas = datas.to(self.device)
            labels = labels.to(self.device)

            start_timer.record()
            outputs = self.super_net(datas)
            end_timer.record()
            torch.cuda.synchronize()
            batch_time.update(start_timer.elapsed_time(end_timer))

            latency = self.predict_latency()
            memory_constrain = self.predict_hardware()
            loss = self.criterion(outputs, labels, latency, memory_constrain)
            acc1, acc5 = _accuracy(outputs, labels, (1, 5))

            losses.update(loss, datas.size(0))
            top1.update(acc1[0], datas.size(0))
            top5.update(acc5[0], datas.size(0))

            # for gradient descend
            self.super_net.zero_grad()
            loss.backward()

            # weight training
            if i < milestone:
                current_lr = self.weight_optimizer.param_groups[0]['lr']
                self.weight_optimizer.step()
                mode = "weight"
            else:
                current_lr = self.theta_optimizer.param_groups[0]['lr']
                self.theta_optimizer.step()
                mode = "theta"

            # print info
            if i % self.print_frequency == 0 or i + 1 == len(self.train_dataloader):
                log_info = "Train Data {}/{}\t " \
                           "Batch Time (avg:{batch_time.avg:.4f} ms)\t" \
                           "Loss (avg:{losses.avg:.4f})\t" \
                           "Top-1 acc (avg:{top1.avg:.4f})\t" \
                           "Top-5 acc (avg:{top5.avg:.4f})\t" \
                           "Learning rate {lr:.5f}\t Mode: {md}\n". \
                    format(i, len(self.train_dataloader) - 1,
                           batch_time=batch_time, losses=losses, top1=top1, top5=top5, lr=current_lr, md=mode)
                logger.NAS_LOGGER.info(log_info)

        # update lr
        self.weight_scheduler.step()

        return losses.avg, top1.avg, top5.avg

    def train(self):
        """
        time unit: ms
        :return:
        """
        # clear cuda cache
        torch.cuda.empty_cache()
        start_timer, end_timer = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for epoch in range(self._start_epoch, self.epoch):
            logger.NAS_LOGGER.info("\n" + ("-" * 30) + f"Train epoch: {epoch + 1}" + ("-" * 30) + "\n")
            start_timer.record()

            # return avg
            train_loss, train_top1, train_top5 = self._train_one_epoch(epoch)
            end_timer.record()
            torch.cuda.synchronize()
            elapsed_time = start_timer.elapsed_time(end_timer)

            epoch_log_info = [
                f"Train {epoch + 1}/{self.epoch}. Time per epoch: {elapsed_time} ms. ",
                "Loss (val:{train_loss:.4f})\t"
                "Top-1 acc (val:{train_top1:.3f})\t"
                "Top-5 acc (val:{train_top5:.3f})".
                format(train_loss=train_loss, train_top1=train_top1, train_top5=train_top5)
            ]
            logger.NAS_LOGGER.info("\n".join(epoch_log_info))

            if epoch % self.valid_frequency == 0 or epoch == self.epoch - 1:
                valid_loss, valid_top1, valid_top5 = self.validate()
                # higher accuracy, update and save
                if valid_top1 > self.best_acc:
                    valid_log_info = f"Train::best accuracy update from {self.best_acc:.2f} to {valid_top1:.2f} at {epoch + 1}/{self.epoch}"
                    logger.NAS_LOGGER.info(valid_log_info)
                    self.best_acc = max(self.best_acc, valid_top1)
                    torch.save(self.super_net, self._best_model_path)
                    logger.NAS_LOGGER.info(f"Train::save best model at  {epoch + 1}/{self.epoch} to {self._best_model_path}")

            # save the check point so that we can keep training after unexpected interruption
            if epoch % self.save_frequency == 0 or epoch == self.epoch - 1:
                basic_state_dict = {
                    "epoch": min(epoch + 1, self.epoch),
                    "best_acc": self.best_acc,
                    "weight_optimizer": self.weight_optimizer.state_dict(),
                    "theta_optimizer": self.theta_optimizer.state_dict(),
                    "state_dict": self.super_net.state_dict(),
                    "done": epoch == self.epoch - 1
                }
                torch.save(basic_state_dict, self._checkpoint_path)
                logger.NAS_LOGGER.info(f"Train::basically save model at {epoch + 1}/{self.epoch} to {self._checkpoint_path}")

    def show_config(self):
        """
        这里是打印RunManager的信息，不是SuperNet or FinalNet的信息，类比Proxyless的run config
        :return:
        """
        logger.NAS_LOGGER.info(self.info)

    def warm_up(self):
        inputs, labels = next(iter(self.valid_dataloader))
        logger.NAS_LOGGER.info("Start to warm up at {}".format(datetime.now()))
        random_input = torch.rand(inputs.shape)
        random_input = random_input.to(self.device)
        for i in range(5):
            logger.NAS_LOGGER.info("Warm up {}/5 at {}".format(i + 1, datetime.now()))
            outs = self.super_net(random_input)

    def get_final_net(self, force=None):
        """
        get the final net after searching and training, and save the final net
        :return:
        """
        if self.final_net is None or force:
            assert isinstance(self.super_net, SuperNet)
            latency, memory = self.estimator.predict_final(self.super_net)
            order_dict = self.super_net.generate_final_net()
            self.final_net = FinalNet(ordered_dict=order_dict, lat=latency, mem=memory)
            # 这里直接将整个网络存下来，里面的变量也会自动存下来的
            torch.save(self.final_net, self._final_net_path)
            logger.NAS_LOGGER.info(f"Save final net to {self._final_net_path}")
        return self.final_net

    def analyze(self):
        """
        但这个是还没有完全重新训练过的最终网络，其实也可以做一个对比，看重新训练后，ACC会有多大的提升
        1) get_test_input_data
        2) validate and calculate the top1 and top5
        3) show runtime, hardware usage
        :return:
        """
        start_timer, end_timer = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        batch_time = AvgMeter()
        losses = AvgMeter()
        top1 = AvgMeter()
        top5 = AvgMeter()

        # show final net first: net structure, lat, mem
        if self.final_net is None:
            self.get_final_net()
        info = "Analyze::Latency:{lat}ms\tMemory:{usage}MB\nModelStructure:\n{net}".format(
            lat=self.final_net.latency, usage=self.final_net.memory,
            net=self.final_net
        )
        logger.NAS_LOGGER.info(info)

        self.warm_up()

        current = 1
        for data, label in tqdm.tqdm(self.valid_dataloader):
            datas = data.to(self.device)
            labels = label.to(self.device)

            start_timer.record()
            outputs = self.final_net(datas)
            end_timer.record()
            torch.cuda.synchronize()
            batch_time.update(start_timer.elapsed_time(end_timer))

            latency = self.predict_latency()
            memory_constrain = self.predict_hardware()
            loss = self.criterion(outputs, labels, latency, memory_constrain)
            acc1, acc5 = _accuracy(outputs, labels, (1, 5))

            losses.update(loss, datas.size(0))
            top1.update(acc1[0], datas.size(0))
            top5.update(acc5[0], datas.size(0))

            if current % self.print_frequency == 0 or current == len(self.valid_dataloader):
                total_parameters = sum(p.numel() for p in self.final_net.parameters()) * 4 / (1024 ** 2)
                log_info = "Analyzing {}/{}\t " \
                           "Infer Time (avg:{batch_time.avg:.4f} ms)\t" \
                           "Parameters ({tps:.4f} MB)\t" \
                           "Loss (val:{losses.val:.4f})\t (avg:{losses.avg:.4f})\t" \
                           "Top-1 acc (val:{top1.val:.4f})\t (avg:{top1.avg:.4f})\t" \
                           "Top-5 acc (val:{top5.val:.4f})\t (avg:{top5.avg:.4f})\n". \
                    format(current, len(self.valid_dataloader),
                           batch_time=batch_time, tps=total_parameters, losses=losses, top1=top1, top5=top5)
                logger.NAS_LOGGER.info(log_info)

            current += 1

    def show_supernet(self):
        logger.NAS_LOGGER.info(self.super_net.show())


class FinalNetRunManager(object):
    def __init__(
            self, net, epoch, lr, momentum, decay,
            train_dataset, train_batch_size, train_sampler,
            valid_dataset, valid_batch_size, valid_sampler,
            directory_path, device=None,
            valid_frequency=1, print_frequency=10, save_frequency=10):
        self.epoch = epoch
        self._start_epoch = 0

        self.device = device if device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = copy.deepcopy(net)
        assert isinstance(self.net, torch.nn.Module)
        self.net = self.net.to(self.device)

        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.best_acc = 0.0

        if train_sampler is None:
            self.train_dataloader = DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=True, pin_memory=True)
        else:
            self.train_dataloader = DataLoader(train_dataset, batch_size=self.train_batch_size, sampler=train_sampler, pin_memory=True)
        if valid_sampler is None:
            self.valid_dataloader = DataLoader(valid_dataset, batch_size=self.valid_batch_size, shuffle=True, pin_memory=True)
        else:
            self.valid_dataloader = DataLoader(valid_dataset, batch_size=self.valid_batch_size, sampler=valid_sampler, pin_memory=True)

        self.optimizer = optim.SGD(self.net.parameters(), momentum=momentum, lr=lr, weight_decay=decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.epoch, eta_min=1e-6)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.valid_frequency = valid_frequency
        self.print_frequency = print_frequency
        self.save_frequency = save_frequency
        self._checkpoint_path, self._best_model_path, _ = get_path(os.path.join(directory_path, "FinalNet"))

    def load_checkpoint(self, path=None):
        """
        only load from final net manager, because "net" at init has load from super net manager

        checkpoint_dict = {
            "epoch": epoch,
            "best_acc": self.best_acc,
            "optimizer": self.optimizer.state_dict(),
            "state_dict": self.super_net.state_dict(),
            "done": last training has been finished or not
        }
        :param path:
        :return:
        """
        if path is None:
            ck_path = self._checkpoint_path
            net_path = self._best_model_path
        else:
            path = os.path.join(path, "FinalNet")
            assert os.path.exists(path)
            ck_path, net_path, _ = get_path(path)

        # skip if file not exist(first run of program)
        if not os.path.exists(ck_path):
            return

        checkpoint_dict = torch.load(ck_path)
        self.best_acc = checkpoint_dict["best_acc"]

        # keep running or restart
        if checkpoint_dict["done"] is not True:
        # if checkpoint_dict["done"] is not True or checkpoint_dict["epoch"] < self.epoch - 1:
            self._start_epoch = checkpoint_dict["epoch"]
            self.optimizer.load_state_dict(checkpoint_dict["optimizer"])

        if os.path.exists(net_path):
            self.net = torch.load(net_path)

    def validate(self):
        """
        batch_time: ms
        :return:
        """
        self.net.eval()
        # 这里就不用list了，直接用这个来存
        # 另外这里是validate，所以不再关注每一层具体跑多久，而是关注整体下来跑多久
        batch_time = AvgMeter()
        losses = AvgMeter()
        top1 = AvgMeter()
        top5 = AvgMeter()

        start_timer, end_timer = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            for i, (datas, labels) in enumerate(self.valid_dataloader):
                datas = datas.to(self.device)
                labels = labels.to(self.device)

                start_timer.record()
                outputs = self.net(datas)
                end_timer.record()
                torch.cuda.synchronize()
                batch_time.update(start_timer.elapsed_time(end_timer))

                # calculate lat, mem, loss and acc
                loss = self.criterion(outputs, labels)
                acc1, acc5 = _accuracy(outputs, labels, (1, 5))

                losses.update(loss, datas.size(0))
                top1.update(acc1[0], datas.size(0))
                top5.update(acc5[0], datas.size(0))

                if i % self.print_frequency == 0 or i + 1 == len(self.valid_dataloader):
                    log_info = "Valid {}/{}\t " \
                               "Batch Time (avg:{batch_time.avg:.4f} ms)\t" \
                               "Loss (val:{losses.val:.4f})\t (avg:{losses.avg:.4f})\t" \
                               "Top-1 acc (val:{top1.val:.4f})\t (avg:{top1.avg:.4f})\t" \
                               "Top-5 acc (val:{top5.val:.4f})\t (avg:{top5.avg:.4f})\n". \
                        format(i, len(self.valid_dataloader) - 1,
                               batch_time=batch_time, losses=losses, top1=top1, top5=top5)
                    logger.NAS_LOGGER.info(log_info)

        return losses.avg, top1.avg, top5.avg

    def _train_one_epoch(self, current_epoch):
        self.net.train()
        batch_time = AvgMeter()
        losses = AvgMeter()
        top1 = AvgMeter()
        top5 = AvgMeter()

        start_timer, end_timer = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        epoch_iterator = tqdm.tqdm(self.train_dataloader, desc=f"Training (Epoch {current_epoch + 1}/{self.epoch})")
        # for i, (datas, labels) in enumerate(self.train_dataloader):
        for i, (datas, labels) in enumerate(epoch_iterator):
            datas = datas.to(self.device)
            labels = labels.to(self.device)

            start_timer.record()
            outputs = self.net(datas)
            end_timer.record()
            torch.cuda.synchronize()
            batch_time.update(start_timer.elapsed_time(end_timer))

            loss = self.criterion(outputs, labels)
            acc1, acc5 = _accuracy(outputs, labels, (1, 5))

            losses.update(loss, datas.size(0))
            top1.update(acc1[0], datas.size(0))
            top5.update(acc5[0], datas.size(0))
            current_lr = self.optimizer.param_groups[0]['lr']

            # for gradient descend
            self.net.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % self.print_frequency == 0 or i + 1 == len(self.train_dataloader):
                log_info = "Train Data {}/{}\t " \
                           "Batch Time (avg:{batch_time.avg:.4f} ms)\t" \
                           "Loss (val:{losses.val:.4f})\t (avg:{losses.avg:.4f})\t" \
                           "Top-1 acc (val:{top1.val:.4f})\t (avg:{top1.avg:.4f})\t" \
                           "Top-5 acc (val:{top5.val:.4f})\t (avg:{top5.avg:.4f})\t" \
                           "Learning rate {lr:.5f}\n". \
                    format(i, len(self.valid_dataloader) - 1,
                           batch_time=batch_time, losses=losses, top1=top1, top5=top5, lr=current_lr)
                logger.NAS_LOGGER.info(log_info)

        # update lr
        self.scheduler.step()

        return losses.avg, top1.avg, top5.avg

    def train(self):
        start_timer, end_timer = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for epoch in range(self._start_epoch, self.epoch):
            logger.NAS_LOGGER.info("\n" + ("-" * 30) + f"Train epoch: {epoch + 1}" + ("-" * 30) + "\n")
            start_timer.record()

            # clear cuda cache
            torch.cuda.empty_cache()
            # return avg
            train_loss, train_top1, train_top5 = self._train_one_epoch(epoch)
            end_timer.record()
            torch.cuda.synchronize()
            elapsed_time = start_timer.elapsed_time(end_timer)

            epoch_log_info = [
                f"Train {epoch + 1}/{self.epoch}. Time per epoch: {elapsed_time} ms. "
                f"Training is expected to complete in {format_time(elapsed_time * 1000)}.",
                "Loss (val:{train_loss:.4f})\t"
                "Top-1 acc (val:{train_top1:.3f})\t"
                "Top-5 acc (val:{train_top5:.3f})".
                format(train_loss=train_loss, train_top1=train_top1, train_top5=train_top5)
            ]
            logger.NAS_LOGGER.info("\n".join(epoch_log_info))

            if epoch % self.valid_frequency == 0 or epoch == self.epoch - 1:
                valid_loss, valid_top1, valid_top5 = self.validate()
                # higher accuracy, update and save
                if valid_top1 > self.best_acc:
                    valid_log_info = f"Train::best accuracy update from {self.best_acc:.2f} to {valid_top1:.2f} at {epoch + 1}/{self.epoch}"
                    logger.NAS_LOGGER.info(valid_log_info)
                    self.best_acc = max(self.best_acc, valid_top1)
                    torch.save(self.net, self._best_model_path)
                    logger.NAS_LOGGER.info(
                        f"Train::save best model at  {epoch + 1}/{self.epoch} to {self._best_model_path}")

            # save the check point so that we can keep training after unexpected interruption
            if epoch % self.save_frequency == 0 or epoch == self.epoch - 1:
                basic_state_dict = {
                    "epoch": min(epoch + 1, self.epoch),
                    "best_acc": self.best_acc,
                    "optimizer": self.optimizer.state_dict(),
                    "state_dict": self.net.state_dict(),
                    "done": epoch == self.epoch - 1
                }
                torch.save(basic_state_dict, self._checkpoint_path)
                logger.NAS_LOGGER.info(
                    f"Train::basically save model at {epoch + 1}/{self.epoch} to {self._checkpoint_path}")

    def show_config(self):
        """
        这里是打印RunManager的信息，不是SuperNet or FinalNet的信息，类比Proxyless的run config
        :return:
        """
        info = "Epoch:{epoch}\tLr:{lr}\tSavePath:{path}". \
            format(epoch=self.epoch, lr=self.optimizer.param_groups[0]['lr'], path=self._checkpoint_path)
        logger.NAS_LOGGER.info(info)

    def warm_up(self):
        inputs, labels = next(iter(self.valid_dataloader))
        logger.NAS_LOGGER.info("Start to warm up at {}".format(datetime.now()))
        random_input = torch.rand(inputs.shape)
        random_input = random_input.to(self.device)
        for i in range(5):
            logger.NAS_LOGGER.info("Warm up {}/5 at {}".format(i+1, datetime.now()))
            outs = self.net(random_input)

    def analyze(self):
        start_timer, end_timer = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        batch_time = AvgMeter()
        losses = AvgMeter()
        top1 = AvgMeter()
        top5 = AvgMeter()

        self.warm_up()

        current = 1
        for data, label in tqdm.tqdm(self.valid_dataloader):
            datas = data.to(self.device)
            labels = label.to(self.device)

            start_timer.record()
            outputs = self.net(datas)
            end_timer.record()
            torch.cuda.synchronize()
            batch_time.update(start_timer.elapsed_time(end_timer))

            loss = self.criterion(outputs, labels)
            acc1, acc5 = _accuracy(outputs, labels, (1, 5))

            losses.update(loss, datas.size(0))
            top1.update(acc1[0], datas.size(0))
            top5.update(acc5[0], datas.size(0))

            if current % self.print_frequency == 0 or current == len(self.valid_dataloader):
                total_parameters = sum(p.numel() for p in self.net.parameters()) * 4 / (1024 ** 2)
                log_info = "Analyzing {}/{}\t " \
                           "Infer Time (avg:{batch_time.avg:.4f} ms)\t" \
                           "Parameters ({tps:.4f} MB)\t" \
                           "Loss (val:{losses.val:.4f})\t (avg:{losses.avg:.4f})\t" \
                           "Top-1 acc (val:{top1.val:.4f})\t (avg:{top1.avg:.4f})\t" \
                           "Top-5 acc (val:{top5.val:.4f})\t (avg:{top5.avg:.4f})\n". \
                    format(current, len(self.valid_dataloader),
                           batch_time=batch_time, tps=total_parameters, losses=losses, top1=top1, top5=top5)
                logger.NAS_LOGGER.info(log_info)

            current += 1


class BasicRunManager(object):
    """
    没办法，还是将RunManager的基类摘出来写了，但是已有的SP和FINAL我就不重写了，主要是懒
    Optimizer都沿用SGD
    """
    def __init__(self, net, epoch, save_directory,
                 w_lr, w_momentum, weight_decay,
                 estimator, criterion, train_dataloader, valid_dataloader,
                 device=None, valid_frequency=1, save_frequency=1, print_frequency=10):
        self.device = device if device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.epoch = epoch

        self.estimator = estimator
        assert isinstance(net, SuperNet)
        self.super_net = copy.deepcopy(net)
        self.super_net = self.super_net.to(self.device)
        self.final_net = None

        # init var about training
        sgd_params = self.select_params()
        self.optimizer = optim.SGD(sgd_params, lr=w_lr, momentum=w_momentum, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.epoch, eta_min=1e-6)
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.best_acc = 0.0

        # init other param
        self.valid_frequency = valid_frequency
        self.save_frequency = save_frequency
        self.print_frequency = print_frequency
        self.path = save_directory
        self._checkpoint_path, self._best_model_path, self._final_net_path = get_path(
            os.path.join(self.path, "SuperNet"))
        self._start_epoch = 0

        info = "Epoch: {epoch}\tLr: {lr:.4f}\tMomentum: {mm:.4f}\tWeightDecay: {decay:.4f}\tSavePath: {path}\n".format(
            epoch=self.epoch, lr=w_lr, mm=w_momentum, decay=weight_decay, path=save_directory
        )
        logger.NAS_LOGGER.info(info)

    def select_params(self):
        raise NotImplementedError

    def load_checkpoint(self, path=None):
        if path is None:
            ck_path = self._checkpoint_path
            fn_path = self._final_net_path
        else:
            path = os.path.join(path, "SuperNet")
            assert os.path.exists(path)
            ck_path, _, fn_path = get_path(path)

        # skip if file not exist(first run of program)
        if not os.path.exists(ck_path):
            return

        checkpoint_dict = torch.load(ck_path)
        self.super_net.load_state_dict(checkpoint_dict["state_dict"])
        self.optimizer.load_state_dict(checkpoint_dict["weight_optimizer"])
        self.best_acc = checkpoint_dict["best_acc"]

        # keep running or restart
        if checkpoint_dict["done"] is not True:
        # if checkpoint_dict["done"] is not True or checkpoint_dict["epoch"] < self.epoch - 1:
            self._start_epoch = checkpoint_dict["epoch"]

        if os.path.exists(fn_path):
            self.final_net = torch.load(fn_path)

    def predict_latency(self):
        """
        调用Estimator预估延迟，返回单位为ms
        :return:
        """
        if isinstance(self.estimator, Estimator):
            total_time = self.estimator.predict_time(self.super_net)
            lat = self.estimator.calculate_lat(total_time)
            logger.NAS_LOGGER.info(f"Predict time: {total_time:.4f} ms, LatConstrain: {lat:.4f}")
            return lat
        else:
            raise TypeError(f"Expected estimator_type<GPU, MOBILE>, receive {type(self.estimator)}")

    def predict_hardware(self):
        """
        类似上面，也是直接调用Estimator，这个返回是一个值（没有单位，被消掉了）
        :return:
        """
        if isinstance(self.estimator, Estimator):
            total_memory = self.estimator.predict_memory(self.super_net)
            hc = self.estimator.calculate_mem(total_memory)
            logger.NAS_LOGGER.info(f"Predict memory: {total_memory:.4f} MB, HardwareConstrain: {hc:.4f}")
            return hc
        else:
            raise TypeError(f"Expected estimator_type<GPU, MOBILE>, receive {type(self.estimator)}")

    def validate(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def get_final_net(self, force=None):
        """
        get the final net after searching and training, and save the final net
        :return:
        """
        if self.final_net is None or force:
            assert isinstance(self.super_net, SuperNet)
            latency, memory = self.estimator.predict_final(self.super_net)
            order_dict = self.super_net.generate_final_net()
            self.final_net = FinalNet(ordered_dict=order_dict, lat=latency, mem=memory)
            # 这里直接将整个网络存下来，里面的变量也会自动存下来的
            torch.save(self.final_net, self._final_net_path)
            logger.NAS_LOGGER.info(f"Save final net to {self._final_net_path}")
        return self.final_net

    def warm_up(self):
        inputs, labels = next(iter(self.valid_dataloader))
        logger.NAS_LOGGER.info("Start to warm up at {}".format(datetime.now()))
        random_input = torch.rand(inputs.shape)
        random_input = random_input.to(self.device)
        for i in range(10):
            logger.NAS_LOGGER.info("Warm up {}/5 at {}".format(i + 1, datetime.now()))
            outs = self.super_net(random_input)

    def analyze(self):
        """
        但这个是还没有完全重新训练过的最终网络，其实也可以做一个对比，看重新训练后，ACC会有多大的提升
        1) get_test_input_data
        2) validate and calculate the top1 and top5
        3) show runtime, hardware usage
        :return:
        """
        start_timer, end_timer = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        batch_time = AvgMeter()
        losses = AvgMeter()
        top1 = AvgMeter()
        top5 = AvgMeter()

        # show final net first: net structure, lat, mem
        if self.final_net is None:
            self.get_final_net()
        info = "Analyze::Latency:{lat}\tHardwareConstrain:{usage}\nModelStructure:\n{net}".format(
            lat=self.final_net.latency, usage=self.final_net.memory,
            net=self.final_net
        )
        logger.NAS_LOGGER.info(info)

        # self.warm_up()

        current = 1
        for data, label in tqdm.tqdm(self.valid_dataloader):
            datas = data.to(self.device)
            labels = label.to(self.device)

            start_timer.record()
            outputs = self.final_net(datas)
            end_timer.record()
            torch.cuda.synchronize()
            batch_time.update(start_timer.elapsed_time(end_timer))

            latency = self.predict_latency()
            memory_constrain = self.predict_hardware()
            loss = self.criterion(outputs, labels, latency, memory_constrain)
            acc1, acc5 = _accuracy(outputs, labels, (1, 5))

            losses.update(loss, datas.size(0))
            top1.update(acc1[0], datas.size(0))
            top5.update(acc5[0], datas.size(0))

            if current % self.print_frequency == 0 or current == len(self.valid_dataloader):
                total_parameters = sum(p.numel() for p in self.final_net.parameters()) * 4 / (1024 ** 2)
                log_info = "Analyzing {}/{}\t " \
                           "Infer Time (avg:{batch_time.avg:.4f} ms)\t" \
                           "Parameters ({tps:.4f} MB)\t" \
                           "Loss (val:{losses.val:.4f})\t (avg:{losses.avg:.4f})\t" \
                           "Top-1 acc (val:{top1.val:.4f})\t (avg:{top1.avg:.4f})\t" \
                           "Top-5 acc (val:{top5.val:.4f})\t (avg:{top5.avg:.4f})\n". \
                    format(current, len(self.valid_dataloader),
                           batch_time=batch_time, tps=total_parameters, losses=losses, top1=top1, top5=top5)
                logger.NAS_LOGGER.info(log_info)

            current += 1

    def show_supernet(self):
        logger.NAS_LOGGER.info(self.super_net.show())


def _autolabel(rects):
    """
    在plt.bar上显示值
    :param rects:
    :return:
    """
    for rect in rects:
        height = rect.get_height()
        plt.annotate(
            format(height, '.2f'),
            xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 2),
            textcoords="offset points", ha='center', va='bottom'
        )


def _multi_bars(labels, datas, tick_step=1, group_gap=0.2, bar_gap=0.0):
    """
    绘图，多个纵轴并列
    :param labels: X轴坐标标签序列
    :param datas:  数据集，要求每个元素的长度相同
    :param tick_step: 默认X轴刻度步长为1，通过该变量来调整步长
    :param group_gap: 不同组的间隙，以免重叠
    :param bar_gap: 同一组内间隙
    :return:
    """
    colors = ["blue", "cyan", "darkorange", "magenta", "skyblue"]

    # ticks为x轴刻度
    ticks = np.arange(len(labels)) * tick_step
    # group_num为数据的组数，即每组柱子的柱子个数
    group_num = len(datas)
    # group_width为每组柱子的总宽度，group_gap 为柱子组与组之间的间隙。
    group_width = tick_step - group_gap
    # bar_span为每组柱子之间在x轴上的距离，即柱子宽度和间隙的总和
    bar_span = group_width / group_num
    # bar_width为每个柱子的实际宽度
    bar_width = bar_span - bar_gap
    # baseline_x为每组柱子第一个柱子的基准x轴位置，随后的柱子依次递增bar_span即可
    baseline_x = ticks - (group_width - bar_span) / 2
    for index, y in enumerate(datas):
        rects = plt.bar(baseline_x + index * bar_span, y, bar_width, color=colors[index])
        _autolabel(rects)
    # x轴刻度标签位置与x轴刻度一致
    plt.xticks(ticks, labels)


class AnalysisRunManager(object):
    def __init__(self, source_dir, dataset, data_path,
                 batch_size,
                 memory_constrain=45, power=6, device="cuda",
                 sp_dataset_path=None, pre_load=True):
        """
        加载目标网络模型，并进行同一的训练与验证评估
        :param source_dir:
        :param dataset:
        :param memory_constrain:
        :param power:
        :param device:
        :parm pre_load: 是否提前载入，面向某些直接传入的模型提供的接口
        """
        self.device = device
        self.source_dir = source_dir
        self.save_dir = os.path.join(self.source_dir, f"analysis_{dataset.lower()}")
        self.dataset = dataset
        self.batch_size = batch_size
        self.reporter = dict()

        # load model from source dir
        self.model_dict = OrderedDict()
        if pre_load:
            self._load_model()

        # build estimator
        self.criterion = torch.nn.CrossEntropyLoss()
        self.estimator = ExternalEstimator(self.criterion, memory_constrain, power)

        # read dataset
        self.special_dataset_path = os.path.join(self.save_dir, f"new_{dataset.lower()}.pkl") if sp_dataset_path is None else sp_dataset_path
        (self.train_dataset, _), (self.valid_dataset, _) = build_dataset(data_path, dataset)

    def _load_model(self):
        """
        读取刚训练完的FinalNet
        :return:
        """
        self.model_dict.clear()
        for dir_name in os.listdir(self.source_dir):
            # pass irrelevant directory
            if self.dataset.lower() not in dir_name.lower() or \
                    not os.path.isdir(os.path.join(self.source_dir, dir_name)) or \
                    dir_name.startswith("analysis"):
                continue
            if dir_name in self.model_dict.keys():
                raise Exception(f"Duplicated model name {dir_name}! Program stop!")
            _, model_path, _ = get_path(os.path.join(self.source_dir, dir_name, "FinalNet"))
            model = torch.load(model_path)
            self.model_dict[dir_name] = model.to(self.device)
            logger.NAS_LOGGER.info(f"Read model <{dir_name}> and move to device <{self.device}>")
        logger.NAS_LOGGER.info(f"Load {len(self.model_dict)} models from {self.source_dir}")

    def _reload_model(self):
        """
        读取最终优化后的模型，并替换到self.model_dict中
        :return:
        """
        self.model_dict.clear()
        for dir_name in os.listdir(os.path.join(self.save_dir, "models")):
            model_name = dir_name.split(".")[0]
            if model_name in self.model_dict.keys():
                raise Exception(f"Duplicated model name {model_name}! Program stop!")
            model = torch.load(os.path.join(self.save_dir, "models", dir_name))
            self.model_dict[model_name] = model.to(self.device)
            logger.NAS_LOGGER.info(f"Reload model <{model_name}> and move to device <{self.device}>")
        logger.NAS_LOGGER.info(f"Reload {len(self.model_dict)} models from {os.path.join(self.save_dir, 'models')}")

    def set_model_dict(self, model_dict):
        """
        不load，直接传入指定模型
        :param model_dict:
        :return:
        """
        if isinstance(model_dict, dict):
            self.model_dict = OrderedDict(model_dict)
        elif isinstance(model_dict, OrderedDict):
            self.model_dict = copy(model_dict)
        else:
            raise Exception(f"Invalid model dict type, expected <dict, OrderedDict>, receive {type(model_dict)}")

    def _train(self, model, loader, optimizer, desc, print_frequency):
        """
        训练模型
        :param model:
        :param loader:
        :param optimizer:
        :param desc: 用于tqdm的描述
        :param print_frequency: 输出日志的间隔
        :return:
        """
        model.train()
        iterator = tqdm.tqdm(loader, desc=desc)
        for j, (data, label) in enumerate(iterator):
            data, label = data.to(self.device), label.to(self.device)
            output = model(data)
            loss = self.criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if j % print_frequency == 0 or j + 1 == len(loader):
                top1, top5 = _accuracy(output, label, (1, 5))
                logger.NAS_LOGGER.info(f"Train Loss: {loss.item():.4f} Top1: {top1.item():.4f} Top5: {top5.item():.4f}")

    def _valid(self, model, loader):
        """
        验证模型
        :param model:
        :param loader:
        :return:
        """
        model.eval()
        for j, (data, label) in enumerate(loader):
            data, label = data.to(self.device), label.to(self.device)
            output = model(data)
            loss = self.criterion(output, label)
            top1, top5 = _accuracy(output, label, (1, 5))
            logger.NAS_LOGGER.info(f"Valid Loss: {loss.item():.4f} Top1: {top1.item():.4f} Top5: {top5.item():.4f}")

    def final_optimize2(self, print_frequency=10):
        """
        参考别人的方法重新进行训练，这一次超参数直接定死
        :param print_frequency: 打印频率
        :return:
        """
        epoch = 200
        lr_list = [0.1, 0.02, 0.004, 0.0008]
        change_list = [0, 59, 119, 159]

        train_loader = DataLoader(self.train_dataset, shuffle=True, pin_memory=True, batch_size=self.batch_size)
        valid_loader = DataLoader(self.valid_dataset, shuffle=True, pin_memory=True, batch_size=self.batch_size)
        save_dir = os.path.join(self.save_dir, "models2")
        os.makedirs(save_dir, exist_ok=True)

        for model_name, model in self.model_dict.items():
            model = model.to(self.device)
            optimizer = optim.SGD(model.parameters(), lr=lr_list[0])
            logger.NAS_LOGGER.info(f"Model {model_name} start to retrain")
            for i in range(epoch):
                # train
                desc = f"Train Epoch {i + 1}/{epoch}"
                self._train(model, train_loader, optimizer, desc, print_frequency)

                # validate
                self._valid(model, valid_loader)

                if i in change_list:
                    lr = lr_list[change_list.index(i)]
                    for p in optimizer.param_groups:
                        p['lr'] = lr
                        logger.NAS_LOGGER.info(f"Change lr to {lr} at epoch {i+1}")

            save_path = os.path.join(save_dir, f"{model_name}.pth")
            torch.save(model, save_path)
            logger.NAS_LOGGER.info(f"Model {model_name} save to {save_path}")

    def final_optimize3(self):
        """
        直接使用SGD进行优化
        :return:
        """
        epoch = 200
        lr = 0.2

        train_loader = DataLoader(self.train_dataset, shuffle=True, pin_memory=True, batch_size=self.batch_size)
        valid_loader = DataLoader(self.valid_dataset, shuffle=True, pin_memory=True, batch_size=self.batch_size)
        save_dir = os.path.join(self.save_dir, "models3")
        os.makedirs(save_dir, exist_ok=True)

        for model_name, model in self.model_dict.items():
            model = model.to(self.device)
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
            logger.NAS_LOGGER.info(f"Model {model_name} start to retrain")
            for i in range(epoch):
                # train
                desc = f"Train Epoch {i + 1}/{epoch}"
                self._train(model, train_loader, optimizer, desc, 10)

                # validate
                self._valid(model, valid_loader)

            save_path = os.path.join(save_dir, f"{model_name}.pth")
            torch.save(model, save_path)
            logger.NAS_LOGGER.info(f"Model {model_name} save to {save_path}")

    def _estimate(self):
        """
        重新测量目标模型的各项指标
        :return:
        """
        del self.reporter
        self.reporter = {
            "label": list(),
            "latency": list(),
            "memory": list(),
            "top1": list(),
            "top5": list(),
            "loss": list(),
            "flops": list(),
        }
        valid_loader = DataLoader(self.valid_dataset, shuffle=True, pin_memory=True, batch_size=self.batch_size)
        data, _ = next(iter(valid_loader))
        data = data.to(self.device)
        for model_name, model in self.model_dict.items():
            if model_name in self.reporter["label"]:
                raise Exception(f"Duplicated model name {model_name}! Program stop!")
            info = self.estimator.estimate(model, valid_loader, self.device)
            flops, _ = profile(model, inputs=(data, ))
            flops /= 1e9
            self.reporter["label"].append(model_name)
            self.reporter["latency"].append(info["lat"])
            self.reporter["memory"].append(info["memory"])
            self.reporter["top1"].append(info["top1"])
            self.reporter["top5"].append(info["top5"])
            self.reporter["loss"].append(info["loss"])
            self.reporter["flops"].append(flops)
            logger.NAS_LOGGER.info(f"Model <{model_name}> Loss: {info['loss']:.4f} "
                                   f"Latency: {info['lat']:.4f}ms "
                                   f"Memory: {info['memory']:.4f}MB Flops: {flops:.4f} "
                                   f"Top1: {info['top1']:.4f} Top5: {info['top5']:.4f} ")
        # logger.NAS_LOGGER.info(f"Estimate {len(self.reporter['label'])} models.")

    def _save(self, pickle_path=None):
        if pickle_path is None:
            pickle_path = os.path.join(self.save_dir, "info.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(self.reporter, f)
            logger.NAS_LOGGER.info(f"Save Estimate Info to {pickle_path}")

    def _show(self):
        """
        展示Estimate的结果，分为2个图
        图1: loss, top1, top5
        图2: latency, memory, flops
        :return:
        """
        # paint the first figure
        plt.subplot(3, 1, 1)
        _multi_bars(
            self.reporter["label"],
            [self.reporter["loss"], self.reporter["top1"], self.reporter["top5"]],
            bar_gap=0.01
        )
        plt.ylabel("Loss/Top1/Top5")
        plt.title("Model Accuracy")
        plt.legend(['loss', 'top1', 'top5'])

        # paint the second figure
        plt.subplot(3, 1, 3)
        _multi_bars(
            self.reporter["label"],
            [self.reporter["latency"], self.reporter["memory"], self.reporter["flops"]],
            bar_gap=0.01
        )
        plt.ylabel("Performance")
        plt.title("Model Hardware Information")
        plt.legend(['latency', 'memory', 'flops'])

        plt.tight_layout()
        plt.show()

    def estimate(self):
        """
        将model_dict中的模型进行对比，并将结果绘制成图
        :return:
        """
        self._estimate()
        self._save()
        self._show()

    def final_estimate(self):
        """
        加载训练过后的最终模型，并进行验证
        :return:
        """
        self._reload_model()
        self._estimate()
        self._save()
        self._show()


class AttentiveRunManager(object):
    """
    超网的训练分为以下几个阶段：
    1） 前面若干轮，暂定为20，由preprocess_epoch控制，在这几伦轮里，所有的网络结构公平的进行训练；
    2） 从preprocess_epoch+1开始，在每层中根据softmax筛选排名最前和最后的几个网络结构，解冻这些结构，冻结其他结构，
    然后进行训练，持续iter_epoch后，继续重复这一过程iter_numbers次
    """
    def __init__(self, net, preprocess_epoch, iter_epoch, iter_numbers,
                 w_lr, w_momentum, w_decay,
                 t_lr, t_decay,
                 i_lr, i_momentum, i_decay,
                 train_dataset, train_batch_size,
                 valid_dataset, valid_batch_size,
                 estimator,
                 save_directory, device=None,
                 train_ratio=0.8,  # ratio of data for weight training
                 valid_frequency=1, print_frequency=10, save_frequency=1):
        # record basic var
        self.preprocess_epoch = preprocess_epoch
        self.iter_epoch = iter_epoch
        self.iter_numbers = iter_numbers
        self.iter_lr = i_lr
        self.iter_momentum = i_momentum
        self.iter_decay = i_decay
        self.total_epoch = self.preprocess_epoch + self.iter_epoch * self.iter_numbers
        self.train_ratio = train_ratio

        self._start_epoch = 0
        self.best_acc = 0.0

        # build data loader
        self.train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True)
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=True, pin_memory=True)

        # build model
        self.device = device if device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.final_net = None
        assert isinstance(net, SuperNet)
        self.super_net = deepcopy(net)
        self.super_net = self.super_net.to(self.device)
        self._load_weight()

        # build constant optimizer and scheduler
        sgd_params, theta_params = self._select_params()
        self.preprocess_optimizer = optim.SGD(sgd_params, lr=w_lr, momentum=w_momentum, weight_decay=w_decay)
        self.preprocess_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.preprocess_optimizer, self.preprocess_epoch, eta_min=1e-6)
        self.theta_optimizer = optim.Adam(theta_params, lr=t_lr, weight_decay=t_decay, betas=(0.5, 0.99))

        # init Estimator and Loss Function
        self.estimator = estimator
        self.criterion = ComprehensiveCriterion()

        # record other prams
        self.valid_frequency = valid_frequency
        self.print_frequency = print_frequency
        self.save_frequency = save_frequency

        # init save path
        self.path = save_directory
        self._checkpoint_path, self._best_model_path, self._final_net_path = get_path(os.path.join(self.path, "SuperNet"))

        # config info
        tab = "\t"
        centralized_info = f"PreprocessEpoch:{self.preprocess_epoch}{tab}IterEpoch:{iter_epoch}{tab}" \
                           f"IterNumbers:{iter_numbers}{tab}TotalEpoch:{self.total_epoch}{tab}" \
                           f"TrainRatio:{train_ratio}{tab}SavePath:{self.path}"
        prep_optimizer_info = f"PreprocessOptimizer:SGD{tab}lr={w_lr}{tab}momentum={w_momentum}{tab}decay={w_decay}"
        theta_optimizer_info = f"ThetaOptimizer:Adam{tab}lr={t_lr}{tab}decay={t_decay}"
        iter_optimizer_info = f"IterOptimizer:SGD{tab}lr={i_lr}{tab}momentum={i_momentum}{tab}decay={i_decay}"
        self.info = "\n".join([centralized_info, prep_optimizer_info, theta_optimizer_info, iter_optimizer_info])

    def _select_params(self):
        """
        分别获取process_optimizer和theta_optimizer需要训练的参数
        :return:
        """
        sgd_params, theta_params = list(), list()
        for layer in self.super_net.net:
            for name, param in layer.named_parameters():
                # record parameters for theta
                if name.startswith("path_theta") and isinstance(layer, MixLayer):
                    theta_params.append(param)
                # record parameters for weight
                elif param.requires_grad:
                    sgd_params.append(param)
                else:
                    continue
        return sgd_params, theta_params

    def _load_weight(self):
        """
        加载权重文件，默认路径为 search/weights/{net_config}.pth
        :return:
        """
        path = os.path.join("search", "weights", f"{self.super_net.config_name}.pth")
        if os.path.exists(path):
            weight_dict = torch.load(path)
            _load(self.super_net, weight_dict)

    def load_checkpoint(self, path=None):
        """
        加载存档点
        :param path:
        :return:
        """
        if path is None:
            ck_path = self._checkpoint_path
            fn_path = self._final_net_path
        else:
            path = os.path.join(path, "SuperNet")
            assert os.path.exists(path)
            ck_path, _, fn_path = get_path(path)

        if not os.path.exists(ck_path):
            return

        checkpoint_dict = torch.load(ck_path)
        self.super_net.load_state_dict(checkpoint_dict["state_dict"])
        self.preprocess_optimizer.load_state_dict(checkpoint_dict["preprocess_optimizer"])
        self.theta_optimizer.load_state_dict(checkpoint_dict["theta_optimizer"])
        self.best_acc = checkpoint_dict["best_acc"]

        if checkpoint_dict["done"] is not True:
            self._start_epoch = checkpoint_dict["epoch"]

        if os.path.exists(fn_path):
            self.final_net = torch.load(fn_path)

    def show_config(self):
        """
        输出RunManager的基本信息
        :return:
        """
        logger.NAS_LOGGER.info(self.info)

    def predict_latency(self):
        """
        调用Estimator预估延迟，返回单位为ms
        :return:
        """
        if isinstance(self.estimator, Estimator):
            total_time = self.estimator.predict_time(self.super_net)
            lat = self.estimator.calculate_lat(total_time)
            logger.NAS_LOGGER.info(f"Predict time: {total_time:.4f} ms, LatConstrain: {lat:.4f}")
            return lat
        else:
            raise TypeError(f"Expected estimator_type<GPU, MOBILE>, receive {type(self.estimator)}")

    def predict_hardware(self):
        """
        类似上面，也是直接调用Estimator，这个返回是一个值（没有单位，被消掉了）
        :return:
        """
        if isinstance(self.estimator, Estimator):
            total_memory = self.estimator.predict_memory(self.super_net)
            hc = self.estimator.calculate_mem(total_memory)
            logger.NAS_LOGGER.info(f"Predict memory: {total_memory:.4f} MB, HardwareConstrain: {hc:.4f}")
            return hc
        else:
            raise TypeError(f"Expected estimator_type<GPU, MOBILE>, receive {type(self.estimator)}")

    def dynamic_freeze_and_free(self, numbers=2):
        """
        BestOptimize与WorstOptimize实现
        动态调整SuperNet中的模块，冻结一部分并解冻一部分
        :param numbers: 选择若干个最好的和最坏的解冻，其他冻结
        :return:
        """
        train_params = list()
        for name, layer in self.super_net.net.named_children():
            # SingleLayer: keep active
            if isinstance(layer, SingleLayer):
                layer.requires_grad_(True)
                train_params.extend(layer.parameters())
            # MixLayer: freeze and unfreeze
            elif isinstance(layer, MixLayer):
                # probability = layer.probability
                probability = layer.path_theta
                _, max_index = torch.topk(probability, k=numbers)
                max_index = max_index.data.cpu().numpy().tolist()
                _, min_index = torch.topk(-probability, k=numbers)
                min_index = min_index.data.cpu().numpy().tolist()
                index = set(max_index).union(min_index)
                logger.NAS_LOGGER.info(f"Unfreeze operators {list(index)} at layer::{name}")
                for i, _ in enumerate(layer.candidate_operators):
                    if i not in index:
                        layer.candidate_operators[i].requires_grad_(False)
                    else:
                        layer.candidate_operators[i].requires_grad_(True)
                        train_params.extend(layer.candidate_operators[i].parameters())
            else:
                raise Exception(f"Invalid layer type, receive <{type(layer)}>")
        return train_params

    def _train_one_epoch(self, current_epoch, total_epoch, optimizer, scheduler):
        assert isinstance(self.super_net, SuperNet)
        self.super_net.train()

        batch_time = AvgMeter()
        losses = AvgMeter()
        top1 = AvgMeter()
        top5 = AvgMeter()

        milestone = (len(self.train_dataloader) * self.train_ratio)
        start_timer, end_timer = _get_timer()
        epoch_iterator = tqdm.tqdm(self.train_dataloader, desc=f"Training (Epoch {current_epoch + 1}/{total_epoch})")
        for i, (datas, labels) in enumerate(epoch_iterator):
            datas = datas.to(self.device)
            labels = labels.to(self.device)

            start_timer.record()
            outputs = self.super_net(datas)
            end_timer.record()
            torch.cuda.synchronize()
            batch_time.update(start_timer.elapsed_time(end_timer))

            latency = self.predict_latency()
            memory_constrain = self.predict_hardware()
            loss = self.criterion(outputs, labels, latency, memory_constrain)
            acc1, acc5 = _accuracy(outputs, labels, (1, 5))

            losses.update(loss, datas.size(0))
            top1.update(acc1[0], datas.size(0))
            top5.update(acc5[0], datas.size(0))

            # for gradient descend
            self.super_net.zero_grad()
            loss.backward()

            # weight training
            if i < milestone:
                current_lr = optimizer.param_groups[0]['lr']
                optimizer.step()
                mode = "weight"
            else:
                current_lr = self.theta_optimizer.param_groups[0]['lr']
                self.theta_optimizer.step()
                mode = "theta"

            # print info
            if i % self.print_frequency == 0 or i + 1 == len(self.train_dataloader):
                log_info = "Train Data {}/{}\t " \
                           "Batch Time (avg:{batch_time.avg:.4f} ms)\t" \
                           "Loss (avg:{losses.avg:.4f})\t" \
                           "Top-1 acc (avg:{top1.avg:.4f})\t" \
                           "Top-5 acc (avg:{top5.avg:.4f})\t" \
                           "Learning rate {lr:.5f}\t Mode: {md}\n". \
                    format(i, len(self.train_dataloader) - 1,
                           batch_time=batch_time, losses=losses, top1=top1, top5=top5, lr=current_lr, md=mode)
                logger.NAS_LOGGER.info(log_info)

        # update lr
        scheduler.step()

        return losses.avg, top1.avg, top5.avg

    def validate(self):
        """
        batch_time: ms
        :return:
        """
        self.super_net.eval()

        batch_time = AvgMeter()
        losses = AvgMeter()
        top1 = AvgMeter()
        top5 = AvgMeter()

        start_timer, end_timer = _get_timer()
        with torch.no_grad():
            for i, (datas, labels) in enumerate(self.valid_dataloader):
                datas = datas.to(self.device)
                labels = labels.to(self.device)

                start_timer.record()
                outputs = self.super_net(datas)
                end_timer.record()
                torch.cuda.synchronize()
                batch_time.update(start_timer.elapsed_time(end_timer))

                # calculate lat, mem, loss and acc
                latency = self.predict_latency()
                memory_constrain = self.predict_hardware()
                loss = self.criterion(outputs, labels, latency, memory_constrain)
                acc1, acc5 = _accuracy(outputs, labels, (1, 5))

                losses.update(loss, datas.size(0))
                top1.update(acc1[0], datas.size(0))
                top5.update(acc5[0], datas.size(0))

                if i % self.print_frequency == 0 or i + 1 == len(self.valid_dataloader):
                    log_info = "Valid {}/{}\t " \
                               "Batch Time (avg:{batch_time.avg:.4f} ms)\t" \
                               "Loss (val:{losses.val:.4f})\t (avg:{losses.avg:.4f})\t" \
                               "Lat ({lat:.4f})\tHC ({mc:.4f})\t" \
                               "Top-1 acc (val:{top1.val:.4f})\t (avg:{top1.avg:.4f})\t" \
                               "Top-5 acc (val:{top5.val:.4f})\t (avg:{top5.avg:.4f})\n". \
                        format(i, len(self.valid_dataloader) - 1,
                               batch_time=batch_time, losses=losses, lat=latency, mc=memory_constrain, top1=top1, top5=top5)
                    logger.NAS_LOGGER.info(log_info)

        return losses.avg, top1.avg, top5.avg

    def preprocess(self):
        """
        在前面preprocess_epoch轮里，整个超级网络公平的进行训练
        :return:
        """
        if self._start_epoch >= self.preprocess_epoch:
            logger.NAS_LOGGER.info(f"Current epoch is {self._start_epoch}. Skip preprocess!")
            return

        torch.cuda.empty_cache()
        for epoch in range(self._start_epoch, self.preprocess_epoch):
            logger.NAS_LOGGER.info("\n" + ("-" * 30) + f"PreprocessEpoch: {epoch + 1}" + ("-" * 30) + "\n")
            train_loss, train_top1, train_top5 = self._train_one_epoch(epoch, self.preprocess_epoch, self.preprocess_optimizer, self.preprocess_scheduler)
            log_info = [
                f"Train {epoch + 1}/{self.preprocess_epoch}",
                "Loss (val:{train_loss:.4f})\t"
                "Top-1 acc (val:{train_top1:.3f})\t"
                "Top-5 acc (val:{train_top5:.3f})".
                format(train_loss=train_loss, train_top1=train_top1, train_top5=train_top5)
            ]
            logger.NAS_LOGGER.info("\t".join(log_info))

            if epoch % self.valid_frequency == 0 or epoch == self.preprocess_epoch - 1:
                valid_loss, valid_top1, valid_top5 = self.validate()
                # higher accuracy, update and save
                if valid_top1 > self.best_acc:
                    valid_log_info = f"Preprocess::best accuracy update from {self.best_acc:.2f} to {valid_top1:.2f} at {epoch + 1}/{self.preprocess_epoch}"
                    logger.NAS_LOGGER.info(valid_log_info)
                    self.best_acc = max(self.best_acc, valid_top1)
                    torch.save(self.super_net, self._best_model_path)
                    logger.NAS_LOGGER.info(f"Preprocess::save best model at  {epoch + 1}/{self.preprocess_epoch} to {self._best_model_path}")

            if epoch % self.save_frequency == 0 or epoch == self.preprocess_epoch - 1:
                basic_state_dict = {
                    "epoch": min(epoch + 1, self.preprocess_epoch),
                    "best_acc": self.best_acc,
                    "preprocess_optimizer": self.preprocess_optimizer.state_dict(),
                    "theta_optimizer": self.theta_optimizer.state_dict(),
                    "state_dict": self.super_net.state_dict(),
                    "done": False
                }
                torch.save(basic_state_dict, self._checkpoint_path)
                logger.NAS_LOGGER.info(f"Preprocess::basically save model at {epoch + 1}/{self.preprocess_epoch} to {self._checkpoint_path}")

        logger.NAS_LOGGER.info(f"Preprocess finished! Set _start_epoch from 0 to {self.preprocess_epoch}")
        self._start_epoch = self.preprocess_epoch

    def sample_and_train(self):
        """
        在iter_epoch × iter_numbers轮里，进行模块选择以及训练
        :return:
        """
        assert self._start_epoch >= self.preprocess_epoch, f"You should finish preprocess before sample_and_train!"
        torch.cuda.empty_cache()

        count = 0
        optimizer = None
        scheduler = None
        while self._start_epoch < self.total_epoch:
            logger.NAS_LOGGER.info("\n" + ("-" * 30) + f"Sample and train: {self._start_epoch + 1}" + ("-" * 30) + "\n")
            # sample and choose to train
            if count % self.iter_epoch == 0:
                params = self.dynamic_freeze_and_free()
                optimizer = optim.SGD(params, lr=self.iter_lr, momentum=self.iter_momentum, weight_decay=self.iter_decay)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.iter_epoch)
                gc.collect()

            # train
            train_loss, train_top1, train_top5 = self._train_one_epoch(self._start_epoch, self.total_epoch, optimizer, scheduler)
            log_info = [
                f"Train {self._start_epoch + 1}/{self.total_epoch}",
                "Loss (val:{train_loss:.4f})\t"
                "Top-1 acc (val:{train_top1:.3f})\t"
                "Top-5 acc (val:{train_top5:.3f})".
                format(train_loss=train_loss, train_top1=train_top1, train_top5=train_top5)
            ]
            logger.NAS_LOGGER.info("\t".join(log_info))

            # validate
            if self._start_epoch % self.valid_frequency == 0 or count == self.iter_epoch - 1:
                valid_loss, valid_top1, valid_top5 = self.validate()
                # higher accuracy, update and save
                if valid_top1 > self.best_acc:
                    valid_log_info = f"Sample and train::best accuracy update from {self.best_acc:.2f} to {valid_top1:.2f} at {self._start_epoch + 1}/{self.total_epoch}"
                    logger.NAS_LOGGER.info(valid_log_info)
                    self.best_acc = max(self.best_acc, valid_top1)
                    torch.save(self.super_net, self._best_model_path)
                    logger.NAS_LOGGER.info(f"Sample and train::save best model at  {self._start_epoch + 1}/{self.total_epoch} to {self._best_model_path}")

            # 这里要注意epoch，要保存最近一次存档点，存档epoch计算如下
            # ∵ current = start + count = start + 5k + (count % 5) => last store = current - (count % 5)
            if self._start_epoch % self.save_frequency == 0 or count == self.iter_epoch - 1:
                # 计算下一轮的上一个存档点
                last_epoch = (self._start_epoch + 1) - ((count + 1) % self.iter_epoch)
                basic_state_dict = {
                    "epoch": min(last_epoch, self.preprocess_epoch),
                    "best_acc": self.best_acc,
                    "preprocess_optimizer": self.preprocess_optimizer.state_dict(),
                    "theta_optimizer": self.theta_optimizer.state_dict(),
                    "state_dict": self.super_net.state_dict(),
                    "done": self._start_epoch == self.total_epoch - 1
                }
                torch.save(basic_state_dict, self._checkpoint_path)
                logger.NAS_LOGGER.info(f"Preprocess::basically save model at {self._start_epoch + 1}/{self.total_epoch} to {self._checkpoint_path}")

            count += 1
            self._start_epoch += 1

    def train(self):
        """
        分为两个阶段：
        1） preprocess对超级网络进行公平训练
        2） sample_and_train对超级网络模块进行选择以及训练
        :return:
        """
        self.preprocess()
        self.sample_and_train()

    def warm_up(self):
        """
        预热，用于推理和验证
        :return:
        """
        torch.cuda.empty_cache()
        inputs, labels = next(iter(self.valid_dataloader))
        logger.NAS_LOGGER.info("Start to warm up at {}".format(datetime.now()))
        random_input = torch.rand(inputs.shape)
        random_input = random_input.to(self.device)
        for i in range(5):
            logger.NAS_LOGGER.info("Warm up {}/5 at {}".format(i + 1, datetime.now()))
            outs = self.super_net(random_input)

    def get_final_net(self, force=None):
        """
        get the final net after searching and training, and save the final net
        :return:
        """
        if self.final_net is None or force:
            assert isinstance(self.super_net, SuperNet)
            latency, memory = self.estimator.predict_final(self.super_net)
            order_dict = self.super_net.generate_final_net()
            self.final_net = FinalNet(ordered_dict=order_dict, lat=latency, mem=memory)
            # 这里直接将整个网络存下来，里面的变量也会自动存下来的
            torch.save(self.final_net, self._final_net_path)
            logger.NAS_LOGGER.info(f"Save final net to {self._final_net_path}")
        return self.final_net

    def analyze(self):
        """
        1) get_test_input_data
        2) validate and calculate the top1 and top5
        3) show runtime, hardware usage
        :return:
        """
        start_timer, end_timer = _get_timer()
        batch_time = AvgMeter()
        losses = AvgMeter()
        top1 = AvgMeter()
        top5 = AvgMeter()

        # show final net first: net structure, lat, mem
        if self.final_net is None:
            self.get_final_net()
        info = "Analyze::Latency:{lat}ms\tMemory:{usage}MB\nModelStructure:\n{net}".format(
            lat=self.final_net.latency, usage=self.final_net.memory,
            net=self.final_net
        )
        logger.NAS_LOGGER.info(info)

        self.warm_up()

        current = 1
        for data, label in tqdm.tqdm(self.valid_dataloader):
            datas = data.to(self.device)
            labels = label.to(self.device)

            start_timer.record()
            outputs = self.final_net(datas)
            end_timer.record()
            torch.cuda.synchronize()
            batch_time.update(start_timer.elapsed_time(end_timer))

            latency = self.predict_latency()
            memory_constrain = self.predict_hardware()
            loss = self.criterion(outputs, labels, latency, memory_constrain)
            acc1, acc5 = _accuracy(outputs, labels, (1, 5))

            losses.update(loss, datas.size(0))
            top1.update(acc1[0], datas.size(0))
            top5.update(acc5[0], datas.size(0))

            if current % self.print_frequency == 0 or current == len(self.valid_dataloader):
                total_parameters = sum(p.numel() for p in self.final_net.parameters()) * 4 / (1024 ** 2)
                log_info = "Analyzing {}/{}\t " \
                           "Infer Time (avg:{batch_time.avg:.4f} ms)\t" \
                           "Parameters ({tps:.4f} MB)\t" \
                           "Loss (val:{losses.val:.4f})\t (avg:{losses.avg:.4f})\t" \
                           "Top-1 acc (val:{top1.val:.4f})\t (avg:{top1.avg:.4f})\t" \
                           "Top-5 acc (val:{top5.val:.4f})\t (avg:{top5.avg:.4f})\n". \
                    format(current, len(self.valid_dataloader),
                           batch_time=batch_time, tps=total_parameters, losses=losses, top1=top1, top5=top5)
                logger.NAS_LOGGER.info(log_info)

            current += 1

    def show_supernet(self):
        logger.NAS_LOGGER.info(self.super_net.show())


if __name__ == '__main__':
    # 补充实验部分
    logger.NAS_LOGGER = logger.setup_logger("/home/why/WhyEnv/ImageNetAdamFinal/log")
    source_path = "/home/why/WhyEnv/ImageNetAdamFinal"
    manager = AnalysisRunManager(source_path, "imagenet", "/home/why/WhyEnv/DataSet/imagenet-100", 128, pre_load=False)
    # load models in /home/why/WhyEnv/models
    models_dir = "/home/why/WhyEnv/models"
    model_dict = dict()
    for name in os.listdir(models_dir):
        if not name.endswith(".pth"):
            continue
        path = os.path.join(models_dir, name)
        model = torch.load(path)
        model_dict[name.split(".")[0]] = model
    manager.set_model_dict(model_dict)
    manager.final_optimize3()
    manager.estimate()


