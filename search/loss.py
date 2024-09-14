# -*- coding:utf8 -*_

import torch.nn as nn


class ComprehensiveCriterion(nn.Module):
    def __init__(self, ls_fn=None):
        """

        :param ls_fn: other loss function instead of ce
        :return:
        """
        super().__init__()
        self.ce = nn.CrossEntropyLoss() if ls_fn is None else ls_fn

    def forward(self, predict, label, latency, hardware, *args, **kwargs):
        """
        Loss = LOSS_FN + latency + hardware
        :param predict:
        :param label:
        :param latency: ms
        :param hardware: MB
        :return:
        """
        if isinstance(self.ce, nn.CrossEntropyLoss):
            ce_loss = self.ce(predict, label)
        else:
            ce_loss = self.ce(predict, label, *args, **kwargs)
        loss = ce_loss + latency + hardware
        return loss



