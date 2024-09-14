# -*- coding:utf8 -*_

class AvgMeter(object):
    def __init__(self):
        """
        摘抄自ProxyLess， 他这个结构可以存各种小结构，好像也挺方便的，先写下来用用
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0
        self.min = 100

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0
        self.min = 100

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.max = max(self.max, val)
        self.min = min(self.min, val)

