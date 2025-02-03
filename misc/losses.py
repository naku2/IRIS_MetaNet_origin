import torch


class CrossEntropyLossSoft(torch.nn.modules.loss._Loss):
    """ inplace distillation for image classification.
    Refer to https://github.com/JiahuiYu/slimmable_networks/blob/master/utils/loss_ops.py
    """
    def forward(self, output, target):
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
        target = target.unsqueeze(1)
        output_log_prob = output_log_prob.unsqueeze(2)
        cross_entropy_loss = -torch.bmm(target, output_log_prob)
        cross_entropy_loss = cross_entropy_loss.mean()
        return cross_entropy_loss


import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import reduce
from quan_ops.conv2d_quan_ops import Conv2d_Q


# KurtosisLoss
class KurtosisLossCalc:
    def __init__(self, weight_tensor, kurtosis_target=3, k_mode='avg'):
        self.kurtosis_loss = 0
        self.kurtosis = 0
        self.weight_tensor = weight_tensor
        self.k_mode = k_mode
        self.kurtosis_target = kurtosis_target

    def fn_regularization(self):
        return self.kurtosis_calc()

    def kurtosis_calc(self):
        mean_output = torch.mean(self.weight_tensor)
        std_output = torch.std(self.weight_tensor)
        kurtosis_val = torch.mean((((self.weight_tensor - mean_output) / std_output) ** 4))
        self.kurtosis_loss = (kurtosis_val - self.kurtosis_target) ** 2
        self.kurtosis = kurtosis_val

        if self.k_mode == 'avg':
            self.kurtosis_loss = torch.mean((kurtosis_val - self.kurtosis_target) ** 2)
            self.kurtosis = torch.mean(kurtosis_val)
        elif self.k_mode == 'max':
            self.kurtosis_loss = torch.max((kurtosis_val - self.kurtosis_target) ** 2)
            self.kurtosis = torch.max(kurtosis_val)
        elif self.k_mode == 'sum':
            self.kurtosis_loss = torch.sum((kurtosis_val - self.kurtosis_target) ** 2)
            self.kurtosis = torch.sum(kurtosis_val)


def KurtosisLoss(model):
    KurtosisList = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, Conv2d_Q):
            w_kurt_inst = KurtosisLossCalc(m.weight)
            w_kurt_inst.fn_regularization()
            KurtosisList.append(w_kurt_inst.kurtosis_loss)
    del KurtosisList[0]
    del KurtosisList[-1]
    w_kurtosis_loss = reduce((lambda a, b: a + b), KurtosisList) / len(KurtosisList)
    w_kurtosis_regularization = w_kurtosis_loss
    return w_kurtosis_regularization

