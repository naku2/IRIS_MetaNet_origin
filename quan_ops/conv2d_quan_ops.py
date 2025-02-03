import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from . import quantizer
from quan_ops.acti_quan_ops import *
from quan_ops.bn_quan_ops import *
from configs.config import *

args = None

def str_to_function(functionname):
    return getattr(sys.modules[__name__], functionname)

def conv2d_quantize_fn(wbit_list, abit_list, choose_CONV=None):
    if choose_CONV:
        quan_fn = str_to_function(choose_CONV)
    else:
        quan_fn = str_to_function(args.CONV)
    return quan_fn(wbit_list, abit_list)

class Conv2d_Q(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        super(Conv2d_Q, self).__init__(*kargs, **kwargs)

def conv2dQ(wbit_list, abit_list):
    class Conv2d_Q_(Conv2d_Q):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                     bias=True):
            super(Conv2d_Q_, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                            bias)
            self.wbit_list = wbit_list
            self.wbit = self.wbit_list[-1]
            self.quantize_fn = quantizer.weight_quantize_fn(self.wbit_list)

        def forward(self, input, order=None):
            weight_q = self.quantize_fn(self.weight)
            return F.conv2d(input, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)

    return Conv2d_Q_


class Conv2d_Q_LR(nn.Module):
    def __init__(self, *kargs, **kwargs):
        super(Conv2d_Q_LR, self).__init__(*kargs, **kwargs)

def conv2dQ_lr(wbit_list, abit_list):
    class Conv2d_Q_LR_(Conv2d_Q_LR):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
            super(Conv2d_Q_LR_, self).__init__()
            self.stride = stride
            self.rank = in_channels // args.rank
            self.groups = args.groups
            # print(LR_cfg.rank, LR_cfg.groups)
            norm_layer = batchnorm_fn(wbit_list)
            self.act = activation_quantize_fn(abit_list)

            conv2d = conv2dQ(wbit_list, abit_list)  
            self.conv_R = conv2d(in_channels=in_channels, 
                                out_channels=self.rank*self.groups, 
                                kernel_size=kernel_size, 
                                stride=stride, 
                                padding=padding, 
                                bias=False, 
                                groups=self.groups)
            self.bn = norm_layer(self.rank*self.groups)
            self.conv_L = conv2d(in_channels=self.rank*self.groups, 
                                out_channels=out_channels, 
                                kernel_size=1, 
                                stride=1, 
                                padding=0, 
                                bias=False)
            

        def forward(self, x):
            x = self.conv_R(x)
            x = self.bn(x)
            x = self.act(x)
            x = self.conv_L(x)
            return x
        
    return Conv2d_Q_LR_




class Conv2d_NoQ(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(Conv2d_NoQ, self).__init__(*args, **kwargs)

def conv2d_noQ(wbit_list, abit_list):
    class Conv2d_NoQ_(Conv2d_NoQ):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                     bias=True):
            super(Conv2d_NoQ_, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                            bias)
          

        def forward(self, input):
            return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    return Conv2d_NoQ_