import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys
from configs.config import *

from variation_injection import apply_variations

def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)()

class quant_dorefa(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k):
      n = float(2**k - 1)
      out = torch.round(input * n) / n
      return out

    @staticmethod
    def backward(ctx, grad_output):
      grad_input = grad_output.clone()
      return grad_input, None

class truncquant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k):
        n = float(2**k - 1)

        out = torch.floor(input * (n+1))
        out = torch.clamp(out, max=n)
        out = out/n

        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None
    
class truncquant_inference(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k):
        n = float(2**k - 1)
        nb = float(2**8 - 1)

        out = torch.floor(input * (nb+1))
        out = torch.clamp(out, max=nb).type(torch.uint8) >> (8-k)
        # # Variation Injection (wbit 조건 추가)
        # if hasattr(NN_cfg, "inject_variation") and NN_cfg.inject_variation:
        #     if k != 32:  # wbit이 32가 아닐 때만 variation 적용
        #         out = apply_variations(out, sigma=0.5, wbit=k) 
        # out = out/n


        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

class weight_quantize_fn(nn.Module):
    def __init__(self, bit_list):
        super(weight_quantize_fn, self).__init__()
        self.bit_list = bit_list
        self.wbit = self.bit_list[-1]
        self.quant = str_to_class(NN_cfg.quant)
        assert self.wbit <= 8 or self.wbit == 32

    def forward(self, x):
        if NN_cfg.biascorr == 'T':
            if self.wbit == 32:
                E = torch.mean(torch.abs(x)).detach()
                weight = torch.tanh(x)
                E32 = torch.mean(weight).detach()
                SD32 = torch.norm((weight-E32), p=2)
            
                weight = weight / torch.max(torch.abs(weight))
                weight_q = weight * E

                Eq = torch.mean(weight_q).detach()
                SDq = torch.norm((weight_q-Eq), p=2)
                eps = SD32/SDq
                weight_q = (weight_q +(E32-Eq))*eps
                # weight_q = (weight_q-Eq)*eps+E32

            else:
                E = torch.mean(torch.abs(x)).detach()
                weight = torch.tanh(x)
                E32 = torch.mean(weight).detach()
                SD32 = torch.norm((weight-E32), p=2)

                weight = weight / 2 / torch.max(torch.abs(weight)) + 0.5
                weight_q = 2 * self.quant.apply(weight, self.wbit) - 1
                weight_q = weight_q * E

                Eq = torch.mean(weight_q).detach()
                SDq = torch.norm((weight_q-Eq), p=2)
                eps = SD32/SDq
                weight_q = (weight_q+(E32-Eq))*eps
                # weight_q = (weight_q-Eq)*eps+E32

        else:
            if self.wbit == 32:
                E = torch.mean(torch.abs(x)).detach()
                weight = torch.tanh(x)
                weight = weight / torch.max(torch.abs(weight))
                weight_q = weight * E
            else:
                E = torch.mean(torch.abs(x)).detach()
                weight = torch.tanh(x)
                weight = weight / 2 / torch.max(torch.abs(weight)) + 0.5
                weight_q = 2 * self.quant.apply(weight, self.wbit) - 1
                weight_q = weight_q * E
            
        return weight_q