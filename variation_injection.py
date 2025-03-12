import torch
import numpy as np
import torch.nn as nn


def inject_variations(weights, sigma):
    """
    Apply log-normal variations to the weights.
    w = w_nominal * e^θ where θ ~ N(0, σ²)
    """
    # 정규분포 샘플링 (θ ~ N(0, σ²))
    theta = torch.randn_like(weights) * sigma

    # 지수 변환하여 w = w_nominal * e^θ 생성
    variations = torch.exp(theta)

    # 변동된 가중치 생성
    noisy_weights = weights * variations
    return noisy_weights

# def apply_variations(model, sigma):
#     """
#     Apply variations to all Conv2d and Linear layers.
#     """
#     with torch.no_grad():
#         for name, layer in model.named_modules():
#             if isinstance(layer, (nn.Conv2d, nn.Linear)):
#                 layer.weight.data = inject_variations(layer.weight.data, sigma=sigma)

#여긴 양자화 하고 쓰기
def apply_variations(model, sigma):
    """
    Apply variations to quantized weights at tensor level.
    """
    with torch.no_grad():
        for name, layer in model.named_modules():
            if isinstance(layer, nn.Linear):
                layer.weight.data = inject_variations(layer.weight.data, sigma=sigma)