import torch
import numpy as np
import torch.nn as nn


# def inject_variations(weights, sigma):
#     """
#     Apply log-normal variations to the weights.
#     w = w_nominal * e^θ where θ ~ N(0, σ²)
#     """
#     # 정규분포 샘플링 (θ ~ N(0, σ²))
#     theta = torch.randn_like(weights) * sigma

#     # 지수 변환하여 w = w_nominal * e^θ 생성
#     variations = torch.exp(theta)

#     # 변동된 가중치 생성
#     noisy_weights = weights * variations
#     return noisy_weights

# def apply_variations(model, sigma):
#     """
#     Apply variations to all Conv2d and Linear layers.
#     """
#     with torch.no_grad():
#         for name, layer in model.named_modules():
#             if isinstance(layer, (nn.Conv2d, nn.Linear)):
#                 layer.weight.data = inject_variations(layer.weight.data, sigma=sigma)

#여긴 양자화 하고 쓰기
def inject_variations(weights, sigma, scale_factor=None):
    """
    Inject variations into the weights (tensor level).
    """
    # 정규분포 샘플링 (θ ~ N(0, σ²))
    theta = torch.randn_like(weights) * sigma

    # 지수 변환하여 w = w_nominal * e^θ 생성
    variations = torch.exp(theta)

    # 변동량을 곱하여 variation 주입
    noisy_weights = weights * (1 + scale_factor * (variations - 1))
    return noisy_weights


def apply_variations(weights, sigma, wbit):
    """
    Apply variations to quantized weights at tensor level.
    기존 모델 코드와 일치하는 조건 적용.
    """
    scale_factor = weights.abs().max() / (2 ** (wbit - 1) - 1)

    # 양자화용 스케일팩터
    max_n = 2**wbit - 1  # MAX_N 값 (양자화 범위 최대값)
    e_abs_w = weights.to(dtype=torch.int32).abs().sum() // weights.numel()  # 정수 연산을 활용한 평균 계산

    scale_factor = e_abs_w / max_n    
    
    weights = inject_variations(weights, sigma=sigma, scale_factor=scale_factor)

    return weights