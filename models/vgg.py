import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from quan_ops import *


__all__ = ['vgg16', 'vgg16_bn', 'vgg16q', 'vgg16q_bn']

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

class Activate(nn.Module):
    """양자화된 활성화 함수 (ReLU + Quantization)"""
    def __init__(self, abit_list, quantize=True):
        super(Activate, self).__init__()
        self.abit_list = abit_list
        self.abit = self.abit_list[-1]
        self.acti = nn.ReLU(inplace=True)
        self.quantize = quantize
        if self.quantize:
            self.quan = activation_quantize_fn(self.abit_list)

    def forward(self, x):
        if self.abit == 32:
            x = self.acti(x)
        else:
            x = torch.clamp(x, 0.0, 1.0)
        if self.quantize:
            x = self.quan(x)
        return x

# class PreActBasicBlockQ(nn.Module):
#     """양자화된 VGG16 기본 블록"""
#     def __init__(self, wbit_list, abit_list, in_channels, out_channels, batch_norm=False):
#         super(PreActBasicBlockQ, self).__init__()
#         self.wbit_list = wbit_list
#         self.abit_list = abit_list
#         self.batch_norm = batch_norm
#         self.in_channels = in_channels
#         self.out_channels = out_channels

#         self.bn = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
#         self.act = Activate(self.abit_list)

#     def forward(self, x):
#         # forward() 실행 시 현재 wbit 가져오기
#         wbit = getattr(self, 'wbit', self.wbit_list[-1])
#         abit = getattr(self, 'abit', self.abit_list[-1])

#         # forward()에서 Conv2d를 동적으로 생성
#         Conv2d = conv2d_quantize_fn([wbit], [abit])
#         conv = Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1, bias=False)

#         out = conv(x)  # wbit이 매번 반영됨
#         out = self.bn(out)
#         out = self.act(out)
#         return out


# class PreActVgg16(nn.Module):
#     """양자화된 VGG16 모델"""
#     def __init__(self, wbit_list, abit_list, num_classes=10, batch_norm=True):
#         super(PreActVgg16, self).__init__()
#         self.wbit_list = wbit_list
#         self.abit_list = abit_list
#         self.num_classes = num_classes

#         cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 
#                512, 512, 512, 'M', 512, 512, 512, 'M']

#         self.layers = self._make_layers(cfg, batch_norm)

#         self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
#         self.fc = nn.Sequential(
#             nn.Linear(512 * 2 * 2, 1024),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(1024, 512),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(512, num_classes),
#         )

#     def _make_layers(self, cfg, batch_norm):
#         layers = []
#         in_channels = 3
#         for v in cfg:
#             if v == 'M':
#                 layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#             else:
#                 layers += [PreActBasicBlockQ(self.wbit_list, self.abit_list, in_channels, v, batch_norm)]
#                 in_channels = v
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = self.layers(x)
#         out = self.avgpool(out)
#         out = torch.flatten(out, 1)
#         out = self.fc(out)
#         return out

# class VGG(nn.Module):

#     def __init__(self, features, num_classes=10, init_weights=True):
#         super(VGG, self).__init__()
#         self.features = features
#         self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
#         self.classifier = nn.Sequential(
#             nn.Linear(512 * 2 * 2, 1024),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(1024, 512),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(512, num_classes),
#         )
#         if init_weights:
#             self._initialize_weights()

#     def forward(self, x):
#         x = self.features(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)


# def make_layers(cfg, batch_norm=False):
#     layers = []
#     in_channels = 3
#     for v in cfg:
#         if v == 'M':
#             layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#         else:
#             conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
#             if batch_norm:
#                 layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
#             else:
#                 layers += [conv2d, nn.ReLU(inplace=True)]
#             in_channels = v
#     return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

#New
class ConvBNReLU(nn.Module):
    """양자화된 Conv2D + ReLU + BatchNorm + Dropout 블록 (TensorFlow 구조 반영)"""
    def __init__(self, wbit_list, abit_list, in_channels, out_channels, drop_rate=0.4, apply_dropout=True):
        super(ConvBNReLU, self).__init__()
        self.wbit_list = wbit_list
        self.abit_list = abit_list

        wbit = wbit_list[-1]
        abit = abit_list[-1]

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.drop_rate = drop_rate
        self.apply_dropout = apply_dropout
        self.dropout = nn.Dropout2d(drop_rate) if apply_dropout else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.apply_dropout:
            x = self.dropout(x)
        return x

#New
class VGG16_BN(nn.Module):
    """TensorFlow 구조를 반영한 VGG16Q 모델 (양자화 포함)"""
    def __init__(self, wbit_list, abit_list, num_classes=10):
        super(VGG16_BN, self).__init__()
        self.wbit_list = wbit_list
        self.abit_list = abit_list

        # Feature Extractor
        self.features = nn.Sequential(
            ConvBNReLU(wbit_list, abit_list, 3, 64, apply_dropout=False),
            ConvBNReLU(wbit_list, abit_list, 64, 64, apply_dropout=False),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ConvBNReLU(wbit_list, abit_list, 64, 128, apply_dropout=False),
            ConvBNReLU(wbit_list, abit_list, 128, 128, apply_dropout=False),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ConvBNReLU(wbit_list, abit_list, 128, 256, apply_dropout=False),
            ConvBNReLU(wbit_list, abit_list, 256, 256, apply_dropout=False),
            ConvBNReLU(wbit_list, abit_list, 256, 256, apply_dropout=False),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ConvBNReLU(wbit_list, abit_list, 256, 512, apply_dropout=False),
            ConvBNReLU(wbit_list, abit_list, 512, 512, apply_dropout=False),
            ConvBNReLU(wbit_list, abit_list, 512, 512, apply_dropout=False),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ConvBNReLU(wbit_list, abit_list, 512, 512, apply_dropout=False),
            ConvBNReLU(wbit_list, abit_list, 512, 512, apply_dropout=False),
            ConvBNReLU(wbit_list, abit_list, 512, 512, apply_dropout=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Fully Connected Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(512 * 1 * 1, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


#New
class ConvBNActivate(nn.Module):
    """양자화된 Conv2D + ReLU + BatchNorm + Dropout 블록 (TensorFlow 구조 반영)"""
    def __init__(self, wbit_list, abit_list, in_channels, out_channels, drop_rate=0.4, apply_dropout=True):
        super(ConvBNActivate, self).__init__()
        self.wbit_list = wbit_list
        self.abit_list = abit_list

        wbit = wbit_list[-1]
        abit = abit_list[-1]
        Conv2d = conv2d_quantize_fn(self.wbit_list, self.abit_list)

        self.conv = Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.relu = Activate(self.abit_list)
        self.bn = nn.BatchNorm2d(out_channels)
        self.drop_rate = drop_rate
        self.apply_dropout = apply_dropout
        self.dropout = nn.Dropout2d(drop_rate) if apply_dropout else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.apply_dropout:
            x = self.dropout(x)
        return x



class VGG16Q_BN(nn.Module):
    """TensorFlow 구조를 반영한 VGG16Q 모델 (양자화 포함)"""
    def __init__(self, wbit_list, abit_list, num_classes=10):
        super(VGG16Q_BN, self).__init__()
        self.wbit_list = wbit_list
        self.abit_list = abit_list

        # Feature Extractor
        self.features = nn.Sequential(
            ConvBNActivate(wbit_list, abit_list, 3, 64, apply_dropout=False),
            ConvBNActivate(wbit_list, abit_list, 64, 64, apply_dropout=False),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ConvBNActivate(wbit_list, abit_list, 64, 128, apply_dropout=False),
            ConvBNActivate(wbit_list, abit_list, 128, 128, apply_dropout=False),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ConvBNActivate(wbit_list, abit_list, 128, 256, apply_dropout=False),
            ConvBNActivate(wbit_list, abit_list, 256, 256, apply_dropout=False),
            ConvBNActivate(wbit_list, abit_list, 256, 256, apply_dropout=False),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ConvBNActivate(wbit_list, abit_list, 256, 512, apply_dropout=False),
            ConvBNActivate(wbit_list, abit_list, 512, 512, apply_dropout=False),
            ConvBNActivate(wbit_list, abit_list, 512, 512, apply_dropout=False),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ConvBNActivate(wbit_list, abit_list, 512, 512, apply_dropout=False),
            ConvBNActivate(wbit_list, abit_list, 512, 512, apply_dropout=False),
            ConvBNActivate(wbit_list, abit_list, 512, 512, apply_dropout=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Fully Connected Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(512 * 1 * 1, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

#New
def _vgg_bn(arch, pretrained, progress, **kwargs):
    """
    Helper function to create a VGG16 with Batch Normalization.
    """
    model = VGG16_BN(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model

def vgg16(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)

def vgg16_bn(pretrained=False, progress=True, **kwargs):
    """
    Returns the VGG16 model with Batch Normalization.
    """
    return _vgg_bn('vgg16_bn', pretrained, progress, **kwargs)

def vgg16q(wbit_list, abit_list, num_classes=10, batch_norm=True):
    """양자화된 VGG16 모델을 생성하는 함수"""
    return PreActVgg16(wbit_list, abit_list, num_classes, batch_norm)

#New
def vgg16q_bn(wbit_list, abit_list, num_classes=10):
    """Batch Normalization을 포함한 양자화된 VGG16 모델 생성"""
    return VGG16Q_BN(wbit_list, abit_list, num_classes)