import torch

# 모델 파라미터를 분리하여, FC Layer에만 더 강한 weight decay 적용
fc_weight_decay = 5e-4  # Fully Connected Layer에 적용할 Weight Decay

def get_optimizer_config(model, name, lr, weight_decay):
    if name == 'sgd':
        optimizer = torch.optim.SGD([
        # Conv2D에 L2 Regularization 적용 (0.0005)
        {'params': [param for name, param in model.named_parameters() if 'features' in name], 'weight_decay': 0.0005},

        # classifier.2 (dense1)에는 강한 weight decay 적용 (0.0005)
        {'params': [param for name, param in model.named_parameters() if 'classifier.2' in name], 'weight_decay': 0.0005},

        # classifier.4 (BatchNorm)에는 decay 적용 X
        {'params': [param for name, param in model.named_parameters() if 'classifier.4' in name], 'weight_decay': 0.0},

        # classifier.6 (dense2)에는 weight_decay=0.0 (적용 안함)
        {'params': [param for name, param in model.named_parameters() if 'classifier.6' in name], 'weight_decay': 0.0},
    ], lr=0.1, momentum=0.9)
    elif name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer


def get_lr_scheduler(name, optimizer, lr_decay, last_epoch=-1):
    if name == 'multi':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_decay, gamma=0.1, last_epoch=last_epoch)
    elif name == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    elif name == 'new':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(range(20, 250, 20)), gamma=0.3, last_epoch=last_epoch)
    return lr_scheduler