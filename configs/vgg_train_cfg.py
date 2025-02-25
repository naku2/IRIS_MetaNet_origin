from configs.config_dataclass import *

__all__ = ["wandb_cfg_dict", "LR_cfg_dict", "NN_cfg_dict", "layer_cfg_dict"]

NN_cfg_dict = {
    "log_id": "train",
    "dataset": "cifar10",
    "train_split": "train",
    "model": "vgg16q_bn",
    "workers": 2,
    "epochs": 250,
    "start_epoch": 0,
    "batch_size": 512,
    "optimizer": "sgd",
    "scheduler": "new",
    "lr": 0.01,
    "lr_decay": "20,40,60",
    "weight_decay": 1e-6,
    "print_freq": 20,
    "pretrain": "/root/IRIS_MetaNet_origin/results/train_20250225_070120/ckpt/model_latest.pth.tar",
    "weight_bit_width": "8, 32",
    "act_bit_width": "4, 32",
    "is_training": 'T',
    "is_calibrate": 'F',
    "cal_bit_width": "0",
    "quant" : "quant_dorefa",
    "CONV": "conv2dQ",
    "inject_variation": False
}

wandb_cfg_dict = {
    "wandb_enabled": True,
    "key": "3914394dc58eb9d88ed682d03779576f35627195",
    "entity": "tentacion990125-sungkyunkwan-university",
    "project": "vgg16_train",
    "resume": None,
    "pretrain": None,
    "sweep_enabled": False,
    "sweep_config": {
        "method": "grid",
        "metric": {"goal": "maximize", "name": "Best_score"}
    },
    "sweep_count": 10000,
    "sweep_id": None
}

LR_cfg_dict = {}
layer_cfg_dict = {}