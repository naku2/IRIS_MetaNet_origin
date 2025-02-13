from configs.config_dataclass import *

__all__ = ["wandb_cfg_dict", "LR_cfg_dict", "NN_cfg_dict", "layer_cfg_dict"]

NN_cfg_dict = {
    "log_id": "test",
    "dataset": "imagenet",
    "imagenet_path": "/home/a2jinhee/vol-1",
    "train_split": "train",
    "model": "preresnet18q",
    "workers": 8,
    "epochs": 1,
    "start_epoch": 0,
    "batch_size": 512,
    "optimizer": "sgd",
    "scheduler": "multi",
    "lr": 0.3,
    "lr_decay": "45,60,70",
    "weight_decay": 1e-5,
    "print_freq": 20,
    "pretrain": None,
    "weight_bit_width": "1,2,4,8,32",
    "act_bit_width": "1,2,4,8,32",
    "is_training": 'F',
    "is_calibrate": 'F',
    "cal_bit_width": "0",
    "quant" : "quant_dorefa",
    "CONV": "conv2dQ",
}

wandb_cfg_dict = {
    "wandb_enabled": False,
    "key": "028a6c9f793dd46f8ead875b50784dde31c413be",
    "entity": "a2jinhee",
    "project": "rn18_imagenet",
    "resume": None,
    "pretrain": "a2jinhee/rn18_imagenet/tup8gxuj",
    "sweep_enabled": False,
    "sweep_config": {
        "method": "grid",
        "metric": {"goal": "maximize", "name": "Best_score"}
    },
    "sweep_count": 10000,
    "sweep_id": None
}

layer_cfg_dict = {}
LR_cfg_dict = {}

