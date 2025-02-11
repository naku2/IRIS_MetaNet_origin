from configs.config_dataclass import *

__all__ = ["wandb_cfg_dict", "LR_cfg_dict", "NN_cfg_dict", "layer_cfg_dict"]

NN_cfg_dict = {
    "log_id": "train",
    "dataset": "cifar10",
    "train_split": "train",
    "model": "resnet20q",
    "workers": 2,
    "epochs": 150,
    "start_epoch": 0,
    "batch_size": 128,
    "optimizer": "adam",
    "scheduler": "multi",
    "lr": 0.001,
    "lr_decay": "100,140,180,220",
    "weight_decay": 0.0,
    "print_freq": 20,
    "pretrain": None,
    "resume": None,
    "weight_bit_width": "2,8,32",
    "act_bit_width": "4,4,32",
    "is_training": 'T',
    "is_calibrate": 'F',
    "cal_bit_width": "0",
    "quant" : "truncquant",
    "CONV": "conv2dQ",
    "bn_type": "bn",
    "biascorr": 'T'
}

LR_cfg_dict = {
    "LR_enabled": False,
    "rank": 4,
    "groups": 4
}

wandb_cfg_dict = {
    "wandb_enabled": False,
    "key": "3914394dc58eb9d88ed682d03779576f35627195",
    "entity": "tentacion990125-sungkyunkwan-university",
    "project": "vgg16_test",
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

layer_cfg_dict = {}

