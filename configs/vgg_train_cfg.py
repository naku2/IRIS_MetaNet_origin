from configs.config_dataclass import *

__all__ = ["wandb_cfg_dict", "LR_cfg_dict", "NN_cfg_dict", "layer_cfg_dict"]

NN_cfg_dict = {
    "log_id": "train",
    "dataset": "cifar10",
    "train_split": "train",
    "model": "vgg16q",
    "workers": 2,
    "epochs": 150,
    "start_epoch": 0,
    "batch_size": 128,
    "optimizer": "sgd",
    "scheduler": "multi",
    "lr": 0.01,
    "lr_decay": "10,15,25",
    "weight_decay": 1e-5,
    "print_freq": 20,
    "pretrain": None,
    "weight_bit_width": "1,2,4,8,32",
    "act_bit_width": "4,4,4,4,32",
    "is_training": 'T',
    "is_calibrate": 'F',
    "cal_bit_width": "0",
    "quant" : "truncquant",
    "CONV": "conv2dQ",
    "inject_variation": True
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

LR_cfg_dict = {}
layer_cfg_dict = {}