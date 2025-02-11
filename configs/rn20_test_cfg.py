from configs.config_dataclass import *

__all__ = ["wandb_cfg_dict", "LR_cfg_dict", "NN_cfg_dict", "layer_cfg_dict"]

NN_cfg_dict = {
    "log_id": "test",
    "dataset": "cifar10",
    "train_split": "train",
    "model": "resnet20q",
    "workers": 2,
    "epochs": 1,
    "start_epoch": 0,
    "batch_size": 128,
    "optimizer": "adam",
    "scheduler": "multi",
    "lr": 0.001,
    "lr_decay": "100,140,180,220",
    "weight_decay": 0.0,
    "print_freq": 20,
    "pretrain": "/root/IRIS_MetaNet_origin-1/results/RN20/train_20250211_110014/ckpt/model_latest.pth.tar",
    "resume": None,
    "weight_bit_width": "1,2,3,4,5,6,7,8,32",
    "act_bit_width": "4,4,4,4,4,4,4,4,32",
    "is_training": 'F',
    "is_calibrate": 'F',
    "cal_bit_width": "0",
    "quant": "truncquant_inference",
    "CONV": "conv2dQ",
    "bn_type": "bn",
    "biascorr": 'T',
    "inject_variation": False
}

LR_cfg_dict = {
    "LR_enabled": False,
    "rank": 4,
    "groups": 8
}

wandb_cfg_dict = {
    "wandb_enabled": True,
    "key": "3914394dc58eb9d88ed682d03779576f35627195",
    "entity": "tentacion990125-sungkyunkwan-university",
    "project": "vgg16_test",
    "pretrain": None,
    "sweep_enabled": False,
    "sweep_config": {
        "name": NN_cfg_dict["log_id"],
        "method": "grid",
        "metric": {"goal": "maximize", "name": "Best_score"}
    },
    "sweep_count": 10000,
    "sweep_id": None
}

layer_cfg_dict = {}
