from configs.config_dataclass import *

__all__ = ["wandb_cfg_dict", "LR_cfg_dict", "NN_cfg_dict", "layer_cfg_dict"]

NN_cfg_dict = {
    "log_id": "test",
    "dataset": "cifar10",
    "train_split": "train",
    "model": "vgg16_bn",
    "workers": 2,
    "epochs": 1,
    "start_epoch": 0,
    "batch_size": 1024,
    "optimizer": "sgd",
    "scheduler": "multi",
    "lr": 0.01,
    "lr_decay": "10,15,25",
    "weight_decay": 1e-5,
    "print_freq": 20,
    "pretrain": "/root/IRIS_MetaNet_origin/vgg93perbest.tar",
    "weight_bit_width": "32",
    "act_bit_width": "32",
    "is_training": 'F',
    "is_calibrate": 'F',
    "cal_bit_width": "0",
    "quant" : "quant_dorefa",
    "CONV": "conv2dQ",
    "inject_variation": True
}

wandb_cfg_dict = {
    "wandb_enabled": True,
    "key": "3914394dc58eb9d88ed682d03779576f35627195",
    "entity": "tentacion990125-sungkyunkwan-university",
    "project": "vgg16_test_sigma5_batch1024",
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