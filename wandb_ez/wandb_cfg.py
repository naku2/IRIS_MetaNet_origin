__all__ = ["wandb_cfg"]

wandb_cfg = {
    "key": "e0c11d3ff2bee4c8775ba05863038fdac671c043",
    "project": "wandb_plugin_dev_test",
    "sweep": True,
    "sweep_param": ["bit_width_list"],
    "sweep_config": {
        "method": "grid",
        "metric": {"goal": "maximize", "name": "Best_score"}
    },
    "sweep_count": 5,
    "sweep_id": None
}