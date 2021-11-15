import hydra
from omegaconf import DictConfig

def build_env(experiment_info: DictConfig):
    custom_env_cfg = DictConfig(dict())
    custom_env_cfg["_target_"] = experiment_info.env._target_
    custom_env_cfg["args"] = experiment_info.env.args

    custom_env = hydra.utils.instantiate(custom_env_cfg)
    return custom_env