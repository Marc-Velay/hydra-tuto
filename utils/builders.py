import hydra
from omegaconf import DictConfig

from datetime import datetime

from models.base import BaseModel



def build_model(
    experiment_info: DictConfig, hyper_params: DictConfig, architecture: DictConfig
):
    """Build agent from DictConfigs via hydra.utils.instantiate()"""
    model_cfg = DictConfig(dict())
    model_cfg["_target_"] = experiment_info.agent
    model_cfg["experiment_info"] = experiment_info
    time_info = datetime.now()
    model_cfg["experiment_info"].time_info = time_info.strftime("%Y-%m-%d/%H-%M-%S/")
    model_cfg["hyper_params"] = hyper_params
    model_cfg["archi_cfg"] = architecture
    agent = hydra.utils.instantiate(model_cfg)
    print("Model built!")
    return agent