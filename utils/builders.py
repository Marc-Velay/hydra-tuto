import hydra
from omegaconf import DictConfig

from datetime import datetime

from utils.abstracts.class_selector import ClassSelector
from utils.abstracts.learner import LearnerBase
from utils.abstracts.loss import Loss
from compute_models.base import BaseModel


def build_model(
     time_info, experiment_info: DictConfig, hyper_params: DictConfig, model: DictConfig
):
    """Build model from DictConfigs via hydra.utils.instantiate()"""
    model_cfg = DictConfig(dict())
    model_cfg["_target_"] = experiment_info.model
    model_cfg["experiment_info"] = experiment_info
    model_cfg["experiment_info"].time_info = time_info
    model_cfg["hyper_params"] = hyper_params
    model_cfg["model_cfg"] = model
    model = hydra.utils.instantiate(model_cfg)
    return model

def build_compute_model(model_cfg: DictConfig, use_cuda: bool) -> BaseModel:
    """Build model from DictConfigs via hydra.utils.instantiate()"""
    model_cfg.model_cfg.use_cuda = use_cuda
    model = hydra.utils.instantiate(model_cfg)
    if use_cuda:
        return model.cuda()
    else:
        return model.cpu()


def build_learner(
    experiment_info: DictConfig, hyper_params: DictConfig, model: DictConfig
) -> LearnerBase:
    """Build learner from DictConfigs via hydra.utils.instantiate()"""
    learner_cfg = DictConfig(dict())
    learner_cfg["_target_"] = experiment_info.learner
    learner_cfg["experiment_info"] = experiment_info
    learner_cfg["hyper_params"] = hyper_params
    learner_cfg["model_cfg"] = model
    learner = hydra.utils.instantiate(learner_cfg)
    if learner.model_cfg.load_model:
        learner.load_params(ckpt=learner.model_cfg.load_path, update_step=learner.model_cfg.update_step, inference=learner.model_cfg.inference)
    return learner

def build_action_selector(
    experiment_info: DictConfig, use_cuda: bool
) -> ClassSelector:
    """Build action selector from DictConfig via hydra.utils.instantiate()"""
    class_selector_cfg = DictConfig(dict())
    #action_selector_cfg["class"] = experiment_info.action_selector
    class_selector_cfg["_target_"] = experiment_info.class_selector
    class_selector_cfg["use_cuda"] = use_cuda
    class_selector_cfg.class_dim = experiment_info.env.class_dim
    class_selector = hydra.utils.instantiate(class_selector_cfg)
    return class_selector