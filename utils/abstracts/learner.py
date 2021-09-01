from abc import ABC, abstractmethod
from datetime import datetime
import os
from typing import Tuple

from omegaconf import DictConfig
import torch

from models.base import BaseModel


class LearnerBase(ABC):
    """Abstract base class for Learner"""

    @abstractmethod
    def update_model(
        self, experience: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        pass

    @abstractmethod
    def get_classifier(self, to_cuda: bool) -> BaseModel:
        pass

    @abstractmethod
    def save_params(self):
        pass

    @abstractmethod
    def load_params(self):
        pass


class Learner(LearnerBase):
    """Abstract class for all learners
    Attributes:
        experiment_info (DictConfig): experiment info
        hyper_params (DictConfig): algorithm hyperparameters
        model_cfg (DictConfig): model configurations
        use_cuda (bool): true if using gpu
    """

    def __init__(
        self,
        experiment_info: DictConfig,
        hyper_params: DictConfig,
        model_cfg: DictConfig,
    ):
        self.experiment_info = experiment_info
        self.hyper_params = hyper_params
        self.model_cfg = model_cfg
        self.use_cuda = self.experiment_info.device == "cuda"

        self.ckpt_path = (
            f"../../{self.experiment_info.time_info}"
            f"{self.experiment_info.env.name}/{self.experiment_info.experiment_name}/checkpoints/"
        )
        os.makedirs(self.ckpt_path, exist_ok=True)

    @abstractmethod
    def _initialize(self):
        pass

    @abstractmethod
    def update_model(
        self, experience: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        pass

    @abstractmethod
    def get_classifier(self, to_cuda: bool) -> BaseModel:
        pass

    @abstractmethod
    def save_params(self):
        pass

    @abstractmethod
    def load_params(self):
        pass


class LearnerWrapper(LearnerBase):
    """Abstract base class for Learner Wrappers
    Attributes:
        learner (Learner): learner to be wrapped
    """

    def __init__(self, learner: Learner):
        self.learner = learner

    def update_model(
        self, experience: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        """Call wrapped learner update_model()"""
        return self.learner.update_model(experience)

    def get_classifier(self, to_cuda: bool) -> BaseModel:
        """Call wrapped learner get_classifier()"""
        return self.learner.get_classifier(to_cuda)