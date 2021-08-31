from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import torch.nn as nn


class ClassSelector(ABC):
    """Abstract base class for callable class selection methods
    Attributes:
        use_cuda (bool): true if using gpu
        bayesian (bool): turn on/off probability scheme
    """

    def __init__(self, use_cuda: bool):
        self.use_cuda = use_cuda
        self.baysian = False

    @abstractmethod
    def __call__(self, model: nn.Module, input: np.ndarray) -> Tuple[np.ndarray, ...]:
        pass