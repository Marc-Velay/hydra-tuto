from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

class Dataset(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def step(
        self, state: np.ndarray, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.float64, bool]:
        pass

    @abstractmethod
    def render(self):
        pass

