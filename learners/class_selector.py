from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from utils.abstracts.class_selector import ClassSelector
from utils.utils import np2tensor


# Can be derived for regressor outputs and other discrete applications
class BayesClassSelector(ClassSelector):
    """Class selector for models outputing probabilities
    Attributes:
        output_dim (int): size of output dimension
    """

    def __init__(self, output_dim: int, use_cuda: bool):
        ClassSelector.__init__(self, use_cuda)
        self.output_dim = output_dim

    def __call__(
        self, model: nn.Module, input: np.ndarray
    ) -> Tuple[torch.Tensor, ...]:
        """Generate action via policy"""
        if input.ndim == 1:
            input = input.reshape(1, -1)
        classes = model.sample(np2tensor(input, self.use_cuda))
        output = self.softmax_normalization(classes)
        output_np = output.cpu().detach().view(-1).numpy()
        return output_np

    def softmax_normalization(self, output: np.ndarray) -> np.ndarray: 
        numerator = np.exp(output)
        denominator = np.sum(np.exp(output))
        softmax_output = numerator / denominator
        return softmax_output