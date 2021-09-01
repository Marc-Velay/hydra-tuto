import hydra
import numpy as np
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseModel


class CategoricalClassifier(BaseModel):
    """Classifier model initializable with hydra config
    Attributes:
        fc_input (LinearLayer): fully connected input layer
        fc_hidden (nn.Sequential): hidden layers
        fc_output (LinearLayer): fully connected output layer
    """

    def __init__(self, model_cfg: DictConfig):
        BaseModel.__init__(self, model_cfg)

        # set input size of fc input layer
        self.model_cfg.fc.input.input_size = self.get_feature_size()

        # set output size of fc output layer
        if self.model_cfg.fc.output.output_size == "undefined":
            self.model_cfg.fc.output.output_size = self.model_cfg.class_dim

        # initialize input layer
        self.fc_input = hydra.utils.instantiate(self.model_cfg.fc.input)

        # initialize hidden layers
        hidden_layers = []
        for layer in self.model_cfg.fc.hidden:
            layer_info = self.model_cfg.fc.hidden[layer]
            hidden_layers.append(hydra.utils.instantiate(layer_info))
        self.fc_hidden = nn.Sequential(*hidden_layers)

        # initialize output layer
        self.fc_output = hydra.utils.instantiate(self.model_cfg.fc.output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_x = x.size(0)
        x = self.features.forward(x)
        x = x.view(x.size(0), -1)
        x = self.fc_input.forward(x)
        x = self.fc_hidden.forward(x)
        x = self.fc_output.forward(x)
        dist = x.view(num_x, -1, self.class_dim)
        dist = F.softmax(dist, dim=2)

        return dist
