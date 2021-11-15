from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from omegaconf import DictConfig

from environments.builder import build_env

from utils.abstracts.class_selector import ClassSelector
from compute_models.base import BaseModel


class Classifier(ABC):
    """Abstract base class for RL agents.
    Attributes:
        experiment_info (DictConfig): configurations for running main loop (like args)
        env_info (DictConfig): env info for initialization trading environment
        hyper_params (DictConfig): algorithm hyperparameters
        model_cfg (DictConfig): configurations for building neural networks
        env : trading environment
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

        self.env = build_env(self.experiment_info)

    @abstractmethod
    def _initialize(self):
        pass

    @abstractmethod
    def step(
        self, state: np.ndarray, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.float64, bool]:
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def load_learner(self):
        pass

    def test(
        self,
        policy: BaseModel,
        action_selector: ClassSelector,
        episode_i: int,
        update_step: int,
    ) -> float:
        """Test policy without random exploration a number of times."""
        print("====TEST START====")
        policy.eval()
        action_selector.exploration = False
        episode_rewards = []
        for test_i in range(self.experiment_info.test_num):
            state = self.env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = action_selector(policy, state)
                state, action, reward, next_state, done = self.step(state, action)
                episode_reward = episode_reward + reward
                state = next_state
            if self.experiment_info.render_test:
                print("TEST Render")
                self.env.render()

            print(
                f"episode num: {episode_i} | test: {test_i} episode reward: {episode_reward}"
            )
            episode_rewards.append(episode_reward)

        mean_rewards = np.mean(episode_rewards)
        print(
            f"EPISODE NUM: {episode_i} | UPDATE STEP: {update_step} |"
            f"MEAN REWARD: {np.mean(episode_rewards)}"
        )
        action_selector.exploration = True
        print("====TEST END====")

        return mean_rewards

    def validate(self):
        #Test policy without random exploration a number of times.
        print("====Validation START====")
        policy = self.learner.get_policy(self.use_cuda)
        policy.eval()
        self.action_selector.exploration = False
        episode_rewards = []
        for test_i in range(self.experiment_info.test_num):
            state = self.env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = self.action_selector(policy, state)
                state, action, reward, next_state, done = self.step(state, action)
                episode_reward = episode_reward + reward
                state = next_state
                
            self.env.render()

            print(
                f"Test: {test_i} episode reward: {episode_reward}"
            )
            episode_rewards.append(episode_reward)

        mean_rewards = np.mean(episode_rewards)
        print(
            f"MEAN REWARD: {np.mean(episode_rewards)}"
        )
        print("====Validation END====")

        return mean_rewards