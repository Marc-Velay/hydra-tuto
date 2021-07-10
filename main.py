import os
# Set hydra debugging level: 1 is max 
os.environ['HYDRA_FULL_ERROR'] = '1'

import hydra
from omegaconf import DictConfig, OmegaConf

from utils.builders import build_model

"""
 Usage: 
    python run_experiments.py for default config
    python run_experiments.py -cp <yaml path> -cn <yaml filename> for custom config
    python run_experiments.py -cp ./configs/singleSin -cn sac
"""
@hydra.main(config_path='./configs')
def runExperiment(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    model = build_model(**cfg)
    model.train()

if __name__ == "__main__":
    runExperiment()