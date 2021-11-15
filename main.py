import os
# Set hydra debugging level: 1 is max 
os.environ['HYDRA_FULL_ERROR'] = '1'

import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime

from utils.builders import build_model

"""
 Usage: 
    python run_experiments.py for default config
    python run_experiments.py -cp <yaml path> -cn <yaml filename> for custom config
    python run_experiments.py -cp ./configs/singleSin -cn sac
"""
@hydra.main(config_path='configs', config_name="config")
def runExperiment(cfg: DictConfig):
    time_info = datetime.now()
    print(OmegaConf.to_yaml(cfg))
    model = build_model(time_info.strftime("%Y-%m-%d/%H-%M-%S/"), **cfg)
    print("Model Built!")
    #model.train()

if __name__ == "__main__":
    runExperiment()