import os
import hydra
from src.train import train

@hydra.main(version_base=None, config_path="config/", config_name="config.yaml")
def main(config):
    if not os.path.isdir(config.log_dir):
        os.makedirs(config.log_dir)
    train(config)

if __name__ == "__main__":
    main()