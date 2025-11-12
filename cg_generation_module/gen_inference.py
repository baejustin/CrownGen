import argparse
from omegaconf import OmegaConf
from tools.runner import inference

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/generation_cfg.yaml')
    cli_args = parser.parse_args()

    inference(OmegaConf.load(cli_args.config))

