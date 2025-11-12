
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,3,6,7'
import torch
import argparse
from omegaconf import OmegaConf

from tools.runner import train

from util import (
    copy_source, get_output_dir
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/generation_cfg.yaml',
                       help='Path to config file')
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    os.environ['MASTER_ADDR'] = cfg.ddp.addr
    os.environ['MASTER_PORT'] = cfg.ddp.port

    exp_id = os.path.splitext(os.path.basename(__file__))[0]
    dir_id = os.path.dirname(__file__)

    output_dir = get_output_dir(dir_id, exp_id)

    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        OmegaConf.save(cfg, f)

    copy_source(__file__, output_dir)

    cfg.ddp.ngpus_per_node = torch.cuda.device_count()
    cfg.ddp.world_size = cfg.ddp.ngpus_per_node

    print('world size:{}'.format(cfg.ddp.world_size))
    torch.multiprocessing.spawn(train, nprocs=cfg.ddp.ngpus_per_node, args=(cfg, output_dir))

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()