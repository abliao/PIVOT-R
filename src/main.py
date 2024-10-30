import yaml
from omegaconf import DictConfig
import torch
import os
from trainer import Trainer
def main(args, world_size):
    with open('config/trainer.yaml') as f:
        cfg = yaml.safe_load(f)
    trainer = Trainer(cfg,args,world_size)
    trainer.run()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
    parser.add_argument('--output_path', default='runs', type=str)
    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    args.world_size = world_size
    main(args, args.world_size)
