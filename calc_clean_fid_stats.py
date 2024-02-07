import argparse
import itertools
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision
from cleanfid import fid
from tqdm import tqdm

from envs import Trainer
from network_parser import parse


class DummyEnv:

    def __init__(self, *args, **kwargs) -> None:
        Trainer.__init__(self, *args, **kwargs)

    def get_dataset(self, *args, **kwargs):
        return Trainer.get_dataset(self, *args, **kwargs)

    def load_model(self, *args, **kwargs):
        return Trainer.load_model(self, *args, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', action='store')
    parser.add_argument('-d', '--gpu-id', type=int, default=0)
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-o', '--output-dir', type=str, default='output/test')
    parser.add_argument('-n', '--num-images', type=int, default=None)
    parser.add_argument('--use-trainset', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    params = parse(args.config)
    config = params['network']
    print(config)

    device = torch.device(args.gpu_id if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    env = DummyEnv(
        {
            'model': nn.Identity(),
            'encoder': nn.Identity(),
            'decoder': nn.Identity()
        }, config)

    image_dir = os.path.join(args.output_dir, 'img')
    os.makedirs(image_dir, exist_ok=args.debug)

    trainset, testset = env.get_dataset(config['dataset'], config['data_path'])
    dataset = trainset if args.use_trainset else testset
    num_images = args.num_images or len(dataset)

    dataset = itertools.islice(dataset, num_images)
    for i, (x, y) in tqdm(enumerate(dataset),
                          desc='saving images',
                          total=num_images):
        torchvision.utils.save_image(x, os.path.join(image_dir, f'{i}.png'))

    name = config['dataset'].lower()
    if 'image_size' in config:
        name += '-' + str(config['image_size'])
    if 'center_crop' in config:
        name += '-crop' + str(config['center_crop'])
    fid.make_custom_stats(name, image_dir, mode="clean", device=device)
