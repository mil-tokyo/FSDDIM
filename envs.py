from collections import defaultdict
import importlib
import itertools
import math
import os
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as trf
import numpy as np
from PIL import Image
from tqdm import tqdm
import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from gaussian_diffusion import GaussianDiffusion
import utils


def convert_values(d: dict):
    for k, v in d.items():
        if isinstance(v, Tensor):
            v = v.item()
            d[k] = v
    return d


class MetricDict(defaultdict):

    def __init__(self, prefix: str = 'train/'):
        super().__init__(float)
        self.prefix = prefix

    def todict(self) -> dict:
        out = dict()
        for k, v in self.items():
            out[self.prefix + k] = v
        return out

    def add(self, x: dict):
        for k in x.keys():
            self[k] += x[k]
        return self

    def div(self, x: float):
        for k in self.keys():
            self[k] /= x
        return self


class DummyScheduler:

    def __init__(self, *args, **kwargs):
        pass

    def step(self, *args, **kwargs):
        pass


class Trainer:

    def __init__(self,
                 models: dict[str, nn.Module],
                 config: dict,
                 checkpoint: Optional[dict] = None) -> None:
        find_unused_parameters = config.get('find_unused_parameters', False)
        kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=find_unused_parameters)
        grad_accum_steps = config.get('gradient_accumulation_steps', 1)
        self.accelerator = Accelerator(
            kwargs_handlers=[kwargs],
            gradient_accumulation_steps=grad_accum_steps)
        self.grad_accum = grad_accum_steps > 1

        device = self.accelerator.device
        self.device = device

        # A model specified 'model' key is used as a main model
        assert 'model' in models

        self.model = models['model']

        for model_name, model in models.items():
            if config.get('sync_batch_norm', True):
                nn.SyncBatchNorm.convert_sync_batchnorm(model)

        self.models = models
        self.load_model(checkpoint)

        self.config = config
        trainset, testset = self.get_dataset(config['dataset'],
                                             config['data_path'])

        self.trainloader = DataLoader(trainset,
                                      batch_size=config['batch_size'],
                                      shuffle=True,
                                      num_workers=config.get('num_workers', 8),
                                      pin_memory=True)

        self.testloader = DataLoader(testset,
                                     batch_size=config['batch_size'],
                                     shuffle=False,
                                     num_workers=config.get('num_workers', 8),
                                     pin_memory=True)

        params = []
        for model_name, model in models.items():
            lr = config['models'][model_name].get('lr', None)
            p = {'params': model.parameters()}
            if lr is not None:
                p['lr'] = lr
            params.append(p)

        # get optimizer class from config and import it automatically
        # default optimizer is Adam
        optimizer_config = config.get('optimizer', dict())
        module, cls = optimizer_config.get('class',
                                           'torch.optim.Adam').rsplit(".", 1)
        Optimizer = getattr(importlib.import_module(module), cls)
        optimizer_args = {'lr': config['lr']}
        optimizer_args.update(optimizer_config.get('args', dict()))
        self.optimizer = Optimizer(params, **optimizer_args)

        encoder = models['encoder']
        decoder = models['decoder']

        self.encoder = encoder
        self.decoder = decoder

        self.diffusion = GaussianDiffusion(self.model,
                                           encoder=encoder,
                                           decoder=decoder,
                                           **self.config['diffusion'])
        self.diffusion_sample = self.diffusion.sample

        self.diffusion, self.optimizer, self.trainloader, self.testloader = self.accelerator.prepare(
            self.diffusion, self.optimizer, self.trainloader, self.testloader)

        self.train_steps_per_epoch = utils.default(
            config.get('train_steps_per_epoch', None), len(self.trainloader))
        self.test_steps_per_epoch = utils.default(
            config.get('test_steps_per_epoch', None), len(self.testloader))

        self.trainloader = utils.cycle(self.trainloader)
        self.testloader = utils.cycle(self.testloader)

        if not 'scheduler' in config or config['scheduler']['class'] is None:
            self.scheduler = DummyScheduler()
        else:
            Scheduler = getattr(lr_scheduler, config['scheduler']['class'])
            self.scheduler = Scheduler(self.optimizer,
                                       **config['scheduler']['args'])
            self.scheduler = self.accelerator.prepare(self.scheduler)

        self.best_fid = float('inf')

    def get_dataset(self, dataset_name: str,
                    data_path: str) -> tuple[Dataset, Dataset]:

        trans = []
        if 'center_crop' in self.config:
            crop = self.config['center_crop']
            trans.append(transforms.CenterCrop(crop))
        if 'image_size' in self.config:
            width = self.config['image_size']
            trans.append(transforms.Resize((width, width)))
        if self.config.get('random_horizontal_flip', False):
            trans.append(transforms.RandomHorizontalFlip())
        trans.append(transforms.ToTensor())

        trans = transforms.Compose(trans)
        Dataset = getattr(torchvision.datasets, dataset_name)
        if dataset_name == 'LSUN':
            category = self.config.get('dataset_category', 'bedroom')
            trainset = Dataset(data_path,
                               classes=[f'{category}_train'],
                               transform=trans)
            testset = Dataset(data_path,
                              classes=[f'{category}_val'],
                              transform=trans)
        elif dataset_name == 'CelebA':
            trainset = Dataset(data_path,
                               split='train',
                               download=True,
                               transform=trans)
            testset = Dataset(data_path,
                              split='valid',
                              download=True,
                              transform=trans)
        else:
            trainset = Dataset(data_path,
                               train=True,
                               download=True,
                               transform=trans)
            testset = Dataset(data_path,
                              train=False,
                              download=True,
                              transform=trans)
        return trainset, testset

    def calc_loss(self, input: Tensor) -> tuple[Tensor, dict]:
        image_level_loss, spike_level_loss, recons_loss = self.diffusion(input)
        diffusion_loss = torch.zeros_like(image_level_loss)

        loss_config = self.config.get('loss', dict())
        if loss_config.get('image_level_loss', True):
            weight = loss_config.get('image_level_loss_weight', 1.0)
            diffusion_loss = diffusion_loss + weight * image_level_loss
        if loss_config.get('spike_level_loss', True):
            weight = loss_config.get('spike_level_loss_weight', 1.0)
            diffusion_loss = diffusion_loss + weight * spike_level_loss

        loss = diffusion_loss
        if loss_config.get('recons_loss', True):
            weight = loss_config.get('recons_loss_weight', 1.0)
            loss = loss + weight * recons_loss

        batch_size = len(input)
        return loss, {
            'loss': loss * batch_size,
            'diffusion_loss': diffusion_loss * batch_size,
            'image_level_loss': image_level_loss * batch_size,
            'spike_level_loss': spike_level_loss * batch_size,
            'reconstruction_loss': recons_loss * batch_size,
        }

    def calc_fid(self):
        self.diffusion.eval()
        batch_size = self.config['batch_size'] * self.config.get(
            'fid_calc_batch_mult', 4)
        batch_size = int(batch_size)
        name = self.config['dataset'].lower()
        if 'image_size' in self.config:
            name += '-' + str(self.config['image_size'])
        if 'center_crop' in self.config:
            name += '-crop' + str(self.config['center_crop'])

        def sample_func(z):
            sample = self.diffusion_sample(batch_size=batch_size)
            sample *= 255
            if sample.shape[1] == 1:
                sample = sample.repeat(1, 3, 1, 1)
            return sample

        batch_size = batch_size * self.accelerator.num_processes
        feats = utils.compute_feats(gen=sample_func,
                                    batch_size=batch_size,
                                    device=self.device,
                                    use_dataparallel=False,
                                    num_gen=self.config.get(
                                        'num_fid_sample', 50000),
                                    z_dim=2)
        feats = torch.tensor(feats, device=self.device)
        feats = self.accelerator.gather(feats).cpu().numpy()
        score = utils.compute_fid(feats,
                                  dataset_name=name,
                                  dataset_split='custom')
        return score

    def calc_final_metrics(self):
        metrics = {}
        self.diffusion.eval()
        if self.accelerator.is_main_process:
            sampled_images = self.diffusion_sample(batch_size=16)
            images = torchvision.utils.make_grid(sampled_images, nrow=4)

            img = wandb.Image(trf.to_pil_image(images))
            metrics['sample'] = img

        score = self.calc_fid()
        metrics['fid'] = score
        return metrics

    def train(self, epoch: int = 0) -> dict:
        train_metrics = dict()
        if epoch % 10 == 0:
            if self.accelerator.is_main_process:
                self.diffusion.eval()
                sampled_images = self.diffusion_sample(batch_size=16)
                images = torchvision.utils.make_grid(sampled_images, nrow=4)

                img = wandb.Image(trf.to_pil_image(images))
                train_metrics['sample'] = img

        metrics = MetricDict('train/')
        num_train_samples = 0

        def train_one_step(x):
            loss, loss_dict = self.calc_loss(x)
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()
            metrics.add(loss_dict)

        self.set_train()
        for x, _ in itertools.islice(self.trainloader,
                                     self.train_steps_per_epoch):
            num_train_samples += x.shape[0]
            if self.grad_accum:
                with self.accelerator.accumulate(self.diffusion):
                    train_one_step(x)
            else:
                train_one_step(x)
        self.scheduler.step()
        metrics = self.accelerator.reduce(metrics.todict(), 'mean')
        metrics = MetricDict.div(metrics, num_train_samples)
        metrics = convert_values(metrics)

        train_metrics.update(metrics)

        if self.config.get('calc_fid', False):
            if (epoch + 1) % self.config.get('calc_fid_step', 10) == 0:
                score = self.calc_fid()
                train_metrics['fid'] = score
                if score < self.best_fid:
                    self.best_fid = score
                    save_dir = './debug' if wandb.run is None else wandb.run.dir
                    self.save_model(epoch, save_dir,
                                    'best_model_checkpoint.pth')
                    print('saved best model checkpoint')

        return train_metrics

    def sample(self, save_dir: str, num_samples: Optional[int] = None) -> None:
        self.diffusion.eval()
        batch_size = self.config['batch_size'] * self.config.get(
            'fid_calc_batch_mult', 4)
        batch_size = int(batch_size)
        batch_size = self.config['metrics'].get(
            'sampling_batch_size_per_process', batch_size)
        num_samples = num_samples or self.config.get('num_fid_sample', 50000)
        print(f'sampling {num_samples} images')
        num_processes = self.accelerator.num_processes
        assert num_samples % num_processes == 0
        num_samples //= num_processes
        for i in tqdm(range(math.ceil(num_samples / batch_size))):
            sampled_images = self.diffusion_sample(batch_size=batch_size)
            for j, img in enumerate(sampled_images):
                idx = (i * batch_size * num_processes + j * num_processes +
                       self.accelerator.process_index)
                if idx >= num_samples * num_processes:
                    break
                torchvision.utils.save_image(
                    img, os.path.join(save_dir, f'{idx}.png'))
        return math.ceil(num_samples / batch_size) * batch_size

    def set_train(self):
        for model in self.models.values():
            model.train()

    def set_eval(self):
        for model in self.models.values():
            model.eval()

    @torch.no_grad()
    def test(self, epoch: int) -> dict:
        test_metrics = MetricDict('test/')
        num_test_samples = 0
        self.set_eval()
        for x, _ in itertools.islice(self.testloader,
                                     self.test_steps_per_epoch):
            num_test_samples += x.shape[0]
            _, loss_dict = self.calc_loss(x)
            test_metrics.add(loss_dict)
        test_metrics = self.accelerator.reduce(test_metrics.todict(), 'mean')
        test_metrics = MetricDict.div(test_metrics, num_test_samples)
        test_metrics = convert_values(test_metrics)

        return test_metrics

    def save_model(self,
                   epoch: int,
                   dir: str,
                   name: str = 'model_checkpoint.pth') -> None:
        checkpoint = {'epoch': epoch}
        for model_name, model in self.models.items():
            if isinstance(model, nn.parallel.DistributedDataParallel):
                checkpoint[
                    f'{model_name}_state_dict'] = model.module.state_dict()
            else:
                checkpoint[f'{model_name}_state_dict'] = model.state_dict()

        torch.save(checkpoint, os.path.join(dir, name))

    def load_model(self, checkpoint=None) -> None:
        for model_name, model in self.models.items():
            if checkpoint is not None:
                key = f'{model_name}_state_dict'
                if key in checkpoint:
                    model.load_state_dict(checkpoint[key])
                    print(f'Loaded {model_name} state dict')
            model.to(self.device)
