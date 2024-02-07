import argparse
import importlib
import os
from copy import deepcopy

import wandb
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from cleanfid import fid

import global_v as glv
from network_parser import parse
from utils import get_git_info, add_hook
from envs import Trainer
import snn_layers
from fad.autoencoder_fid import compute_autoencoder_frechet_distance


def train(trainer: Trainer, start_epoch: int, args: argparse.Namespace,
          accelerator: Accelerator):

    for epoch in range(start_epoch, start_epoch + glv.network_config['epochs']):

        train_metrics = trainer.train(epoch)
        test_metrics = trainer.test(epoch)

        if accelerator.is_main_process:
            trainer.save_model(epoch,
                               './debug' if args.debug else wandb.run.dir)
            metrics = {**train_metrics, **test_metrics}
            if not args.debug:
                wandb.log(metrics, step=epoch, commit=True)
            print(f'epoch: {epoch}, ' + ', '.join(
                k + ': ' + f'{v:.4f}'
                for k, v in metrics.items() if isinstance(v, (int, float))))
    if glv.network_config.get('calc_metrics_at_last', True):

        metrics = trainer.calc_final_metrics()
        if accelerator.is_main_process:
            if not args.debug:
                if metrics:
                    wandb.log(metrics)


def calc_metrics(trainer: Trainer, args: argparse.Namespace, checkpoint: dict,
                 accelerator: Accelerator):

    if args.generated_image_dir:
        output_dir = args.generated_image_dir
    else:
        epoch = checkpoint['epoch']
        output_dir = os.path.join('output', args.name, 'img', str(epoch))
        if accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=args.debug)

    metrics_config = glv.network_config['metrics']
    metrics = {}

    if metrics_config['sample']:
        assert checkpoint is not None
        trainer.sample(output_dir, metrics_config.get('num_images', None))

    if not accelerator.is_main_process:
        return

    if metrics_config['fid']:
        name = glv.network_config['dataset'].lower()
        if 'image_size' in glv.network_config:
            name += '-' + str(glv.network_config['image_size'])
        if 'center_crop' in glv.network_config:
            name += '-crop' + str(glv.network_config['center_crop'])
        fid_score = fid.compute_fid(fdir1=output_dir,
                                    batch_size=metrics_config.get(
                                        'batch_size', 128),
                                    device=trainer.device,
                                    dataset_name=name,
                                    dataset_split='custom',
                                    num_workers=0,
                                    use_dataparallel=False)
        metrics['fid'] = fid_score

    if metrics_config['fad']:
        name = glv.network_config['dataset'].lower()
        fad_score = compute_autoencoder_frechet_distance(
            name,
            fdir=output_dir,
            device=trainer.device,
            num_gen=metrics_config.get('num_images', 10000),
            batch_size=metrics_config.get('batch_size', 128))
        metrics['fad'] = fad_score

    if metrics_config['calc_mul_add']:
        count_mul_add, hook_handles = add_hook(trainer.model)
        trainer.model.eval()
        dummy_input = torch.randn(1,
                                  glv.network_config['in_channels'],
                                  glv.network_config['image_size'],
                                  glv.network_config['image_size'],
                                  glv.network_config['n_steps'],
                                  device=trainer.device)
        trainer.model(dummy_input, torch.tensor([0], device=trainer.device))
        count_mul_add.clear()
        sample_num = metrics_config.get('calc_mul_add_sample', 100)
        trainer.diffusion_sample(sample_num)
        mul_sum = count_mul_add.mul_sum / sample_num
        add_sum = count_mul_add.add_sum / sample_num
        mac_sum = count_mul_add.mac_sum / sample_num
        ac_sum = count_mul_add.ac_sum / sample_num
        for handle in hook_handles:
            handle.remove()
        metrics['mul_add_num/mul'] = mul_sum
        metrics['mul_add_num/add'] = add_sum
        metrics['mul_add_num/mac'] = mac_sum
        metrics['mul_add_num/ac'] = ac_sum
        print('mul:', mul_sum)
        print('add:', add_sum)
        print('mac:', mac_sum)
        print('ac:', ac_sum)

    if not args.debug:
        wandb.log(metrics, commit=True)
    print(', '.join(k + ': ' + f'{v:.4f}' for k, v in metrics.items()
                    if isinstance(v, (int, float))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', action='store')
    parser.add_argument('-d', '--gpu-id', type=int, default=0)
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-n', '--name', type=str, default='test')
    parser.add_argument('-m', '--model-checkpoint', type=str, default='')
    parser.add_argument('-i', '--resume-id', type=str)
    parser.add_argument('-o', '--generated-image-dir', type=str, default='')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    accelerator = Accelerator()

    set_seed(args.seed + accelerator.process_index)

    params = parse(args.config)
    network_config = deepcopy(params['network'])

    if accelerator.is_main_process:
        print(network_config)

    glv.init(network_config, accelerator.num_processes)
    snn_layers.init_layer_config(glv.layer_config)

    models = {}
    for model_name, model_config in network_config['models'].items():
        module, cls = model_config['class'].rsplit(".", 1)
        Model = getattr(importlib.import_module(module), cls)
        model = Model(**model_config.get('args', {}))
        models[model_name] = model

    start_epoch = 0
    if args.model_checkpoint:
        checkpoint = torch.load(args.model_checkpoint,
                                map_location=torch.device('cpu'))
        if args.resume_id is not None:
            start_epoch = checkpoint['epoch'] + 1
    else:
        checkpoint = None
        if args.resume_id is not None:
            raise ValueError(
                '--resume-id is specified but --model-checkpoint is not specified'
            )

    if params['experiment_type'] == 'diffusion':
        trainer = Trainer(models, network_config, checkpoint)
    else:
        ValueError('invalid experiment type', params['experiment_type'])

    if accelerator.is_main_process and not args.debug:
        if args.resume_id is not None:
            kwargs = dict(resume='must', id=args.resume_id)
        else:
            kwargs = {}
        wandb.init(dir=network_config.get('wandb_dir', None),
                   project='fsddim',
                   name=args.name,
                   config={
                       'original': params['network'],
                       'adjusted': network_config
                   },
                   **kwargs)
        commit, diff = get_git_info()
        table = wandb.Table(columns=['title', 'contents'],
                            data=[['commit', commit], ['diff', diff]])
        wandb.log({'git_info': table}, step=0)
        for model in models.values():
            wandb.watch(model)
    else:
        os.makedirs('./debug', exist_ok=True)

    train(trainer, start_epoch, args, accelerator)
    checkpoint = torch.load(os.path.join(
        './debug' if args.debug else wandb.run.dir,
        'best_model_checkpoint.pth'),
                            map_location=torch.device('cpu'))
    calc_metrics(trainer, args, checkpoint, accelerator)
