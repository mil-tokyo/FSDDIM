import glob
import os
import random
import subprocess
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
from cleanfid.features import build_feature_extractor, get_reference_statistics
from cleanfid.fid import get_model_features, frechet_distance
from PIL import Image
import snn_layers
import snn_model


def get_git_info():
    commit = subprocess.run('git show -q'.split(),
                            capture_output=True,
                            text=True).stdout
    diff = subprocess.run('git diff'.split(), capture_output=True,
                          text=True).stdout

    return commit, diff


def direct_spike_input(x: Tensor, n_steps: int) -> Tensor:
    spike_input = x.unsqueeze(-1).repeat(1, 1, 1, 1, n_steps)
    return spike_input


def membrane_potential_output(x: Tensor) -> Tensor:
    n_steps = x.shape[-1]
    arr = torch.arange(n_steps - 1, -1, -1, device=x.device)
    coef = torch.pow(0.8, arr)[(None, ) * (x.ndim - 1)]
    mem_out = (x * coef).sum(-1)
    return mem_out


def membrane_potential_inverse(x: Tensor,
                               n_steps: int,
                               gamma: float = 0.8) -> Tensor:
    assert x.ndim == 4  # BCHW
    arr = torch.arange(n_steps - 1, -1, -1, device=x.device)
    coef = torch.pow(gamma, arr)[(None, ) * x.ndim]  # 1111S
    coef = coef / (coef**2).sum()
    spike_input = x.unsqueeze(-1) * coef
    return spike_input


def mean_output(x: Tensor) -> Tensor:
    return x.mean(-1)


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def default(val, default):
    if val is None:
        return default
    else:
        return val


def get_pure_torch_module(model):
    if isinstance(model, nn.parallel.DistributedDataParallel):
        return model.module
    else:
        return model


def compute_feats(gen=None,
                  mode="clean",
                  model_name="inception_v3",
                  batch_size=32,
                  device=torch.device("cuda"),
                  num_gen=50_000,
                  z_dim=512,
                  custom_feat_extractor=None,
                  verbose=True,
                  custom_image_tranform=None,
                  custom_fn_resize=None,
                  use_dataparallel=True):
    # build the feature extractor based on the mode and the model to be used
    if custom_feat_extractor is None and model_name == "inception_v3":
        feat_model = build_feature_extractor(mode,
                                             device,
                                             use_dataparallel=use_dataparallel)
    elif custom_feat_extractor is None and model_name == "clip_vit_b_32":
        from cleanfid.clip_features import CLIP_fx, img_preprocess_clip
        clip_fx = CLIP_fx("ViT-B/32", device=device)
        feat_model = clip_fx
        custom_fn_resize = img_preprocess_clip
    else:
        feat_model = custom_feat_extractor

    # Generate features of images generated by the model
    np_feats = get_model_features(gen,
                                  feat_model,
                                  mode=mode,
                                  z_dim=z_dim,
                                  num_gen=num_gen,
                                  batch_size=batch_size,
                                  device=device,
                                  verbose=verbose,
                                  custom_image_tranform=custom_image_tranform,
                                  custom_fn_resize=custom_fn_resize)

    return np_feats


def compute_fid(feats,
                mode="clean",
                model_name="inception_v3",
                dataset_name="FFHQ",
                dataset_res=1024,
                dataset_split="train"):

    # compute fid for a generator, using reference statistics
    print(
        f"compute FID of a model with {dataset_name}-{dataset_res} statistics")

    # Load reference FID statistics (download if needed)
    ref_mu, ref_sigma = get_reference_statistics(dataset_name,
                                                 dataset_res,
                                                 mode=mode,
                                                 model_name=model_name,
                                                 seed=0,
                                                 split=dataset_split)

    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    fid = frechet_distance(mu, sigma, ref_mu, ref_sigma)

    return fid


def make_grid_image(images: list, num_rows: int = 4):
    assert len(images) % num_rows == 0
    num_cols = len(images) // num_rows
    images = [
        np.hstack(images[i * num_cols:(i + 1) * num_cols])
        for i in range(num_rows)
    ]
    images = np.vstack(images)
    return images


def sample_images(path: str, num_images: int = 16, name: str = "sample.png"):
    num_rows = int(num_images**0.5)
    images = glob.glob(os.path.join(path, '**/*.png'), recursive=True)
    images = random.sample(images, num_images)
    images = [np.array(Image.open(img)) for img in images]
    img = make_grid_image(images, num_rows)
    img = Image.fromarray(img)
    img.save(name)


def is_binary(x: Tensor) -> bool:
    return torch.all((x == 0) | (x == 1))


class CountMulAdd:

    def __init__(self) -> None:
        self.mul_sum = 0
        self.add_sum = 0
        self.mac_sum = 0
        self.ac_sum = 0

    def __call__(self, module, module_in, module_out):

        if isinstance(module_in, tuple):
            input_list = module_in
            module_in = module_in[0]
        if isinstance(module_out, tuple):
            module_out = module_out[0]

        modified = False
        concat = False
        if isinstance(module,
                      (snn_model.DownsamplingLayer, snn_model.UpsamplingLayer)):
            module_out.prev_input = module_in
            module_out.parent_node = module
            for child in module.modules():
                assert not isinstance(child, snn_layers.LIFSpike)
                child.avoid = True
        if isinstance(module, snn_model.ConcatLayer):
            module_out.prev_input = input_list
            module_out.parent_node = module
        if hasattr(module_in, 'parent_node'):
            if isinstance(module_in.parent_node, snn_model.DownsamplingLayer):
                modified = True
                module_in = module_in.prev_input
                modified_kernel = (2 * module.kernel_size[0],
                                   2 * module.kernel_size[1], 1)
                modified_stride = (2, 2, 1)
                assert is_binary(module_in)
            elif isinstance(module_in.parent_node, snn_model.ConcatLayer):
                modified = True
                concat = True
                module_in2 = module_in.prev_input[1]
                if hasattr(module_in.prev_input[0], 'parent_node'):
                    assert isinstance(module_in.prev_input[0].parent_node,
                                      snn_model.UpsamplingLayer)
                    modified_kernel = (module.kernel_size[0] + 2,
                                       module.kernel_size[1] + 2, 1)
                    modified_stride = (1, 1, 1)
                    module_in = module_in.prev_input[0].prev_input
                else:
                    modified_kernel = module.kernel_size
                    modified_stride = module.stride
                    module_in = module_in.prev_input[0]
                assert is_binary(module_in)

        if not module.training:
            with torch.no_grad():
                if isinstance(module, nn.AvgPool3d):
                    module = torch.nn.Conv3d(1, 1, module.kernel_size,
                                             module.stride, module.padding)
                if getattr(module, 'avoid', False):
                    add = 0
                    mul = 0
                elif isinstance(module, torch.nn.Conv3d):
                    if not is_binary(module_in):
                        # real-value images are input to the first conv layer.
                        s_in = module_in.shape
                        s_out = module_out.shape
                        mul = s_in[0] * s_in[1] * s_in[2] * s_in[3] * s_in[
                            4] * module.kernel_size[0] * module.kernel_size[
                                1] * module.out_channels / (module.stride[0] *
                                                            module.stride[1])
                        add = mul + s_out[0] * s_out[1] * s_out[2] * s_out[
                            3] * s_out[4]  # calc of bias
                    else:
                        if modified:
                            kernel_size = modified_kernel
                            stride = modified_stride
                        else:
                            kernel_size = module.kernel_size
                            stride = module.stride
                        add = module_in.sum() * kernel_size[0] * kernel_size[
                            1] * module.out_channels / (stride[0] * stride[1])
                        s = module_out.shape  # (N,C,H,W,T)
                        add += s[0] * s[1] * s[2] * s[3] * s[4]  # calc of bias
                        mul = 0
                        if concat:
                            kernel_size = module.kernel_size
                            stride = module.stride
                            if is_binary(module_in2):
                                add += (module_in2.sum() * kernel_size[0] *
                                        kernel_size[1] * module.out_channels /
                                        (stride[0] * stride[1]))
                                add += s[0] * s[1] * s[2] * s[3] * s[4]
                            else:
                                s_in = module_in2.shape
                                s_out = module_out.shape
                                mul += (s_in[0] * s_in[1] * s_in[2] * s_in[3] *
                                        s_in[4] * module.kernel_size[0] *
                                        kernel_size[1] * module.out_channels /
                                        (stride[0] * stride[1]))
                                add += (mul + s_out[0] * s_out[1] * s_out[2] *
                                        s_out[3] * s_out[4])  # calc of bias
                elif isinstance(module, torch.nn.Conv2d):  # ANN
                    assert not is_binary(module_in)
                    s_in = module_in.shape
                    s_out = module_in.shape
                    mul = (s_in[0] * s_in[1] * s_in[2] * s_in[3] *
                           module.kernel_size[0] * module.kernel_size[1] *
                           module.out_channels /
                           (module.stride[0] * module.stride[1]))
                    add = mul + s_out[0] * s_out[1] * s_out[2] * s_out[3]
                elif isinstance(module, torch.nn.Linear):
                    if module_in.ndim == 3:  # SNN
                        if is_binary(module_in):
                            add = module_in.sum() * module.out_features
                            s = module_out.shape  # (N,C,T)
                            add += s[0] * s[1] * s[2]
                            mul = 0
                        else:
                            s_in = module_in.shape  # (N, C, T)
                            s_out = module_in.shape  # (N, C, T)
                            mul = s_in[0] * s_in[1] * s_out[1] * s_out[2]
                            add = mul + s_out[0] * s_out[1] * s_out[2]
                    elif module_in.ndim == 2:  # ANN
                        s_in = module_in.shape  # (N, C)
                        s_out = module_in.shape
                        mul = s_in[0] * s_in[1] * s_out[1]
                        add = mul + s_out[0] * s_out[1]
                    else:
                        raise ValueError()
                elif isinstance(module, nn.BatchNorm2d):
                    if not is_binary(module_in):
                        add = 0
                        mul = 0
                    else:
                        add = module_in.numel()
                        mul = module_in.numel()

                elif isinstance(module, snn_layers.LIFSpike):
                    s_in = module_in.shape
                    if len(s_in) == 5:  # conv layer
                        add = s_in[0] * s_in[1] * s_in[2] * s_in[3] * s_in[4]
                    elif len(s_in) == 3:  # linear layer
                        add = s_in[0] * s_in[1] * s_in[2]
                    else:
                        raise ValueError()
                    mul = (1 - module_out).sum()  # event-based activation
                else:
                    add = 0
                    mul = 0

                if isinstance(module, nn.BatchNorm2d) and is_binary(module_in):
                    ac = add
                    mac = mul
                elif isinstance(module, snn_layers.LIFSpike):
                    ac = 0
                    mac = 0
                else:
                    ac = add - mul
                    mac = mul

                self.mul_sum = self.mul_sum + mul
                self.add_sum = self.add_sum + add
                self.mac_sum = self.mac_sum + mac
                self.ac_sum = self.ac_sum + ac

        return module_out

    def clear(self):
        self.mul_sum = 0
        self.add_sum = 0
        self.mac_sum = 0
        self.ac_sum = 0


def add_hook(net):
    count_mul_add = CountMulAdd()
    hook_handles = []
    for m in net.modules():
        if isinstance(m, (torch.nn.Conv3d, torch.nn.Linear, snn_layers.LIFSpike,
                          torch.nn.Conv2d, snn_model.DownsamplingLayer,
                          snn_model.UpsamplingLayer, snn_model.ConcatLayer,
                          torch.nn.BatchNorm2d)):

            handle = m.register_forward_hook(count_mul_add)
            hook_handles.append(handle)
    return count_mul_add, hook_handles
