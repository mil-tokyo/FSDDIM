import math
from collections import namedtuple
from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from einops import reduce

from tqdm.auto import tqdm

# constants

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


# gaussian diffusion trainer class


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1, ) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5)**2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-(
        (t *
         (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


# normalization functions


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


class GaussianDiffusion(nn.Module):

    def __init__(self,
                 model,
                 encoder,
                 decoder,
                 *,
                 channels,
                 image_size,
                 timesteps=1000,
                 objective='pred_v',
                 beta_schedule='sigmoid',
                 schedule_fn_kwargs=dict()):
        super().__init__()

        self.model = model
        self.encoder = encoder
        self.decoder = decoder

        self.channels = channels

        self.image_size = image_size

        self.objective = objective

        assert objective in {
            'pred_noise', 'pred_x0', 'pred_v', 'pred_xprev'
        }, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod',
                        torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod',
                        torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for ddim posterior q_{\sigma}(x_{t-1} | x_t, x_0)
        # when \sigma = 0

        coef2 = (torch.sqrt(1. - alphas_cumprod_prev) /
                 torch.sqrt(1. - alphas_cumprod))
        coef1 = alphas_cumprod_prev.sqrt() - alphas_cumprod.sqrt() * coef2

        register_buffer('ddim_posterior_mean_coef1', coef1)
        register_buffer('ddim_posterior_mean_coef2', coef2)

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (
            1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped',
                        torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer(
            'posterior_mean_coef1',
            betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) *
                        torch.sqrt(alphas) / (1. - alphas_cumprod))

        register_buffer('x0_mean_coef1', (1. - alphas_cumprod) /
                        (betas * torch.sqrt(alphas_cumprod_prev)))
        register_buffer('x0_mean_coef2',
                        (1. - alphas_cumprod_prev) * torch.sqrt(alphas) /
                        (betas * torch.sqrt(alphas_cumprod_prev)))

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        if objective == 'pred_noise':
            register_buffer('loss_weight', torch.ones_like(snr))
        elif objective == 'pred_x0':
            register_buffer('loss_weight', snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', snr / (snr + 1))
        elif objective == 'pred_xprev':
            register_buffer('loss_weight', 0.5 / posterior_variance)

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one
        self.unnormalize = unnormalize_to_zero_to_one

    @property
    def device(self):
        return self.betas.device

    def predict_start_from_noise(self, x_t, t, noise):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise)

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) *
                x_start)

    def predict_start_from_v(self, x_t, t, v):
        return (extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v)

    def predict_start_from_prev(self, x_t, t, x_prev):
        return (extract(self.x0_mean_coef1, t, x_t.shape) * x_prev -
                extract(self.x0_mean_coef2, t, x_t.shape) * x_t)

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def ddim_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.ddim_posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.ddim_posterior_mean_coef2, t, x_t.shape) * x_t)
        return posterior_mean

    def model_predictions(self, x, t):
        model_output = self.model(x, t)

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)

        elif self.objective == 'pred_x0':
            x_start = model_output
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_xprev':
            x_start = self.predict_start_from_prev(x, t, model_output)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def ddim_p_mean(self, x, t, clip_denoised=True):
        preds = self.model_predictions(x, t)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean = self.ddim_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean

    def get_recons_loss_weight(self, t, shape, regularize=True):
        # x_prev = coef1 * x_0 + coef2 * x_t
        # x_0 = a * model_out + b * x_t
        # -> x_prev = a * coef1 * model_out + (b * coef1 + coef2) * x_t

        coef1 = self.ddim_posterior_mean_coef1
        coef2 = self.ddim_posterior_mean_coef2

        if self.objective == 'pred_v':
            a = -self.sqrt_one_minus_alphas_cumprod
            b = self.sqrt_alphas_cumprod
        elif self.objective == 'pred_x0':
            a = 1.
            b = 0.
        elif self.objective == 'pred_noise':
            a = -self.sqrt_recipm1_alphas_cumprod
            b = self.sqrt_recip_alphas_cumprod
        elif self.objective == 'pred_xprev':
            a = self.x0_mean_coef1
            b = -self.x0_mean_coef2

        recons_weight = torch.abs(a * coef1)
        x_t_weight = torch.abs(b * coef1 + coef2)

        if regularize:
            recons_weight /= recons_weight.max()
            x_t_weight /= x_t_weight.max()

        recons_weight = extract(recons_weight, t, shape)
        x_t_weight = extract(x_t_weight, t, shape)

        return recons_weight, x_t_weight

    @torch.no_grad()
    def snn_sample(self, x, t: int, use_enc_dec=False):
        b, device = x.shape[0], self.device
        batched_times = torch.full((b, ), t, device=device, dtype=torch.long)

        x = self.ddim_p_mean(x=x, t=batched_times, clip_denoised=use_enc_dec)
        return x

    @torch.no_grad()
    def snn_sample_loop(self, shape, return_all_timesteps=False):
        device = self.device

        img = torch.randn(shape, device=device)
        if return_all_timesteps:
            imgs = [img]
        spiking_img = self.encoder(img)

        for t in tqdm(reversed(range(0, self.num_timesteps)),
                      desc='sampling loop time step',
                      total=self.num_timesteps):
            spiking_img = self.snn_sample(spiking_img, t, False)

            img = self.decoder(spiking_img)
            img.clamp_(-1., 1.)
            if return_all_timesteps:
                imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)

        ret = self.unnormalize(ret)
        return ret

    @torch.no_grad()
    def sample(self, batch_size=16, return_all_timesteps=False):
        image_size, channels = self.image_size, self.channels
        return self.snn_sample_loop(
            (batch_size, channels, image_size, image_size),
            return_all_timesteps=return_all_timesteps)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) *
                noise)

    def p_losses(self, x_start, t, noise=None):

        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_enc = self.encoder(x)

        # predict and take gradient step

        spike_out = self.model(x_enc, t)
        model_out = self.decoder(spike_out)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        elif self.objective == 'pred_xprev':
            target, _, _ = self.q_posterior(x_start=x_start, x_t=x, t=t)
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)

        loss_spike = F.mse_loss(spike_out,
                                self.encoder(target),
                                reduction='none')
        loss_spike = reduce(loss_spike, 'b ... -> b (...)', 'mean')

        loss_spike = loss_spike * extract(self.loss_weight, t, loss_spike.shape)

        reconstructed_output = self.encoder(self.decoder(spike_out))
        loss_recons = F.mse_loss(reconstructed_output,
                                 spike_out,
                                 reduction='none')
        loss_recons = reduce(loss_recons, 'b ... -> b (...)', 'mean')
        recons_weight, _ = self.get_recons_loss_weight(t, loss_recons.shape)
        loss_recons = loss_recons * recons_weight

        return loss.mean(), loss_spike.mean(), loss_recons.mean()

    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b, ), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, *args, **kwargs)
