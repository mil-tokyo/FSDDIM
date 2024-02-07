import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from denoising_diffusion_pytorch import Unet
import einops

from snn_model import DirectInputEncoder, MPInverseEncoder, \
    AverageDecoder, MembranePotentialDecoder


class ANNUNet(Unet):
    pass


class SNNLikeUNet(Unet):

    def __init__(self,
                 dim,
                 n_steps,
                 init_dim=None,
                 out_dim=None,
                 dim_mults=[],
                 channels=3,
                 self_condition=False,
                 resnet_block_groups=8,
                 learned_variance=False,
                 learned_sinusoidal_cond=False,
                 random_fourier_features=False,
                 learned_sinusoidal_dim=16):

        out_dim = out_dim if out_dim is not None else channels
        adjusted_out_dim = out_dim * n_steps
        adjusted_channels = channels * n_steps
        self.n_steps = n_steps

        super().__init__(dim, init_dim, adjusted_out_dim, dim_mults,
                         adjusted_channels, self_condition, resnet_block_groups,
                         learned_variance, learned_sinusoidal_cond,
                         random_fourier_features, learned_sinusoidal_dim)

        self.out_dim = out_dim
        self.channels = channels

    def forward(self, x, time, x_self_cond=None):
        x = einops.rearrange(x, 'b c h w s -> b (c s) h w')
        x = super().forward(x, time, x_self_cond)
        x = einops.rearrange(x, 'b (c s) h w -> b c h w s', s=self.n_steps)
        return x


class ResnetBlock(nn.Module):

    def __init__(self,
                 in_ch,
                 out_ch,
                 dropout: float,
                 temb_ch: Optional[int] = None,
                 class_ch: Optional[int] = None) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch,
                               out_ch,
                               kernel_size=3,
                               stride=1,
                               padding=1)

        if class_ch is None:
            self.class_proj = None
        else:
            self.class_proj = nn.Linear(class_ch, out_ch)

        if temb_ch is None:
            self.temb_proj = None
        else:
            self.temb_proj = nn.Linear(temb_ch, out_ch)

        self.bn1 = nn.BatchNorm2d(out_ch)
        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv2d(out_ch,
                               out_ch,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

        if in_ch != out_ch:
            self.nin_shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_ch))
        else:
            self.nin_shortcut = nn.BatchNorm2d(out_ch)

        self.relu = nn.ReLU()

    def forward(self, x, *, temb, y):
        batch_size = x.shape[0]

        h = x
        h = self.conv1(h)
        if temb is not None:
            assert temb.shape[0] == batch_size and temb.ndim == 2
            assert self.temb_proj is not None
            h += self.temb_proj(self.relu(temb))[:, :, None, None]

        if y is not None:
            assert self.class_proj is not None
            assert y.ndim == 2 and y.shape[0] == batch_size
            h += self.class_proj(y)[:, :, None, None]

        h = self.relu(self.dropout(self.bn1(h)))
        h = self.bn2(self.conv2(h))

        x = self.nin_shortcut(x)

        assert x.shape == h.shape

        return self.relu(x + h)


class DownsamplingLayer(nn.Module):

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2, padding=0)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        _, _, height, width = x.shape
        x = self.relu(self.bn(self.conv(x)))
        assert x.shape[2:] == (height // 2, width // 2)
        return x


class UpsamplingLayer(nn.Module):

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        bs, channels, height, width = x.shape
        x = self.upsample(x)
        x = self.relu(self.bn(self.conv(x)))
        assert x.shape[2:] == (height * 2, width * 2)
        return x


class UNet(nn.Module):

    def __init__(self,
                 num_classes: int,
                 image_size: int,
                 in_ch: int,
                 ch: int,
                 out_ch: int,
                 ch_mult: tuple[int],
                 num_res_blocks: int,
                 dropout: float,
                 max_time: int = 1000,
                 **kwargs) -> None:
        super().__init__()

        self.out_ch = out_ch
        self.num_res_blocks = num_res_blocks

        channels = [ch] + [ch * mult for mult in ch_mult]
        self.channels = channels
        in_out_channeles = list(zip(channels[:-1], channels[1:]))
        num_resolutions = len(in_out_channeles) - 1
        temb_ch = ch * 4
        class_ch = ch * 4

        assert num_classes >= 1
        self.num_classes = num_classes
        if num_classes > 1:
            self.class_emb = nn.Linear(num_classes, class_ch)
        else:
            self.class_emb = None

        self.init_timestep_embedding(ch, max_time=max_time)
        self.temb = nn.Sequential(nn.Linear(ch, temb_ch), nn.ReLU(),
                                  nn.Linear(temb_ch, temb_ch))

        self.conv_in = nn.Conv2d(in_ch, ch, kernel_size=3, stride=1, padding=1)
        self.bn_in = nn.BatchNorm2d(ch)

        downs = nn.ModuleList()

        for i, (ch_in, ch_out) in enumerate(in_out_channeles):
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(
                    ResnetBlock(ch_in,
                                ch_in,
                                dropout=dropout,
                                temb_ch=temb_ch,
                                class_ch=class_ch))

            if i < len(in_out_channeles) - 1:
                blocks.append(DownsamplingLayer(ch_in, ch_out))
            else:
                blocks.append(
                    nn.Sequential(
                        nn.Conv2d(ch_in,
                                  ch_out,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1),
                        nn.BatchNorm2d(ch_out),
                        nn.ReLU(),
                    ))

            downs.append(blocks)

        ch_mid = channels[-1]

        self.mid_block1 = ResnetBlock(ch_mid, ch_mid, dropout, temb_ch,
                                      class_ch)
        self.mid_block2 = ResnetBlock(ch_mid, ch_mid, dropout, temb_ch,
                                      class_ch)

        ups = nn.ModuleList()

        for i, (ch_in, ch_out) in enumerate(reversed(in_out_channeles)):
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(
                    ResnetBlock(ch_out + ch_in,
                                ch_out,
                                dropout=dropout,
                                temb_ch=temb_ch,
                                class_ch=class_ch))

            if i < len(in_out_channeles) - 1:
                blocks.append(UpsamplingLayer(ch_out, ch_in))
            else:
                blocks.append(
                    nn.Sequential(
                        nn.Conv2d(ch_out,
                                  ch_in,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1),
                        nn.BatchNorm2d(ch_out),
                        nn.ReLU(),
                    ))

            ups.append(blocks)

        self.downs = downs
        self.ups = ups

        self.final_res_block = ResnetBlock(ch * 2,
                                           ch,
                                           dropout=dropout,
                                           temb_ch=temb_ch,
                                           class_ch=class_ch)

        self.conv_out = nn.Conv2d(channels[0],
                                  out_ch,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)

    def forward(self, x: Tensor, t: Tensor, y: Tensor):
        bs, _, height, width = x.shape
        assert height == width
        assert t.shape == (bs, )

        if self.class_emb is not None:
            assert y.shape == (bs, )
            y = F.one_hot(y, self.num_classes).to(torch.get_default_dtype())
            y = y.unsqueeze(-1)
            y = self.class_emb(y)
        else:
            y = None

        temb = self.get_timestep_embedding(t)
        temb = self.temb(temb)

        x = self.conv_in(x)
        hs = [x]

        for blocks in self.downs:
            *res_blocks, downsampling = blocks
            for block in res_blocks:
                x = block(x, temb=temb, y=y)
                hs.append(x)
            x = downsampling(x)

        x = self.mid_block1(x, temb=temb, y=y)
        x = self.mid_block2(x, temb=temb, y=y)

        for blocks in self.ups:
            *res_blocks, upsampling = blocks
            for block in res_blocks:
                x = block(torch.cat([x, hs.pop()], dim=1), temb=temb, y=y)
            x = upsampling(x)

        x = self.final_res_block(torch.cat([x, hs.pop()], dim=1),
                                 temb=temb,
                                 y=y)

        assert not hs

        x = self.conv_out(x)

        return x

    def init_timestep_embedding(self, embedding_dim, max_time=1000):
        position = torch.arange(max_time).unsqueeze(1)
        half_dim = embedding_dim // 2
        div_term = torch.exp(
            torch.arange(half_dim) * (-math.log(10000.0) / (half_dim - 1)))
        pe = torch.zeros(max_time, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1:embedding_dim // 2 * 2:2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def get_timestep_embedding(self, timesteps):
        assert timesteps.ndim == 1
        return self.pe[timesteps]


class UNetV2(UNet):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        in_ch = self.channels[0] * 2 + self.channels[1] * 2 * self.num_res_blocks
        out_ch = self.conv_out.out_channels

        self.conv_out = nn.Conv2d(in_ch,
                                  out_ch,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)

    def forward(self, x: Tensor, t: Tensor, y: Tensor) -> Tensor:
        bs, _, height, width = x.shape
        assert height == width
        assert t.shape == (bs, )

        if self.class_emb is not None:
            assert y.shape == (bs, )
            y = F.one_hot(y, self.num_classes).to(torch.get_default_dtype())
            y = y.unsqueeze(-1)
            y = self.class_emb(y)
        else:
            y = None

        temb = self.get_timestep_embedding(t)
        temb = self.temb(temb)

        x = self.conv_in(x)
        hs = [x]
        h_to_final = [x]

        for i, blocks in enumerate(self.downs):
            *res_blocks, downsampling = blocks
            for block in res_blocks:
                x = block(x, temb=temb, y=y)
                hs.append(x)
                if i == 0:
                    h_to_final.append(x)
            x = downsampling(x)

        x = self.mid_block1(x, temb=temb, y=y)
        x = self.mid_block2(x, temb=temb, y=y)

        for i, blocks in enumerate(self.ups):
            *res_blocks, upsampling = blocks
            for block in res_blocks:
                x = block(torch.cat([x, hs.pop()], dim=1), temb=temb, y=y)
                if i == len(self.ups) - 1:
                    h_to_final.append(x)
            x = upsampling(x)

        x = self.final_res_block(torch.cat([x, hs.pop()], dim=1),
                                 temb=temb,
                                 y=y)

        assert not hs

        x = self.conv_out(torch.cat([x, *h_to_final], dim=1))

        return x


class UNetTest(UNet):

    def forward(self, x: Tensor):
        batch_size = x.shape[0]
        device = x.device
        t = torch.zeros(batch_size, dtype=torch.int64, device=device)
        y = torch.zeros(batch_size, dtype=torch.int64, device=device)
        return super().forward(x, t, y)


class DenoisingUNetTest(nn.Module):

    def __init__(self,
                 *,
                 unet: str,
                 in_ch: int,
                 out_ch: int,
                 n_steps: int,
                 encoder: Optional[str] = None,
                 decoder: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__()
        assert unet in globals()
        self.unet = globals()[unet](in_ch=in_ch, out_ch=out_ch, **kwargs)
        self.n_steps = n_steps
        self.channels = out_ch
        self.out_dim = out_ch
        self.random_or_learned_sinusoidal_cond = False
        self.self_condition = False

        if encoder is None:
            self.encoder = nn.Identity()
        elif encoder == 'direct_input':
            self.encoder = DirectInputEncoder(n_steps)
        elif encoder == 'membrane_inverse':
            self.encoder = MPInverseEncoder(n_steps)

        if decoder is None:
            self.decoder = nn.Identity()
        elif decoder == 'average':
            self.decoder = AverageDecoder()
        elif decoder == 'membrane_potential':
            self.decoder = MembranePotentialDecoder()

    def forward(self, x: Tensor, t: Tensor, y: Optional[Tensor] = None):
        x = self.decoder(x)
        out = self.unet(x, t, y)
        out = self.encoder(out)
        return out