import math
from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from snn_layers import (tdLinear, tdConv, tdBatchNorm, LIFSpike)
import utils
import global_v as glv


class ResnetBlock(nn.Module):

    def __init__(self,
                 in_ch,
                 out_ch,
                 dropout: float,
                 num_conv: int = 2,
                 temb_ch: Optional[int] = None) -> None:
        super().__init__()

        self.conv1 = tdConv(in_ch, out_ch, kernel_size=3, stride=1, padding=1)

        if temb_ch is None:
            self.temb_lif = None
            self.temb_proj = None
        else:
            self.temb_lif = LIFSpike()
            self.temb_proj = tdLinear(temb_ch, out_ch)

        self.bn1 = tdBatchNorm(out_ch)

        self.dropout = nn.Dropout(dropout)

        if num_conv == 2:
            self.lif1 = LIFSpike()
            self.conv2 = tdConv(out_ch,
                                out_ch,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bn=tdBatchNorm(out_ch, alpha=1 / math.sqrt(2)))

        if in_ch != out_ch:
            self.nin_shortcut = tdConv(
                in_ch,
                out_ch,
                kernel_size=1,
                stride=1,
                padding=0,
                bn=tdBatchNorm(out_ch, alpha=1 / math.sqrt(2)),
            )
        else:
            self.nin_shortcut = tdBatchNorm(out_ch, alpha=1 / math.sqrt(2))

        self.lif2 = LIFSpike()

        self.num_conv = num_conv

    def forward(self, x, *, temb):
        batch_size = x.shape[0]

        h = x
        h = self.conv1(h)
        if temb is not None:
            assert temb.shape[0] == batch_size and temb.ndim == 3
            assert self.temb_proj is not None
            h += self.temb_proj(self.temb_lif(temb))[:, :, None, None, :]

        h = self.dropout(self.bn1(h))
        if self.num_conv == 2:
            h = self.lif1(h)
            h = self.conv2(h)

        x = self.nin_shortcut(x)

        assert x.shape == h.shape

        return self.lif2(x + h)


class DownsamplingLayer(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, spike: bool = True) -> None:
        super().__init__()
        mode = glv.layer_config.get('downsampling', '2x2conv')
        if mode == '2x2conv':
            self.conv = tdConv(in_ch,
                               out_ch,
                               kernel_size=2,
                               stride=2,
                               padding=0,
                               bn=tdBatchNorm(out_ch) if spike else None,
                               spike=LIFSpike() if spike else None)
        elif mode == '3x3conv':
            self.conv = tdConv(in_ch,
                               out_ch,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bn=tdBatchNorm(out_ch) if spike else None,
                               spike=LIFSpike() if spike else None)
        elif mode == 'avg_pool':
            self.conv = nn.Sequential(
                nn.AvgPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1),
                             padding=0),
                tdConv(in_ch,
                       out_ch,
                       kernel_size=1,
                       stride=1,
                       padding=0,
                       bn=tdBatchNorm(out_ch) if spike else None,
                       spike=LIFSpike() if spike else None))
        else:
            raise ValueError(f'Unsupported downsampling type: {mode}.')

    def forward(self, x: Tensor) -> Tensor:
        _, _, height, width, n_steps = x.shape
        x = self.conv(x)
        assert x.shape[2:] == (height // 2, width // 2, n_steps)
        return x


class UpsamplingLayer(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, spike: bool = True) -> None:
        super().__init__()
        self.conv = tdConv(in_ch,
                           out_ch,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           bn=tdBatchNorm(out_ch) if spike else None,
                           spike=LIFSpike() if spike else None)
        self.upsample = nn.Upsample(scale_factor=(2, 2, 1), mode='nearest')

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        _, _, height, width, n_steps = x.shape
        x = self.upsample(x)
        x = self.conv(x)
        assert x.shape[2:] == (height * 2, width * 2, n_steps)
        return x


class ConcatLayer(nn.Module):

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.cat([x, y], dim=1)


class SpikingUNet(nn.Module):

    def __init__(self,
                 in_ch: int,
                 ch: int,
                 out_ch: int,
                 ch_mult: tuple[int],
                 num_res_blocks: int,
                 dropout: float,
                 max_time: int = 1000,
                 activate_first_conv: bool = True,
                 num_conv_in_res_block: int = 2,
                 temb_at_res_block: bool = True,
                 bn_in_temb: bool = False,
                 num_conv_in_temb: int = 2,
                 temb_ch_mult: int = 4,
                 spike_up_down: bool = True,
                 **kwargs) -> None:
        super().__init__()

        self.out_ch = out_ch
        self.num_res_blocks = num_res_blocks
        self.temb_at_res_block = temb_at_res_block

        channels = [ch] + [ch * mult for mult in ch_mult]
        self.channels = channels
        in_out_channeles = list(zip(channels[:-1], channels[1:]))
        first_temb_ch = ch * temb_ch_mult
        if temb_at_res_block:
            temb_ch = first_temb_ch
            self.temb_proj = None
        else:
            temb_ch = None
            self.temb_proj = tdLinear(first_temb_ch, ch)

        self.init_timestep_embedding(ch, max_time=max_time)
        self.temb = nn.Sequential(
            tdLinear(ch,
                     first_temb_ch,
                     bn=tdBatchNorm(first_temb_ch) if bn_in_temb else None,
                     spike=LIFSpike()),
            *(tdLinear(first_temb_ch,
                       first_temb_ch,
                       bn=tdBatchNorm(first_temb_ch) if bn_in_temb else None,
                       spike=LIFSpike()) for _ in range(num_conv_in_temb - 2)),
            tdLinear(first_temb_ch, first_temb_ch),
        )

        self.conv_in = tdConv(
            in_ch,
            ch,
            kernel_size=3,
            stride=1,
            padding=1,
            bn=tdBatchNorm(ch, alpha=1.) if activate_first_conv else None,
            spike=LIFSpike() if activate_first_conv else None)

        downs = nn.ModuleList()

        for i, (ch_in, ch_out) in enumerate(in_out_channeles):
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(
                    ResnetBlock(ch_in,
                                ch_in,
                                dropout=dropout,
                                num_conv=num_conv_in_res_block,
                                temb_ch=temb_ch))

            if i < len(in_out_channeles) - 1:
                blocks.append(DownsamplingLayer(ch_in, ch_out, spike_up_down))
            else:
                blocks.append(
                    tdConv(ch_in,
                           ch_out,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           bn=tdBatchNorm(ch_out, alpha=1.),
                           spike=LIFSpike()))
            downs.append(blocks)

        ch_mid = channels[-1]

        self.mid_block1 = ResnetBlock(ch_mid, ch_mid, dropout,
                                      num_conv_in_res_block, temb_ch)
        self.mid_block2 = ResnetBlock(ch_mid, ch_mid, dropout,
                                      num_conv_in_res_block, temb_ch)

        ups = nn.ModuleList()

        for i, (ch_in, ch_out) in enumerate(reversed(in_out_channeles)):
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(
                    ResnetBlock(ch_out + ch_in,
                                ch_out,
                                dropout=dropout,
                                num_conv=num_conv_in_res_block,
                                temb_ch=temb_ch))

            if i < len(in_out_channeles) - 1:
                blocks.append(UpsamplingLayer(ch_out, ch_in, spike_up_down))
            else:
                blocks.append(
                    tdConv(ch_out,
                           ch_in,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           bn=tdBatchNorm(ch_in, alpha=1.),
                           spike=LIFSpike()))
            ups.append(blocks)

        self.downs = downs
        self.ups = ups

        self.final_res_block = ResnetBlock(ch * 2,
                                           ch,
                                           dropout=dropout,
                                           num_conv=num_conv_in_res_block,
                                           temb_ch=temb_ch)

        self.conv_out = tdConv(channels[0],
                               out_ch,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bn=None,
                               spike=None)

        self.cat = ConcatLayer()

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        bs, _, height, width, n_steps = x.shape
        assert height == width
        assert t.shape == (bs, )

        temb = self.get_timestep_embedding(t)
        temb = temb.unsqueeze(-1).repeat(1, 1, n_steps)
        temb = self.temb(temb)

        x = self.conv_in(x)
        if not self.temb_at_res_block:
            x = x + self.temb_proj(temb)[:, :, None, None, :]
            temb = None
        hs = [x]

        for blocks in self.downs:
            *res_blocks, downsampling = blocks
            for block in res_blocks:
                x = block(x, temb=temb)
                hs.append(x)
            x = downsampling(x)

        x = self.mid_block1(x, temb=temb)
        x = self.mid_block2(x, temb=temb)

        for blocks in self.ups:
            *res_blocks, upsampling = blocks
            for block in res_blocks:
                x = block(self.cat(x, hs.pop()), temb=temb)
            x = upsampling(x)

        x = self.final_res_block(self.cat(x, hs.pop()), temb=temb)

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


class SpikingUNetV2(SpikingUNet):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        in_ch = self.channels[0] * 2 + self.channels[1] * 2 * self.num_res_blocks
        out_ch = self.conv_out.out_channels
        self.conv_out = tdConv(in_ch,
                               out_ch,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bn=None,
                               spike=None)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        bs, _, height, width, n_steps = x.shape
        assert height == width
        assert t.shape == (bs, )

        temb = self.get_timestep_embedding(t)
        temb = temb.unsqueeze(-1).repeat(1, 1, n_steps)
        temb = self.temb(temb)

        x = self.conv_in(x)
        if not self.temb_at_res_block:
            x = x + self.temb_proj(temb)[:, :, None, None, :]
            temb = None
        hs = [x]
        h_to_final = [x]

        for i, blocks in enumerate(self.downs):
            *res_blocks, downsampling = blocks
            for block in res_blocks:
                x = block(x, temb=temb)
                hs.append(x)
                if i == 0:
                    h_to_final.append(x)
            x = downsampling(x)

        x = self.mid_block1(x, temb=temb)
        x = self.mid_block2(x, temb=temb)

        for i, blocks in enumerate(self.ups):
            *res_blocks, upsampling = blocks
            for block in res_blocks:
                x = block(self.cat(x, hs.pop()), temb=temb)
                if i == len(self.ups) - 1:
                    h_to_final.append(x)
            x = upsampling(x)

        x = self.final_res_block(self.cat(x, hs.pop()), temb=temb)

        assert not hs

        x = self.conv_out(torch.cat([x, *h_to_final], dim=1))

        return x


class DirectInputEncoder(nn.Module):

    def __init__(self, n_steps: int) -> None:
        super().__init__()
        self.n_steps = n_steps

    def forward(self, x: Tensor) -> Tensor:
        return utils.direct_spike_input(x, self.n_steps)


class AverageDecoder(nn.Module):

    def forward(self, x: Tensor) -> Tensor:
        return x.mean(-1)
