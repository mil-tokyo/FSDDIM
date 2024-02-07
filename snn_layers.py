import torch
import torch.nn as nn
import torch.nn.functional as F
import global_v as glv
import utils

dt = 5
a = 0.25
aa = 0.5
Vth = 0.2
tau = 0.25


def init_layer_config(layer_config):
    global dt, a, aa, Vth, tau
    dt = layer_config.get('dt', dt)
    a = layer_config.get('a', a)
    aa = layer_config.get('aa', aa)
    Vth = layer_config.get('Vth', Vth)
    tau = layer_config.get('tau', tau)


class SpikeAct(torch.autograd.Function):
    """ 
        Implementation of the spiking activation function with an approximation of gradient.
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # if input = u > Vth then output = 1
        output = torch.gt(input, Vth)
        return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # hu is an approximate func of df/du
        sg = glv.layer_config.get('surrogate_function', None)
        if sg == 'arctan':
            width = glv.layer_config.get('surrogate_width', 1.0)
            hu = 1.0 / (1.0 + width * (input - Vth)**2)
        elif sg == 'triangle':
            width = glv.layer_config.get('surrogate_width', aa)
            hu = torch.clamp(1 - abs(input - Vth) / width, min=0.0)
        elif sg == 'rectangle' or sg is None:
            width = glv.layer_config.get('surrogate_width', aa)
            hu = abs(input - Vth) < aa
        else:
            raise ValueError(f'Unsupported surrogate function: {sg}')

        hu = hu.float() / (2 * aa)
        return grad_input * hu


class LIFSpike(nn.Module):
    """
        Generates spikes based on LIF module. It can be considered as an activation function and is used similar to ReLU. The input tensor needs to have an additional time dimension, which in this case is on the last dimension of the data.
    """

    def __init__(self, soft_reset=None):
        super(LIFSpike, self).__init__()
        self.soft_reset = utils.default(soft_reset, glv.soft_reset)
        self.detach_reset = glv.layer_config.get('detach_reset', False)
        if self.soft_reset is None:
            import warnings
            warnings.warn(
                'Soft reset or hard reset is not specified. Hard reset will be used.'
            )
            self.soft_reset = False

    def forward(self, x):
        nsteps = x.shape[-1]
        u = torch.zeros(x.shape[:-1], device=x.device)
        out = torch.zeros(x.shape, device=x.device)
        for step in range(nsteps):
            u, out[..., step] = self.state_update(u, out[...,
                                                         max(step - 1, 0)],
                                                  x[..., step])
        return out

    def state_update(self, u_t_n1, o_t_n1, W_mul_o_t1_n, decay=None):
        decay = utils.default(decay, tau)
        if self.detach_reset:
            o_t_n1 = o_t_n1.detach()
        if self.soft_reset:
            u_t1_n1 = decay * (u_t_n1 - o_t_n1 * Vth) + W_mul_o_t1_n
        else:
            u_t1_n1 = decay * u_t_n1 * (1 - o_t_n1) + W_mul_o_t1_n
        o_t1_n1 = SpikeAct.apply(u_t1_n1)
        return u_t1_n1, o_t1_n1


class tdLinear(nn.Linear):

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 bn=None,
                 spike=None):
        assert type(
            in_features
        ) == int, 'inFeatures should not be more than 1 dimesnion. It was: {}'.format(
            in_features.shape)
        assert type(
            out_features
        ) == int, 'outFeatures should not be more than 1 dimesnion. It was: {}'.format(
            out_features.shape)

        super(tdLinear, self).__init__(in_features, out_features, bias=bias)

        self.bn = bn
        self.spike = spike

    def forward(self, x):
        """
        x : (N,*,C,T)
        """
        x = x.transpose(-1, -2)  # (N, *, T, C)
        y = F.linear(x, self.weight, self.bias)
        y = y.transpose(-1, -2)  # (N, *, C, T)

        if self.bn is not None:
            y = y[:, :, None, None, :]
            y = self.bn(y)
            y = y[:, :, 0, 0, :]
        if self.spike is not None:
            y = self.spike(y)
        return y


class tdConv(nn.Conv3d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 bn=None,
                 spike=None,
                 is_first_conv=False):

        # kernel
        if type(kernel_size) == int:
            kernel = (kernel_size, kernel_size, 1)
        elif len(kernel_size) == 2:
            kernel = (kernel_size[0], kernel_size[1], 1)
        else:
            raise Exception(
                'kernelSize can only be of 1 or 2 dimension. It was: {}'.format(
                    kernel_size.shape))

        # stride
        if type(stride) == int:
            stride = (stride, stride, 1)
        elif len(stride) == 2:
            stride = (stride[0], stride[1], 1)
        else:
            raise Exception(
                'stride can be either int or tuple of size 2. It was: {}'.
                format(stride.shape))

        # padding
        if type(padding) == int:
            padding = (padding, padding, 0)
        elif len(padding) == 2:
            padding = (padding[0], padding[1], 0)
        else:
            raise Exception(
                'padding can be either int or tuple of size 2. It was: {}'.
                format(padding.shape))

        # dilation
        if type(dilation) == int:
            dilation = (dilation, dilation, 1)
        elif len(dilation) == 2:
            dilation = (dilation[0], dilation[1], 1)
        else:
            raise Exception(
                'dilation can be either int or tuple of size 2. It was: {}'.
                format(dilation.shape))

        super().__init__(in_channels,
                         out_channels,
                         kernel,
                         stride,
                         padding,
                         dilation,
                         groups,
                         bias=bias)
        self.bn = bn
        self.spike = spike
        self.is_first_conv = is_first_conv

    def forward(self, x):
        x = F.conv3d(x, self.weight, self.bias, self.stride, self.padding,
                     self.dilation, self.groups)
        if self.bn is not None:
            x = self.bn(x)
        if self.spike is not None:
            x = self.spike(x)
        return x


class tdBatchNorm(nn.BatchNorm2d):
    """
        Implementation of tdBN. Link to related paper: https://arxiv.org/pdf/2011.05280. In short it is averaged over the time domain as well when doing BN.
    Args:
        num_features (int): same with nn.BatchNorm2d
        eps (float): same with nn.BatchNorm2d
        momentum (float): same with nn.BatchNorm2d
        alpha (float): an addtional parameter which may change in resblock.
        affine (bool): same with nn.BatchNorm2d
        track_running_stats (bool): same with nn.BatchNorm2d
    """

    def __init__(self,
                 num_features,
                 eps=1e-05,
                 momentum=0.1,
                 alpha=1,
                 affine=True,
                 track_running_stats=True):
        super(tdBatchNorm, self).__init__(num_features, eps, momentum, affine,
                                          track_running_stats)
        self.alpha = alpha

    def forward(self, input):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(
                        self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3, 4])
            # use biased var in train
            var = input.var([0, 2, 3, 4], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = self.alpha * Vth * (input - mean[None, :, None, None, None]) / (
            torch.sqrt(var[None, :, None, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None, None] + self.bias[
                None, :, None, None, None]

        return input
