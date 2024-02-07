import torch

dtype = None
n_steps = None
network_config = None
layer_config = None
num_devices = None
params = {}
soft_reset = None


def init(n_config, num_devs):
    global dtype, num_devices, n_steps, network_config, layer_config, params, soft_reset
    dtype = torch.float32
    num_devices = num_devs
    network_config = n_config
    assert network_config['batch_size'] % num_devices == 0
    network_config['lr'] *= (
        network_config['batch_size'] / 256 *
        network_config.get('gradient_accumulation_steps', 1))
    network_config['batch_size'] //= num_devices
    layer_config = n_config.get('layer_config', {})
    n_steps = network_config['n_steps']
    soft_reset = network_config.get('soft_reset', False)