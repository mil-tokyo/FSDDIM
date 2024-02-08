# Fully Spiking Denoising Diffusion Implicit Models

Official implementation of FSDDIM

arxiv: https://arxiv.org/abs/2312.01742

# Initialization

1. Install requirements

   ```bash
   pip install -r requirements.txt
   ```

2. Login to [Weights & Biases](https://wandb.ai/site/)

   Please follow the official instruction of Weights & Biases for more details.

   ```bash
   wandb login
   ```

3. Initialize fid stats

   ```bash
   python calc_clean_fid_stats.py -c <config_file> -d 0 -o <output_directory>
   ```

   - args
     - `-c`, `--config`
       - Path to the config file.
       - Sample config files can be found in `configs/`.
     - `-d`, `--gpu-id`
       - GPU id.
     - `-o`, `--output-dir`
       - Directory in which dataset images will be saved.

# Training

We use [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/index). Please follow the official instruction of Accelerate for more details.

```bash
accelerate launch --multi_gpu --num_processes=4 --gpu_ids=0,1,2,3 --mixed_precision fp16 main.py -c <config_file> -n <experiment_name>
```

- args
  - `-c`, `--config`
    - Path to the config file.
    - Sample config files can be found in `configs/`.
  - `-n`, `--name`
    - Experiment name.
    - Please specify a unique name because generated images will be saved in `output/<experiment_name>`.

# Evaluation metrics

The scores of evaluation metrics are approximately as follows.

| Dataset       | Time steps | Fréchet Inception Distance (FID) | Fréchet Autoencoder Distance (FAD) |
| ------------- | ---------- | -------------------------------- | ---------------------------------- |
| MNIST         | 8          | 3.99                             | 5.71                               |
| MNIST         | 4          | 7.48                             | 3.62                               |
| Fashion MNIST | 8          | 11.78                            | 4.91                               |
| Fashion MNIST | 4          | 9.17                             | 9.25                               |
| CIFAR-10      | 8          | 46.14                            | 12.61                              |
| CIFAR-10      | 4          | 51.46                            | 8.63                               |
| CelebA        | 4          | 36.08                            | 66.52                              |

# Citation

Please cite our paper if you use this code in your own work:

```
@article{FSDDIM,
  title={Fully Spiking Denoising Diffusion Implicit Models},
  author={Watanabe, Ryo and Mukuta, Yusuke and Harada, Tatsuya},
  journal={arXiv preprint arXiv:2312.01742},
  year={2023}
}
```
