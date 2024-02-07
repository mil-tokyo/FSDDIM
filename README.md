# Fully Spiking Denoising Diffusion Implicit Models

Implementation of FSDDIM

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
accelerate launch --num_processes=4 --gpu_ids=0,1,2,3 --mixed_precision fp16 main.py -c <config_file> -n <experiment_name>
```

- args
  - `-c`, `--config`
    - Path to the config file.
    - Sample config files can be found in `configs/`.
  - `-n`, `--name`
    - Experiment name.
    - Please specify a unique name because generated images will be saved in `output/<experiment_name>`.
