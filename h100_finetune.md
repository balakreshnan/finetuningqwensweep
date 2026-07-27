# Finetuning Setup and run a fine tune job

## Requirements

- Colossus baremetal lease
- Choosing ubuntu 26.04 OS version
- login using ssh local-username@IP address

## Steps to get the environment going

### Install nvidia driver and cuda toolkits

- First check if nvidia-smi driver exist.

```
nvidia-smi
```

- if no found

```
# Option 1: Install the latest available driver (595)
sudo apt update
sudo apt install -y nvidia-driver-595 nvidia-utils-595

# Reboot to load the kernel module
sudo reboot
```

### Now setup torch installations

- First install the python pip and environment
- Create a python virtual environment
- Install torch and validate if can run

```
# Install Python & pip
sudo apt install -y python3 python3-pip python3-venv git

# Create a virtual environment
python3 -m venv ~/finetune-env
source ~/finetune-env/bin/activate

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Verify GPU
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Now fine tuning

- Let's now install necessary fine tuning libraries

```
pip install transformers       # Hugging Face Transformers
pip install datasets           # Hugging Face Datasets
pip install accelerate         # Distributed training launcher
pip install peft               # LoRA / QLoRA / parameter-efficient FT
pip install bitsandbytes       # 4-bit / 8-bit quantization for QLoRA
pip install trl                # SFT / RLHF trainers
pip install wandb              # Experiment tracking (optional)
pip install flash-attn --no-build-isolation  # FlashAttention-2 for H100
```

- install and login huggingface

```
pip install huggingface_hub
huggingface-cli login
```

- Login into wandb for logging metrics

```
wandb login
```

- Copy the files for fine tuning

```
scp -r C:\Code\finetuning\finetuningqwenlocal local-username@<IP address>:
```

- now run the fine tuning job

```
python finetune_qwen3b_4runs_wandb_fixed.py \
  --run_experiments \
  --output_dir ./output/qwen3b-oasst1-lora \
  --hub_namespace Balab2021 \
  --hub_repo_prefix Qwen2.5-3B-OASST1-QLoRA \
  --wandb_project qwen3b-finetuning \
  --eval_batch_size 1
```

- wait for job to complete
- After job is done, model and metrics should be uploaded to huggingface, wandb.