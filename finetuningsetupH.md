# Fine tuning setup for H Series and B Series for Colossus Bare metal

## nvidia-smi installation

- First when you ssh check if nvidia-smi is installed
- Every time when i installed ubuntu 26.04 i didn't see nvidia-smi

```
# Option 1: Install the latest available driver (595)
sudo apt update
sudo apt install -y nvidia-driver-595 nvidia-utils-595

# Reboot to load the kernel module
sudo reboot
```

- once rebooted

```
nvidia-smi
```

- you should see the GPU listed
- Now to install python environment

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

- now to fine tune
- install the necessary libraries

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

- ignore if flash attention fails

- install huggingface

```
pip install huggingface_hub
```

- login into huggingface

```
hf auth login --token "xxxxxx"
```

- login into wandb login

```
wandb login
```

- provide the key to authenticate
- copy the source files from local machine into GPU Compute VM
- Next run the fine tuning job

```
python finetune_qwen3b_4runs_wandb_fixed.py \
  --run_experiments \
  --output_dir ./output/qwen3b-oasst1-lora \
  --hub_namespace Balab2021 \
  --hub_repo_prefix Qwen2.5-3B-OASST1-QLoRA_h200 \
  --wandb_project qwen3b-finetuning \
  --eval_batch_size 1
```

- wath the GPU consumption

```
watch -n 1 nvidia-smi
```

- wait for it to complete
- Next Run the mulitmodal finetuning

```
python finetune_qwen3_vl_2b_llava_wandb_fixed_v2.py \
  --hub-repo-id Balab2021/Qwen3-VL-2B-LLaVA-LoRA \
  --max-train-samples 10000 \
  --max-eval-samples 500 \
  --num-train-epochs 1 \
  --learning-rate 2e-5 \
  --max-length 2048 \
  --per-device-train-batch-size 8 \
  --per-device-eval-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --num-workers 8 \
  --attn-implementation sdpa
```