# GH200 Ubuntu Setup Guide for Qwen Fine-Tuning

This guide documents installing NVIDIA drivers, fixing GH200 memory
hotplug, creating a Python environment, installing PyTorch and
fine-tuning dependencies, authenticating with Hugging Face and Weights &
Biases, and running the training script.

### Model used

- [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)

## 1. Verify GPU

``` bash
lspci | grep -i nvidia
```

## 2. Install Driver

``` bash
sudo apt update
sudo apt install nvidia-driver-595-server-open
sudo reboot
```

## 3. Verify Driver

``` bash
modinfo nvidia | grep -E '^license:|^version:'
cat /proc/driver/nvidia/version
nvidia-smi
```

## 4. GH200 Memory Hotplug Fix

If `nvidia-smi` reports 'No devices were found' and `dmesg` reports
`online_movable`:

``` bash
echo online_movable | sudo tee /sys/devices/system/memory/auto_online_blocks
sudo modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia
sudo modprobe nvidia
sudo modprobe nvidia_uvm
```

Optionally make it persistent using a systemd service.

## 5. Python Environment

``` bash
sudo apt install -y python3 python3-venv python3-pip git
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

## 6. Install PyTorch

``` bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## 7. Install Dependencies

``` bash
pip install trl transformers accelerate peft bitsandbytes datasets huggingface_hub wandb sentencepiece pillow
```

If needed:

``` bash
pip install -U "transformers @ git+https://github.com/huggingface/transformers.git@main"
```

## 8. Authenticate

``` bash
huggingface-cli login
wandb login
```

or set `HF_TOKEN` and `WANDB_API_KEY`.

## 9. Verify CUDA

``` bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

## 10. Run Training

``` bash
python finetune_qwen35_2b_trl_wandb.py --hub-repo-id Balab2021/Qwen3.5-2B-Capybara-LoRA --max-train-samples 10000 --max-eval-samples 500 --num-train-epochs 1 --learning-rate 1e-4 --max-length 2048
```

- This was trial to optimize the run. 4 hours and 45 mins to run.

```
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python finetune_qwen35_2b_trl_wandb.py \
  --hub-repo-id Balab2021/Qwen3.5-2B-Capybara-LoRA \
  --max-train-samples 10000 \
  --max-eval-samples 200 \
  --num-train-epochs 1 \
  --learning-rate 1e-4 \
  --max-length 2048 \
  --per-device-train-batch-size 8 \
  --per-device-eval-batch-size 4 \
  --gradient-accumulation-steps 2 \
  --packing \
  --eval-steps 250 \
  --save-steps 250
```