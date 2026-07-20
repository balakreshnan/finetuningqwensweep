# Fine-Tune Qwen3-VL-2B-Instruct on `HuggingFaceH4/llava-instruct-mix-vsft`

This project fine-tunes **`Qwen/Qwen3-VL-2B-Instruct`** on the multimodal dataset **`HuggingFaceH4/llava-instruct-mix-vsft`** using Hugging Face Transformers, TRL, PEFT/QLoRA, bitsandbytes, Weights & Biases, and the Hugging Face Hub.

## Files

```text
finetune_qwen3_vl_2b_llava_wandb_fixed_v2.py
requirements_qwen3_vl.txt
README_Qwen3_VL_LLaVA_Finetuning.md
```

Use the `fixed_v2` script. It includes:

- A Python 3.14-compatible top-level multimodal data collator
- Image/text batching
- Protection against image-token mismatch caused by truncation
- 4-bit QLoRA
- W&B logging
- Evaluation
- Hugging Face Hub upload

## Model

`Qwen/Qwen3-VL-2B-Instruct` is a vision-language model for:

- Visual question answering
- Image captioning
- OCR and document understanding
- Chart and diagram reasoning
- Image-grounded instruction following

## Dataset

`HuggingFaceH4/llava-instruct-mix-vsft` contains image-and-text conversations. Typical examples contain:

```python
{
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image."}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "The image shows..."}
            ]
        }
    ],
    "images": [PIL.Image.Image(...)]
}
```

The script loads the `train` split for training and the `test` split for evaluation.

## GH200 prerequisite

Verify the GPU:

```bash
nvidia-smi
```

If GH200 reports `No devices were found` and `dmesg` mentions `online_movable`, run:

```bash
echo online_movable | sudo tee /sys/devices/system/memory/auto_online_blocks

sudo modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia
sudo modprobe nvidia
sudo modprobe nvidia_uvm
```

Then verify again:

```bash
nvidia-smi
```

## Create a clean Python 3.14 environment

```bash
cd ~/finetuning

deactivate 2>/dev/null || true
rm -rf .venv

python3.14 -m venv .venv
source .venv/bin/activate

python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools wheel packaging ninja psutil
```

Do not use `sudo pip` inside the virtual environment.

## Install CUDA-enabled PyTorch

```bash
python -m pip install   torch torchvision torchaudio   --index-url https://download.pytorch.org/whl/cu128
```

Verify:

```bash
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("BF16 supported:", torch.cuda.is_bf16_supported())
PY
```

## Install requirements

```bash
python -m pip install -r requirements_qwen3_vl.txt
```

Or install manually:

```bash
python -m pip install   trl peft accelerate datasets bitsandbytes   wandb huggingface_hub pillow sentencepiece hf_xet
```

Install the latest Transformers:

```bash
python -m pip install -U   "transformers @ git+https://github.com/huggingface/transformers.git@main"
```

## Configure credentials

```bash
export HF_TOKEN="hf_your_token"
export WANDB_API_KEY="your_wandb_key"
export WANDB_PROJECT="qwen3-vl-finetuning"
```

## Smoke test

```bash
python finetune_qwen3_vl_2b_llava_wandb_fixed_v2.py   --hub-repo-id Balab2021/Qwen3-VL-2B-LLaVA-LoRA-test   --max-train-samples 100   --max-eval-samples 20   --max-steps 10   --learning-rate 2e-5   --max-length 2048   --per-device-train-batch-size 1   --per-device-eval-batch-size 1   --gradient-accumulation-steps 4   --num-workers 4   --attn-implementation sdpa
```

## Recommended GH200 run

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python finetune_qwen3_vl_2b_llava_wandb_fixed_v2.py   --hub-repo-id Balab2021/Qwen3-VL-2B-LLaVA-LoRA   --max-train-samples 10000   --max-eval-samples 500   --num-train-epochs 1   --learning-rate 2e-5   --max-length 2048   --per-device-train-batch-size 4   --per-device-eval-batch-size 1   --gradient-accumulation-steps 4   --num-workers 8   --attn-implementation sdpa   --eval-steps 250   --save-steps 250   --logging-steps 10
```

Effective batch size:

```text
4 × 4 × 1 GPU = 16
```

## Increase GPU utilization

Prefer increasing the real micro-batch before increasing gradient accumulation.

Suggested progression:

```text
batch 4, accumulation 4
batch 8, accumulation 2
batch 16, accumulation 1
```

Monitor with:

```bash
nvidia-smi dmon -s pucm
```

A GH200 may still show moderate utilization because a 2B QLoRA workload is small relative to the GPU.

## Attention backend

Default:

```bash
--attn-implementation sdpa
```

Optional:

```bash
--attn-implementation flash_attention_2
```

Flash Attention requires a compatible local CUDA toolkit, compiler, and wheel or source build. If installation fails, use SDPA.

## Important multimodal truncation note

The fixed script disables processor-level truncation because truncating a vision-language sequence can cut through expanded image-token blocks and cause:

```text
Mismatch in image token count between text and input_ids
```

Control memory with:

- Smaller micro-batch
- Gradient checkpointing
- Lower image resolution
- Smaller evaluation batch
- Fewer visual tokens

## DataLoader workers

The fixed script uses a top-level callable collator compatible with Python 3.14 multiprocessing.

Recommended:

```bash
--num-workers 4
```

or:

```bash
--num-workers 8
```

For troubleshooting:

```bash
--num-workers 0
```

## Common issues

### CUDA out of memory

Reduce:

```bash
--per-device-train-batch-size
```

Increase:

```bash
--gradient-accumulation-steps
```

Keep gradient checkpointing enabled and consider:

```bash
--max-pixels 401408
```

### Low GPU utilization

Try:

```bash
--per-device-train-batch-size 8
--gradient-accumulation-steps 2
--num-workers 8
```

Also reduce evaluation and checkpoint frequency.

### `ModuleNotFoundError: torch`

Activate the virtual environment and install PyTorch:

```bash
source ~/finetuning/.venv/bin/activate

python -m pip install   torch torchvision torchaudio   --index-url https://download.pytorch.org/whl/cu128
```

### Flash Attention build failure

Use:

```bash
--attn-implementation sdpa
```

## Outputs

The output directory contains:

```text
adapter_config.json
adapter_model.safetensors
processor files
tokenizer files
trainer_state.json
train_results.json
eval_results.json
run_summary.json
checkpoint-*/
```

The LoRA adapter and processor are pushed to the repository specified by:

```bash
--hub-repo-id Balab2021/Qwen3-VL-2B-LLaVA-LoRA
```

## Final validation

Before a full run, verify:

```bash
nvidia-smi
```

```bash
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
PY
```

At the end of training, confirm:

- The W&B run contains training and evaluation metrics
- The local output directory contains adapter files
- The Hugging Face repository contains the uploaded LoRA adapter
