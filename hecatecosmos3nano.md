# Nemo Cosmos 3 Nano Finetuning in Vera Rubin

## Introduction

- Fine tung cosmos 3 nano multimodal.
- to see can we fine tune with custom dataset

## Steps to Reproduce

- First login into the Vera Rubin
- use slurm to get the actual gpu compute
- Create the credential folders

```
mkdir -p ~/.config/enroot
touch ~/.config/enroot/.credentials
```

- Set up the credentials

```
cat > ~/.config/enroot/.credentials << 'EOF'
machine x.com login xxxx@nvidxxxia.com password xxxxx
machine nvcrdoc.io login $oauthtoken password xxxxxxx
EOF

```
srun --account=general_sa --partition=backfill-spx --nodes=1 \
     --time=02:00:00 --mem=64G \
     --container-image=nvcr.io#nvidia/nemo:25.04 \
     --job-name=general_sa-finetune:interactive \
     --container-mounts="/lustre:/lustre" \
     --pty bash
```

```
pip install huggingface_hub wandb
```


- Create the directory

```
mkdir -p /lustre/fsw/general_sa/bbalakreshna/cosmos3
```

- Create the download dataset folder

```
cat > /lustre/fsw/general_sa/bbalakreshna/cosmos3/download_dataset.py << 'EOF'
import os
from huggingface_hub import snapshot_download

# Set your Hugging Face token
os.environ["HF_TOKEN"] = "hf_YOUR_TOKEN"  # Replace with your actual token

BASE_DIR = "/lustre/fsw/general_sa/bbalakreshna/cosmos3"

# Create directories
os.makedirs(f"{BASE_DIR}/data", exist_ok=True)
os.makedirs(f"{BASE_DIR}/models", exist_ok=True)
os.makedirs(f"{BASE_DIR}/checkpoints", exist_ok=True)

# Download small test dataset first (validate pipeline)
print("📥 Downloading dataset...")
dataset_path = snapshot_download(
    repo_id="sayakpaul/ucf101-subset",
    repo_type="dataset",
    local_dir=f"{BASE_DIR}/data/cosmos3_dataset",
    token=os.environ["HF_TOKEN"]
)
print(f"✅ Dataset downloaded to: {dataset_path}")

# Download Cosmos 3 Nano base model
print("📥 Downloading Cosmos 3 Nano model...")
model_path = snapshot_download(
    repo_id="nvidia/Cosmos-Predict2-2B-Video2World",
    local_dir=f"{BASE_DIR}/models/cosmos3-nano",
    token=os.environ["HF_TOKEN"]
)
print(f"✅ Model downloaded to: {model_path}")

print("\n🎉 All downloads complete!")
print(f"   Dataset: {BASE_DIR}/data/cosmos3_dataset")
print(f"   Model:   {BASE_DIR}/models/cosmos3-nano")
EOF
```

- run the download

```
python /lustre/fsw/general_sa/bbalakreshna/cosmos3/download_dataset.py
```

- create the yaml config file for fine tuning cosmos 3 nano

```
cat > /lustre/fsw/general_sa/bbalakreshna/cosmos3/configs/cosmos3_nano_finetune.yaml << 'EOF'
# Cosmos 3 Nano Fine-Tuning Configuration
# ----------------------------------------

model:
  pretrained_model_path: /lustre/fsw/general_sa/bbalakreshna/cosmos3/models/cosmos3-nano
  model_type: cosmos_predict2_2b

trainer:
  devices: 8
  num_nodes: 1
  max_epochs: 3
  precision: bf16-mixed
  accelerator: gpu
  strategy: ddp
  val_check_interval: 100
  gradient_clip_val: 1.0
  log_every_n_steps: 10


  train_dir: /lustre/fsw/general_sa/bbalakreshna/cosmos3/data/cosmos3_dataset
  batch_size: 4
  num_workers: 8
  max_seq_length: 512

optimizer:
  name: adamw
  lr: 2e-5
  weight_decay: 0.01
  betas: [0.9, 0.999]

scheduler:
  name: cosine
  warmup_steps: 100
  min_lr: 1e-6

peft:
  enabled: true
  method: lora
  lora_rank: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules:
    - q_proj
    - v_proj
    - k_proj
    - o_proj

exp_manager:
  exp_dir: /lustre/fsw/general_sa/bbalakreshna/cosmos3/checkpoints
  create_wandb_logger: true
  wandb_logger_kwargs:
    project: cosmos3-nano-finetune
    name: cosmos3-nano-lora-verarubin
    entity: bbalakreshna
  checkpoint_callback_params:
    save_top_k: 3
    monitor: val_loss
    mode: min
  resume_if_exists: true
EOF
```

- now create the fine tuning python file

```
cat > /lustre/fsw/general_sa/bbalakreshna/cosmos3/finetune_cosmos3_nano.py << 'EOF'
import os
import sys
import torch
import wandb
import yaml
import glob
from pathlib import Path
from datetime import datetime

# ============================================================
# Fine-Tuning Script for Cosmos 3 Nano
# Uses NeMo / PyTorch with LoRA + wandb logging
# ============================================================

def load_config(config_path):
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_wandb(config):
    """Initialize Weights & Biases logging."""
    wandb_config = config.get("exp_manager", {}).get("wandb_logger_kwargs", {})
    wandb.init(
        project=wandb_config.get("project", "cosmos3-nano-finetune"),
        name=wandb_config.get("name", f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"),
        entity=wandb_config.get("entity", None),
        config=config,
    )
    print(f"✅ wandb initialized: {wandb.run.url}")
    return wandb.run


def setup_data(config):
    """Load and prepare dataset."""
    from torch.utils.data import Dataset, DataLoader
    import json

    data_config = config["data"]
    train_dir = data_config["train_dir"]

    class VideoTextDataset(Dataset):
        def __init__(self, data_dir, max_seq_length=512):
            self.data_dir = data_dir
            self.max_seq_length = max_seq_length
            # Find all video files
            self.video_files = sorted(
                glob.glob(f"{data_dir}/**/*.mp4", recursive=True)
                + glob.glob(f"{data_dir}/**/*.avi", recursive=True)
                + glob.glob(f"{data_dir}/**/*.webm", recursive=True)
            )
            # If no videos, look for manifest
            manifest_path = os.path.join(data_dir, "train_manifest.jsonl")
            if os.path.exists(manifest_path):
                self.entries = []
                with open(manifest_path) as f:
                    for line in f:
                        self.entries.append(json.loads(line))
            else:
                self.entries = [{"video_path": v, "text": ""} for v in self.video_files]

            print(f"📂 Dataset loaded: {len(self.entries)} samples from {data_dir}")

        def __len__(self):
            return len(self.entries)

        def __getitem__(self, idx):
            entry = self.entries[idx]
            # Return entry dict — model-specific processing happens downstream
            return entry

    dataset = VideoTextDataset(train_dir, data_config.get("max_seq_length", 512))
    dataloader = DataLoader(
        dataset,
        batch_size=data_config.get("batch_size", 4),
        shuffle=True,
        num_workers=data_config.get("num_workers", 8),
        pin_memory=True,
    )
    return dataloader


def setup_model(config):
    """Load Cosmos 3 Nano model with LoRA."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    model_config = config["model"]
    peft_config = config.get("peft", {})

    model_path = model_config["pretrained_model_path"]
    print(f"📦 Loading model from: {model_path}")

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Apply LoRA if enabled
    if peft_config.get("enabled", True):
        lora_config = LoraConfig(
            r=peft_config.get("lora_rank", 16),
            lora_alpha=peft_config.get("lora_alpha", 32),
            lora_dropout=peft_config.get("lora_dropout", 0.05),
            target_modules=peft_config.get("target_modules", ["q_proj", "v_proj"]),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model


def train(config):
    """Main training loop."""
    # Setup
    run = setup_wandb(config)
    dataloader = setup_data(config)
    model = setup_model(config)

    trainer_config = config["trainer"]
    opt_config = config["optimizer"]
    sched_config = config["scheduler"]
    exp_config = config["exp_manager"]

    # Move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_config.get("lr", 2e-5),
        weight_decay=opt_config.get("weight_decay", 0.01),
        betas=tuple(opt_config.get("betas", [0.9, 0.999])),
    )

    # Scheduler
    total_steps = len(dataloader) * trainer_config.get("max_epochs", 3)
    warmup_steps = sched_config.get("warmup_steps", 100)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=sched_config.get("min_lr", 1e-6)
    )

    # Training loop
    global_step = 0
    best_loss = float("inf")
    checkpoint_dir = exp_config.get("exp_dir", "/lustre/fsw/general_sa/bbalakreshna/cosmos3/checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"\n🚀 Starting training: {trainer_config.get('max_epochs', 3)} epochs")
    print(f"   Total steps: {total_steps}")
    print(f"   Device: {device}")
    print(f"   GPUs: {torch.cuda.device_count()}\n")

    model.train()
    for epoch in range(trainer_config.get("max_epochs", 3)):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()

            # Forward pass (adapt based on actual model API)
            outputs = model(**{k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()})
            loss = outputs.loss

            # Backward pass
            loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), trainer_config.get("gradient_clip_val", 1.0)
            )

            optimizer.step()
            if global_step >= warmup_steps:
                scheduler.step()

            # Logging
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            if global_step % trainer_config.get("log_every_n_steps", 10) == 0:
                avg_loss = epoch_loss / num_batches
                current_lr = optimizer.param_groups[0]["lr"]

                wandb.log({
                    "train/loss": loss.item(),
                    "train/avg_loss": avg_loss,
                    "train/learning_rate": current_lr,
                    "train/grad_norm": grad_norm.item(),
                    "train/epoch": epoch,
                    "train/global_step": global_step,
                    "system/gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
                    "system/gpu_memory_reserved_gb": torch.cuda.memory_reserved() / 1e9,
                }, step=global_step)

                print(f"  Step {global_step}/{total_steps} | Loss: {loss.item():.4f} | "
                      f"Avg Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | "
                      f"GPU Mem: {torch.cuda.memory_allocated()/1e9:.1f}GB")

        # End of epoch
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        print(f"\n📊 Epoch {epoch+1}/{trainer_config.get('max_epochs', 3)} — Avg Loss: {avg_epoch_loss:.4f}\n")

        wandb.log({
            "epoch/avg_loss": avg_epoch_loss,
            "epoch/epoch": epoch + 1,
        }, step=global_step)

        # Save checkpoint
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_ckpt_path = os.path.join(checkpoint_dir, "best_model")
            model.save_pretrained(best_ckpt_path)
            print(f"  💾 Best model saved to: {best_ckpt_path}")
            wandb.log({"best_val_loss": best_loss}, step=global_step)

        # Save epoch checkpoint
        epoch_ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}")
        model.save_pretrained(epoch_ckpt_path)
        print(f"  💾 Epoch checkpoint saved to: {epoch_ckpt_path}")

    # Final summary
    wandb.run.summary["final_loss"] = avg_epoch_loss
    wandb.run.summary["best_loss"] = best_loss
    wandb.run.summary["total_steps"] = global_step
    wandb.finish()

    print(f"\n🎉 Training complete!")
    print(f"   Best loss: {best_loss:.4f}")
    print(f"   Best model: {checkpoint_dir}/best_model")
    print(f"   wandb run: {run.url}")

    return checkpoint_dir


if __name__ == "__main__":
    config_path = os.environ.get(
        "CONFIG_PATH",
        "/lustre/fsw/general_sa/bbalakreshna/cosmos3/configs/cosmos3_nano_finetune.yaml"
    )
    print(f"📋 Loading config: {config_path}")
    config = load_config(config_path)
    train(config)
EOF
```

- upload to huggingface

```
cat > /lustre/fsw/general_sa/bbalakreshna/cosmos3/upload_to_hf.py << 'EOF'
import os
from huggingface_hub import HfApi, create_repo

# ============================================================
# Upload Fine-Tuned Cosmos 3 Nano to Hugging Face
# ============================================================

# Configuration — UPDATE THESE
HF_TOKEN = os.environ.get("HF_TOKEN", "hf_YOUR_TOKEN")
HF_USERNAME = "bbalakreshna"  # Your HF username
REPO_NAME = "cosmos3-nano-finetuned"
MODEL_DIR = "/lustre/fsw/general_sa/bbalakreshna/cosmos3/checkpoints/best_model"

repo_id = f"{HF_USERNAME}/{REPO_NAME}"

print(f"📤 Uploading model to: https://huggingface.co/{repo_id}")

# Create repo
api = HfApi(token=HF_TOKEN)
create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True, token=HF_TOKEN)

# Create model card
model_card = """---
license: other
license_name: nvidia-open-model-license
base_model: nvidia/Cosmos-Predict2-2B-Video2World
tags:
  - cosmos
  - physical-ai
  - video-generation
  - fine-tuned
  - lora
  - nvidia
  - vera-rubin
---

# Cosmos 3 Nano — Fine-Tuned (LoRA)

Fine-tuned version of [NVIDIA Cosmos Predict2 2B](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World)
using LoRA on NVIDIA Vera Rubin (Hecate) cluster.

## Training Details
- **Base Model:** nvidia/Cosmos-Predict2-2B-Video2World
- **Method:** LoRA (rank=16, alpha=32)
- **Hardware:** 8x Vera Rubin GPUs (Hecate cluster)
- **Framework:** NeMo + PyTorch + PEFT
- **Epochs:** 3
- **Learning Rate:** 2e-5
- **Optimizer:** AdamW
- **Precision:** BF16

## Usage

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("nvidia/Cosmos-Predict2-2B-Video2World")
model = PeftModel.from_pretrained(base_model, "bbalakreshna/cosmos3-nano-finetuned")
```

- next will SLURM batch script

```
## Step 9: Create the Slurm Batch Script (Full Pipeline)

```bash
cat > /lustre/fsw/general_sa/bbalakreshna/cosmos3/run_finetune.sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=cosmos3-nano-ft
#SBATCH --account=<your-PPP>
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/lustre/fsw/general_sa/bbalakreshna/cosmos3/logs/cosmos3-ft-%j.out
#SBATCH --error=/lustre/fsw/general_sa/bbalakreshna/cosmos3/logs/cosmos3-ft-%j.err

# ============================================================
# Cosmos 3 Nano Fine-Tuning on Vera Rubin (Hecate)
# ============================================================

# --- CREDENTIALS (REPLACE THESE) ---
export HF_TOKEN="hf_YOUR_TOKEN"
export WANDB_API_KEY="YOUR_WANDB_KEY"

# --- PATHS ---
export BASE_DIR="/lustre/fsw/general_sa/bbalakreshna/cosmos3"
export CONFIG_PATH="${BASE_DIR}/configs/cosmos3_nano_finetune.yaml"

# --- CREATE LOG DIR ---
mkdir -p ${BASE_DIR}/logs

echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "GPUs:   ${SLURM_GPUS_ON_NODE}"
echo "Start:  $(date)"
echo "============================================"

# --- RUN TRAINING ---
srun --container-image=nvcr.io#nvidia/nemo:25.04 \
     --container-mounts="/lustre:/lustre" \
     --no-container-mount-home \
     bash -c "
       echo '📦 Installing dependencies...'
       pip install -q huggingface_hub wandb peft transformers accelerate datasets
       
       echo '🚀 Starting fine-tuning...'
       cd ${BASE_DIR}
       python finetune_cosmos3_nano.py
       
       echo '📤 Uploading model to Hugging Face...'
       python upload_to_hf.py
       
       echo '✅ Done!'
     "

echo "============================================"
echo "End:    $(date)"
echo "============================================"
EOF
```

- create log directories
- submit the job
- make sure to note the jobid number

```
mkdir -p /lustre/fsw/general_sa/bbalakreshna/cosmos3/logs
mkdir -p /lustre/fsw/general_sa/bbalakreshna/cosmos3/configs
sbatch /lustre/fsw/general_sa/bbalakreshna/cosmos3/run_finetune.sbatch
```

- Monitor the job

```
# Check job status
squeue -u $USER

# Watch logs (replace JOB_ID)
tail -f /lustre/fsw/general_sa/bbalakreshna/cosmos3/logs/cosmos3-ft-<JOB_ID>.out

# Check wandb dashboard
echo "https://wandb.ai/bbalakreshna/cosmos3-nano-finetune"
```

- next will be to validate the accuracy