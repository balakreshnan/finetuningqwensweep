# GRPO Multi-Node Fine-Tuning Skill (16 GPUs)

## Purpose
Fine-tune language models using GRPO (Group Relative Policy Optimization) across multiple nodes on the HECATE cluster (4 nodes × 4 GPUs = 16 GPUs), then generate analytics comparing results.

---

## Prerequisites

- Access to HECATE cluster (`batch-xdr` partition)
- Account: `general_sa`
- Container: `gitlab-master.nvidia.com/dl/dgx/pytorch:main-py3-devel`
- Lustre storage: `/lustre/fsw/general_sa/bbalakreshna/`
- Hugging Face token (`HF_TOKEN`)
- Weights & Biases API key (`WANDB_API_KEY`)

---

## Step 1: Create the Launch Script

Save this as `/lustre/fsw/general_sa/bbalakreshna/launch_sweep.sh`:

```bash
#!/bin/bash
set -e

echo "$(hostname): Installing dependencies..."
pip install --quiet --cache-dir=/lustre/fsw/general_sa/bbalakreshna/.pip_cache \
    transformers datasets accelerate peft bitsandbytes trl huggingface_hub wandb

# HF_TOKEN and WANDB_API_KEY passed from login node via --export=ALL
echo "$(hostname): Logging into Hugging Face..."
huggingface-cli login --token $HF_TOKEN

echo "$(hostname): W&B API key set via WANDB_API_KEY env var"

echo "$(hostname): Launching training..."
torchrun \
  --nnodes=$SLURM_JOB_NUM_NODES \
  --nproc_per_node=4 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  /lustre/fsw/general_sa/bbalakreshna/grpo_5runs_sweep.py
```

Make executable:
```bash
chmod +x /lustre/fsw/general_sa/bbalakreshna/launch_sweep.sh
```

---

## Step 2: Set Environment Variables

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
export WANDB_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

Or read from secure files:
```bash
export HF_TOKEN=$(cat ~/.hf_token)
export WANDB_API_KEY=$(cat ~/.wandb_key)
```

---

## Step 3: Launch Multi-Node Training

```bash
srun --account=general_sa \
     --partition=batch-xdr \
     --nodes=4 \
     --ntasks-per-node=1 \
     --time=5:00:00 \
     --job-name=general_sa-grpo-sweep \
     --container-image=gitlab-master.nvidia.com/dl/dgx/pytorch:main-py3-devel \
     --container-mount-home \
     --container-mounts=/lustre:/lustre \
     --no-container-remap-root \
     --mpi=pmix \
     --export=ALL \
     /lustre/fsw/general_sa/bbalakreshna/launch_sweep.sh
```

### Key Flags Explained

| Flag | Purpose |
|------|---------|
| `--nodes=4` | 4 compute nodes |
| `--ntasks-per-node=1` | 1 torchrun launcher per node (spawns 4 GPU workers) |
| `--container-mounts=/lustre:/lustre` | Mount lustre inside container (critical — without this, scripts on lustre are invisible) |
| `--container-mount-home` | Mount home dir for tokens/configs |
| `--no-container-remap-root` | Run as root inside container |
| `--mpi=pmix` | Enable MPI for inter-node communication |
| `--export=ALL` | Pass env vars (HF_TOKEN, WANDB_API_KEY) into container |

### Do NOT Use
- `--gpus-per-node=4` → Invalid gres error on HECATE. Omit entirely — nodes provide all GPUs by default.

---

## Step 4: Monitor & Manage Jobs

```bash
# Check job status
squeue -u bbalakreshna

# Detailed view
squeue -u bbalakreshna -O JobId,Name,State,NodeList,TimeUsed

# Job history
sacct -u bbalakreshna --starttime=today --format=JobID,JobName,State,ExitCode,Start,End,NodeList

# Cancel a job
scancel <job_id>

# Cancel all jobs
scancel -u bbalakreshna
```

---

## Step 5: Generate Analytics Report

After the sweep completes, generate a comparison report. The analytics should include:

### 5a. Results Summary Table

```markdown
| Run | Config | Accuracy | Loss | Time (min) | W&B |
|-----|--------|----------|------|------------|-----|
| run1-baseline | LR 5e-6, LoRA r=16, 4 gen | XX% | X.XXX | X.X | [link] |
| run2-high-lr-more-gen | LR 2e-5, LoRA r=32, 8 gen | XX% | X.XXX | X.X | [link] |
| run3-moderate-lr | LR 1e-5, LoRA r=32, 4 gen | XX% | X.XXX | X.X | [link] |
| run4-large-lora | LR 2e-6, LoRA r=64, 4 gen | XX% | X.XXX | X.X | [link] |
| run5-max-exploration | LR 8e-6, LoRA r=32, 16 gen | XX% | X.XXX | X.X | [link] |
```

### 5b. Multi-Node vs Single-Node Comparison

Compare accuracy, training time, and speedup:

```markdown
| Run | 4 GPU Accuracy | 16 GPU Accuracy | 4 GPU Time | 16 GPU Time | Speedup |
|-----|---------------|-----------------|------------|-------------|---------|
```

### 5c. Key Metrics to Track

- **Accuracy** — eval on held-out GSM8K problems
- **Training loss** — negative = reward improving beyond reference policy
- **Wall-clock time** — total training time per run
- **Speedup** — 4 GPU time / 16 GPU time per run

---

## Sweep Configurations (Recommended)

Based on empirical results from this project:

### Best for Multi-Node (16 GPU) — run4-large-lora 🏆

```yaml
learning_rate: 2e-6
lora_r: 64
lora_alpha: 128
num_generations: 4
num_samples: 1000
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
num_train_epochs: 1
max_completion_length: 512
```

**Result:** 76% accuracy in 5.4 minutes

### Best for Single-Node (4 GPU) — run3-moderate-lr

```yaml
learning_rate: 1e-5
lora_r: 32
lora_alpha: 64
num_generations: 4
num_samples: 1000
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
num_train_epochs: 1
max_completion_length: 512
```

**Result:** 76% accuracy in 21 minutes

---

## Scaling Insights

1. **Low LR + large LoRA (run4)** scales best to multi-node — higher effective batch size favors conservative updates
2. **High generation count (run5)** benefits from more GPUs — more parallel sampling
3. **Moderate LR configs (run3)** may need LR adjustment when scaling (apply linear scaling rule)
4. **Average speedup: 2.7×** going from 4 → 16 GPUs (range 2.3×–3.9×)
5. **Linear LR scaling rule:** `new_lr = base_lr × (new_gpus / base_gpus)` — consider when configs don't transfer

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `EADDRINUSE` on port | Previous run left a zombie process | Change `$MASTER_PORT` (e.g., 29500, 39500) |
| `No such file or directory` for .py | Lustre not mounted in container | Add `--container-mounts=/lustre:/lustre` |
| `Invalid user token` (HF) | Token placeholder not replaced | Set real `HF_TOKEN` env var + use `--export=ALL` |
| `Invalid generic resource (gres)` | `--gpus-per-node` not supported | Remove that flag entirely |
| Hangs at rendezvous | Stale rendezvous or port conflict | Add `--rdzv_id=$SLURM_JOB_ID` to torchrun args |
| Job cancelled / task failure | One node crashed, killed the rest | Check `sacct` for the failed node, re-run |

---

## File Locations

| Item | Path |
|------|------|
| Training script | `/lustre/fsw/general_sa/bbalakreshna/grpo_5runs_sweep.py` |
| Launch wrapper | `/lustre/fsw/general_sa/bbalakreshna/launch_sweep.sh` |
| Results output | `/lustre/fsw/general_sa/bbalakreshna/grpo-results-sweep/` |
| Pip cache | `/lustre/fsw/general_sa/bbalakreshna/.pip_cache/` |
| HF token (secure) | `~/.hf_token` |
| W&B key (secure) | `~/.wandb_key` |
| W&B Project | https://wandb.ai/balabala76/qwen3-grpo-math |
