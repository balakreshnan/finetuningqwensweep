# Skill: NeMo AutoModel LoRA Fine-Tuning

## Description
Fine-tune large language models (LLMs) using LoRA (Low-Rank Adaptation) with NVIDIA NeMo AutoModel backend. Supports Nemotron, Llama, Qwen, and other HuggingFace-compatible models. Generates YAML configs, runs distributed training via `torchrun`, and uploads adapters to HuggingFace Hub.

## When to Use
- User asks to fine-tune a model with LoRA / QLoRA / PEFT
- User asks to create a NeMo AutoModel fine-tuning config
- User asks to train / fine-tune with nemo_automodel
- User asks to fine-tune Nemotron, Llama, Qwen, or similar models
- User wants to upload LoRA adapters to HuggingFace

## Prerequisites
- **Container / Environment**: NeMo AutoModel installed (`nemo_automodel` package)
- **Hardware**: 1+ NVIDIA GPUs (tested on 4× GPU setups)
- **Data**: JSONL training file in chat format
- **HuggingFace**: Account + token for model download and adapter upload

---

## Step 1: Verify the Environment

Run these checks before starting:

```bash
# Check nemo_automodel is installed
python -c "import nemo_automodel; print('nemo_automodel OK')"

# Check available GPUs
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Count GPUs (needed for --nproc_per_node)
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
```

---

## Step 2: Inspect the Model's PEFT Config

**CRITICAL**: Do NOT guess LoRA field names or target module names. Always introspect first.

### 2a. Get PeftConfig field names

```bash
python -c "
from nemo_automodel.components._peft.lora import PeftConfig
import dataclasses
print([f.name for f in dataclasses.fields(PeftConfig)])
"
```

**Known fields** (as of nemo_automodel mid-2026):
| Field | Description | Default |
|-------|-------------|---------|
| `target_modules` | List of module name patterns to apply LoRA to | `[]` |
| `exclude_modules` | Module patterns to exclude | `[]` |
| `match_all_linear` | Auto-match all nn.Linear layers | `false` |
| `dim` | LoRA rank (NOT `r`) | — |
| `alpha` | LoRA scaling factor (NOT `lora_alpha`) | — |
| `dropout` | LoRA dropout (NOT `lora_dropout`) | `0.0` |
| `dropout_position` | Where to apply dropout | — |
| `lora_A_init` | Initialization for LoRA A matrix | — |
| `lora_dtype` | Data type for LoRA weights | — |
| `use_dora` | Enable DoRA (Weight-Decomposed LoRA) | `false` |
| `use_memory_efficient_lora` | Memory-efficient LoRA implementation | `false` |
| `use_triton` | Use Triton-accelerated LoRA kernels | `false` |
| `moe_rank_scaling` | Rank scaling for MoE experts | — |

### 2b. Find target module names (if not using match_all_linear)

```bash
python -c "
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    '<MODEL_NAME_OR_PATH>',
    trust_remote_code=True,
    device_map='cpu',
    torch_dtype='auto'
)
linear_names = set()
for name, module in model.named_modules():
    if 'Linear' in type(module).__name__:
        short = name.split('.')[-1]
        linear_names.add(short)
print('Unique linear module names:', sorted(linear_names))
"
```

> **Tip**: Use `match_all_linear: true` in the YAML to skip this step. It automatically finds and patches all linear layers.

---

## Step 3: Prepare Training Data

Training data should be in JSONL format with chat structure:

```jsonl
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is AI?"}, {"role": "assistant", "content": "AI stands for Artificial Intelligence..."}]}
{"messages": [{"role": "user", "content": "Explain GPUs."}, {"role": "assistant", "content": "GPUs are specialized processors..."}]}
```

Save to a path accessible from the training nodes, e.g.:
```bash
/lustre/fsw/general_sa/<username>/train.jsonl
```

---

## Step 4: Generate the YAML Config

### Template — LoRA Fine-Tuning

```yaml
recipe: nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction

model:
  _target_: nemo_automodel._transformers.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: <MODEL_NAME_OR_PATH>
  trust_remote_code: true

peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  dim: 16                    # LoRA rank (NOT "r")
  alpha: 32                  # LoRA alpha (NOT "lora_alpha") — rule of thumb: 2× dim
  dropout: 0.05              # LoRA dropout (NOT "lora_dropout")
  match_all_linear: true     # Auto-target all linear layers (safest option)
  # use_dora: true           # Optional: enable DoRA for better quality
  # use_triton: true         # Optional: Triton-accelerated LoRA kernels

dataset:
  _target_: nemo_automodel.components.datasets.llm.ChatDataset
  path_or_dataset_id: <PATH_TO_TRAIN_JSONL>
  split: train
  seq_length: 2048
  truncation: longest_first
  padding: max_length

dataloader:
  _target_: torch.utils.data.DataLoader
  batch_size: 1
  num_workers: 0
  pin_memory: true

distributed:
  strategy: fsdp2
  ep_size: <NUM_GPUS>        # Expert parallelism size (match GPU count for MoE models)
  moe: {}
  activation_checkpointing: true

loss_fn:
  _target_: nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy

step_scheduler:
  max_steps: 100             # Total training steps
  local_batch_size: 1
  global_batch_size: 4       # = local_batch_size × num_gpus × grad_accum_steps
  ckpt_every_steps: 100      # Save checkpoint every N steps

checkpoint:
  checkpoint_dir: <CHECKPOINT_DIR>

optimizer:
  _target_: torch.optim.AdamW
  lr: 2.0e-4
  weight_decay: 0.01

seed: 42
```

### Common Model Paths
| Model | pretrained_model_name_or_path |
|-------|------------------------------|
| Nemotron 3.5 Lightning 30B (MoE) | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` |
| Nemotron 4 340B | `nvidia/Nemotron-4-340B-Base` |
| Llama 3.1 8B | `meta-llama/Llama-3.1-8B-Instruct` |
| Llama 3.2 1B | `meta-llama/Llama-3.2-1B` |
| Qwen 2.5 3B | `Qwen/Qwen2.5-3B-Instruct` |

---

## Step 5: Run Training

```bash
torchrun --nproc_per_node=<NUM_GPUS> \
  -m nemo_automodel.cli.app \
  <PATH_TO_YAML>
```

Example with 4 GPUs:
```bash
torchrun --nproc_per_node=4 \
  -m nemo_automodel.cli.app \
  /lustre/fsw/general_sa/bbalakreshna/finetune_nemotron.yaml
```

### What to Watch During Training
| Metric | Healthy Range | Action if Unhealthy |
|--------|--------------|---------------------|
| `loss` | Steadily decreasing | Check data quality, reduce lr |
| `grad_norm` | 0.1 – 10.0 | Add gradient clipping if > 50 |
| `mem` | Below GPU VRAM limit | Reduce batch_size, enable QLoRA |
| `tps` (tokens/sec) | Stable, not dropping | Check data loading (num_workers) |
| Trainable params % | > 0% | If 0% → target_modules didn't match; use `match_all_linear: true` |

---

## Step 6: Upload to HuggingFace

```bash
# Login (one-time)
hf auth login

# Upload checkpoint
hf upload <HF_USERNAME>/<REPO_NAME> \
  <CHECKPOINT_DIR>/epoch_X_step_Y \
  --private --commit-message "LoRA fine-tuned checkpoint"
```

> **Note**: If `hf` CLI is not available, fall back to:
> ```bash
> huggingface-cli upload <HF_USERNAME>/<REPO_NAME> <CHECKPOINT_PATH> --private
> ```
> If `huggingface-cli` is also deprecated, use Python:
> ```python
> from huggingface_hub import HfApi
> api = HfApi()
> api.create_repo('<HF_USERNAME>/<REPO_NAME>', private=True, exist_ok=True)
> api.upload_folder(
>     folder_path='<CHECKPOINT_PATH>',
>     repo_id='<HF_USERNAME>/<REPO_NAME>',
>     commit_message='LoRA fine-tuned checkpoint'
> )
> ```

---

## Step 7: Cleanup

```bash
# Remove checkpoints
rm -rf <CHECKPOINT_DIR>/

# Remove cached HF model
rm -rf ~/.cache/huggingface/hub/models--<MODEL_ORG>--<MODEL_NAME>/

# Remove YAML config (optional)
rm <PATH_TO_YAML>

# Remove training data (optional)
rm <PATH_TO_TRAIN_JSONL>

# Check what's using space
du -sh <WORKING_DIR>/*
du -sh ~/.cache/huggingface/hub/*
```

---

## Optimization Reference

### Memory Optimization
| Technique | Savings | Config Change|
|-----------|---------|---------------|
| QLoRA (4-bit) | ~60-70% | Add bitsandbytes 4-bit quantization |
| Gradient Checkpointing | ~30-50% | `activation_checkpointing: true` |
| Mixed Precision (BF16) | ~50% | `torch_dtype: bfloat16` |
| CPU Offloading | ~40-60% | `cpu_offload: true` in FSDP |
| Flash Attention 2 | ~20-40% | `attn_implementation: flash_attention_2` |
| Reduce LoRA Rank | Proportional | `dim: 8` instead of 16 |

### Speed Optimization
| Technique | Speedup | Config Change |
|-----------|---------|---------------|
| Gradient Accumulation | ~1.5-3× | Increase `global_batch_size` |
| torch.compile | ~10-30% | Add compiler flag |
| Sequence Packing | ~1.5-3× | `packing: true` in dataset |
| Data Workers | Removes I/O bottleneck | `num_workers: 4` |
| Fused AdamW | ~5-15% | `fused: True` in optimizer |

### Quality Optimization
| Technique | Impact | Config Change |
|-----------|--------|---------------|
| LR Schedule | Better convergence | Add cosine warmup scheduler |
| DoRA | Often outperforms LoRA | `use_dora: true` |
| Increase Rank | More expressive | `dim: 32` or 64 |
| Eval Loop | Prevent overfitting | Add eval dataset + eval_every_steps |
| Data Curation | Biggest single lever | Clean, deduplicated JSONL |

### Checkpoint Storage Sizes (30B model reference)
| Method | Checkpoint Size |
|--------|----------------|
| Full Fine-Tune (bf16) | ~60 GB |
| Full Fine-Tune (fp32) | ~120 GB |
| LoRA dim=8 | ~213 MB |
| LoRA dim=16 | ~426 MB |
| LoRA dim=32 | ~852 MB |
| LoRA dim=64 | ~1.7 GB |
| Full + Optimizer States | ~180-360 GB |

---

## Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `has no attribute 'LoRA'` | Wrong `_target_` class name | Use `nemo_automodel.components._peft.lora.PeftConfig` |
| `trainable_params cannot be empty` | `target_modules` didn't match any layers | Use `match_all_linear: true` or inspect model layer names |
| `Checkpoint key mismatch` (WARNING) | Expected for some model architectures | Usually safe to ignore if training proceeds |
| `grouped_gemm is not available` | Optional MoE dependency | Install with `pip install git+https://github.com/fanshiqing/grouped_gemm@v1.1.4` or ignore |
| `pynvml deprecated` | Old pynvml package | `pip install nvidia-ml-py` (cosmetic warning, safe to ignore) |
| CUDA OOM | Model + activations exceed GPU VRAM | Reduce batch_size, enable gradient checkpointing, try QLoRA |

---

## Placeholder Reference

When generating configs, replace these placeholders:

| Placeholder | Example Value |
|-------------|---------------|
| `<MODEL_NAME_OR_PATH>` | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` |
| `<PATH_TO_TRAIN_JSONL>` | `/lustre/fsw/general_sa/bbalakreshna/train.jsonl` |
| `<CHECKPOINT_DIR>` | `/lustre/fsw/general_sa/bbalakreshna/checkpoints` |
| `<NUM_GPUS>` | `4` |
| `<PATH_TO_YAML>` | `/lustre/fsw/general_sa/bbalakreshna/finetune_nemotron.yaml` |
| `<HF_USERNAME>` | `Balab2021` |
| `<REPO_NAME>` | `nemotron-3.5-lightning-30b-lora` |
