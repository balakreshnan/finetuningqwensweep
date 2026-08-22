# Hecate - Nemo Back Fine Tuning LoRA - Nemotron Lighting 30B model

## Pre-requiste

- Need a Rubin 4 GPU Machine
- using nemo framework - nvcr.io#nvidia/nemo:26.08
- To validate multi gpu fine tuning
- Access to huggingface
- Access to wandb for metrics
- to validate multi model training
- model used nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16

## Steps

- Log into terminal
- Check the available machines to use

```
sinfo --summarize
sacctmgr show associations user=$USER format=Account,Partition
squeue -u $USER
```

- create enroot credentials

```
mkdir -p ~/.config/enroot
touch ~/.config/enroot/.credentials
```

- now create the credentials file

```
cat > ~/.config/enroot/.credentials << 'EOF'
machine gxxxxxx login xxx@xxxx.com password xxxxxxxxx
machine nvcr.io login $oauthtoken password xxxxxxxx
EOF
```

- Now lets get a interactive cluster to quick check
- Depends on the availability
- to get latest image use

- SLURM interactive session

```
srun --account=general_sa \
     --partition=36x2-a01r \
     --nodes=1 \
     --ntasks-per-node=1 \
     --time=5:00:00 \
     --job-name=general_sa-finetune:interactive \
     --container-image=nvcr.io#nvidia/nemo:26.08 \
     --container-mount-home \
     --no-container-remap-root \
     --mpi=pmix \
     --pty bash
```

- set the storage path

```
# Set your account from the output above
export ACCOUNT="xxxxx"

# Create and enter your Lustre directory
export LUSTRE_DIR="/lustre/fsw/$ACCOUNT/$USER"
mkdir -p "$LUSTRE_DIR"
cd "$LUSTRE_DIR"

# Verify access
pwd
touch storage_test
ls -l storage_test
rm storage_test
df -hT "$LUSTRE_DIR"
```

- setup space fist
- to avoid disk space issues

```
ls -la /lustre/fsw/xxxx/bbalakreshna/
du -sh /lustre/fsw/xxxx/bbalakreshna/
```

- install auto model

```
pip install nemo-automodel
```

- for model weights

```
rm -rf ~/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16

# 2. Point future downloads to lustre
export HF_HOME=/lustre/fsw/general_sa/bbalakreshna/.cache/huggingface
export TRANSFORMERS_CACHE=/lustre/fsw/general_sa/bbalakreshna/.cache/huggingface/hub
mkdir -p $HF_HOME/hub
```

- check space

```
du -sh /lustre/fsw/general_sa/bbalakreshna/.cache/huggingface/hub/
df -h /lustre/fsw/general_sa/bbalakreshna/
du -sh /lustre/fsw/general_sa/bbalakreshna/.cache/huggingface/hub/models--*
du -sh ~/.cache/huggingface/hub/ 2>/dev/null
```

- install to use parallel ep_size

```
pip install --no-build-isolation git+https://github.com/fanshiqing/grouped_gemm@v1.1.4
```

- create the config to run

```
cat << 'EOF' > /lustre/fsw/general_sa/bbalakreshna/finetune_nemotron.yaml
recipe: nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction

model:
  _target_: nemo_automodel._transformers.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
  trust_remote_code: true

peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  dim: 16
  alpha: 32
  dropout: 0.05
  match_all_linear: true

dataset:
  _target_: nemo_automodel.components.datasets.llm.ChatDataset
  path_or_dataset_id: /lustre/fsw/general_sa/bbalakreshna/train.jsonl
  tokenizer:
    _target_: transformers.AutoTokenizer.from_pretrained
    pretrained_model_name_or_path: nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
    trust_remote_code: true
  split: train
  seq_length: 2048
  padding: max_length
  truncation: longest_first

dataloader:
  _target_: torch.utils.data.DataLoader
  batch_size: 1
  num_workers: 0
  pin_memory: true

distributed:
  strategy: fsdp2
  ep_size: 4
  moe: {}
  activation_checkpointing: true

loss_fn:
  _target_: nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy

step_scheduler:
  max_steps: 100
  local_batch_size: 1
  global_batch_size: 4
  ckpt_every_steps: 101

checkpoint:
  checkpoint_dir: /lustre/fsw/general_sa/bbalakreshna/checkpoints

optimizer:
  _target_: torch.optim.AdamW
  lr: 2.0e-4
  weight_decay: 0.01

seed: 42
EOF
```

- patch dataset labels

```
cat << 'PYEOF' > /lustre/fsw/general_sa/bbalakreshna/run_finetune.py
"""Wrapper that patches tilelang arch detection + dataset, then runs nemo_automodel."""
import torch

# ==== Patch 0: Force tilelang to use sm_90a instead of sm_107a ====
import tilelang.contrib.nvcc as _tl_nvcc

_orig_get_target_arch = _tl_nvcc.get_target_arch
def _patched_get_target_arch(compute_version):
    result = _orig_get_target_arch(compute_version)
    # If arch is beyond sm_90a (e.g. sm_107a), cap to sm_90a
    numeric = ''.join(c for c in result if c.isdigit())
    if numeric and int(numeric) > 90:
        print(f"[PATCH] Capping tilelang arch from sm_{result} to sm_90a")
        return "90a"
    return result
_tl_nvcc.get_target_arch = _patched_get_target_arch

# Also patch get_target_compute_version if it reads from GPU
_orig_get_cv = _tl_nvcc.get_target_compute_version
def _patched_get_cv(target=None):
    result = _orig_get_cv(target)
    if isinstance(result, tuple) and result[0] >= 10:
        print(f"[PATCH] Capping compute version from {result} to (9, 0)")
        return (9, 0)
    if isinstance(result, str):
        parts = result.split(".")
        if len(parts) == 2 and int(parts[0]) >= 10:
            print(f"[PATCH] Capping compute version from {result} to 9.0")
            return "9.0"
    return result
_tl_nvcc.get_target_compute_version = _patched_get_cv
print("[PATCH] tilelang arch detection patched (sm_107a -> sm_90a)")

# ==== Patch 1: ChatDataset returns tensors ====
from nemo_automodel.components.datasets.llm import ChatDataset
_orig_getitem = ChatDataset.__getitem__
def _patched_getitem(self, idx):
    sample = _orig_getitem(self, idx)
    for key in sample:
        if isinstance(sample[key], list):
            try:
                sample[key] = torch.tensor(sample[key])
            except (ValueError, TypeError):
                pass
    return sample
ChatDataset.__getitem__ = _patched_getitem
print("[PATCH] ChatDataset.__getitem__ patched to return tensors")

from nemo_automodel.cli.app import main
main()
PYEOF
```

- now create a sample data set
- Code is available in 

- Code is available in [pytchenemobacklightft.md](pytchenemobacklightft.md)

- run the training
- clear previous runs

```
pkill -9 -u bbalakreshna python
rm -rf /lustre/fsw/general_sa/bbalakreshna/checkpoints/
mkdir /lustre/fsw/general_sa/bbalakreshna/checkpoints/

TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1 \
torchrun --nproc_per_node=4 \
  /lustre/fsw/general_sa/bbalakreshna/run_finetune.py \
  /lustre/fsw/general_sa/bbalakreshna/finetune_nemotron.yaml
```

- once the run is successful check the file size

```
du -sh /lustre/fsw/general_sa/bbalakreshna/checkpoints/
```

- error

```
Training:   0%|                                                                             | 0/100 [00:00<?, ?step/s]'sm_107a' is not a recognized processor for this target (ignoring processor)
'sm_107a' is not a recognized processor for this target (ignoring processor)
'sm_107a' is not a recognized processor for this target (ignoring processor)
'sm_107a' is not a recognized processor for this target (ignoring processor)
'sm_107a' is not a recognized processor for this target (ignoring processor)
'sm_107a' is not a recognized processor for this target (ignoring processor)
'sm_107a' is not a recognized processor for this target (ignoring processor)
'sm_107a' is not a recognized processor for this target (ignoring processor)
'sm_107a' is not a recognized processor for this target (ignoring processor)
'sm_107a' is not a recognized processor for this target (ignoring processor)
'sm_107a' is not a recognized processor for this target (ignoring processor)
'sm_107a' is not a recognized processor for this target (ignoring processor)
LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.shfl.sync.up.i32
'sm_107a' is not a recognized processor for this target (ignoring processor)
'sm_107a' is not a recognized processor for this target (ignoring processor)
'sm_107a' is not a recognized processor for this target (ignoring processor)
'sm_107a' is not a recognized processor for this target (ignoring processor)
'sm_107a' is not a recognized processor for this target (ignoring processor)
'sm_107a' is not a recognized processor for this target (ignoring processor)
'sm_107a' is not a recognized processor for this target (ignoring processor)
'sm_107a' is not a recognized processor for this target (ignoring processor)
'sm_107a' is not a recognized processor for this target (ignoring processor)
'sm_107a' is not a recognized processor for this target (ignoring processor)
'sm_107a' is not a recognized processor for this target (ignoring processor)
'sm_107a' is not a recognized processor for this target (ignoring processor)
'sm_107a' is not a recognized processor for this target (ignoring processor)
'sm_107a' is not a recognized processor for this target (ignoring processor)
'sm_107a' is not a recognized processor for this target (ignoring processor)
'sm_107a' is not a recognized processor for this target (ignoring processor)
'sm_107a' is not a recognized processor for this target (ignoring processor)
'sm_107a' is not a recognized processor for this target (ignoring processor)
LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.shfl.sync.up.i32
'sm_107a' is not a recognized processor for this target (ignoring processor)
'sm_107a' is not a recognized processor for this target (ignoring processor)
'sm_107a' is not a recognized processor for this target (ignoring processor)
'sm_107a' is not a recognized processor for this target (ignoring processor)
'sm_107a' is not a recognized processor for this target (ignoring processor)
'sm_107a' is not a recognized processor for this target (ignoring processor)
LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.shfl.sync.up.i32
'sm_107a' is not a recognized processor for this target (ignoring processor)
'sm_107a' is not a recognized processor for this target (ignoring processor)
'sm_107a' is not a recognized processor for this target (ignoring processor)
'sm_107a' is not a recognized processor for this target (ignoring processor)
'sm_107a' is not a recognized processor for this target (ignoring processor)
'sm_107a' is not a recognized processor for this target (ignoring processor)
'sm_107a' is not a recognized processor for this target (ignoring processor)
'sm_107a' is not a recognized processor for this target (ignoring processor)
'sm_107a' is not a recognized processor for this target (ignoring processor)
'sm_107a' is not a recognized processor for this target (ignoring processor)
'sm_107a' is not a recognized processor for this target (ignoring processor)
'sm_107a' is not a recognized processor for this target (ignoring processor)
LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.shfl.sync.up.i32
```