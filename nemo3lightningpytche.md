# Pytche - Nemotron 3 Lightning

## Steps

- Login into pytche
- Create enroot credentials
- First create directory

```
sinfo --summarize
sacctmgr show associations user=$USER format=Account,Partition
squeue -u $USER
```

```
mkdir -p ~/.config/enroot
touch ~/.config/enroot/.credentials
```

```
cat > ~/.config/enroot/.credentials << 'EOF'
machine gxxxxxx login xxx@xxxx.com password xxxxxxxxx
machine nvcr.io login $oauthtoken password xxxxxxxx
EOF
```

- now run the interactive container

```
srun --account=general_sa \
     --partition=36x2-a01r \
     --nodes=1 \
     --ntasks-per-node=1 \
     --time=5:00:00 \
     --job-name=general_sa-finetune:interactive \
     --container-image=nvcr.io#nvidia/nemo:26.02 \
     --container-mount-home \
     --no-container-remap-root \
     --mpi=pmix \
     --pty bash
```

- install wandb

```
pip install wandb huggingface_hub
```

- download dataset

```
huggingface-cli download nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 --local-dir /workspace/nemotron-mini-4b
```

```
pip install datasets
```

- split datasize

```
cat << 'EOF' > /workspace/prepare_data.py
from datasets import load_dataset
import json

ds = load_dataset('databricks/databricks-dolly-15k', split='train')
ds = ds.shuffle(seed=42)

train_data = ds.select(range(500))
val_data = ds.select(range(500, 550))

def format_example(ex):
    context = f'Context: {ex["context"]}\n\n' if ex['context'] else ''
    return {
        'input': f'{context}Instruction: {ex["instruction"]}',
        'output': ex['response']
    }

for split, data, path in [('train', train_data, '/workspace/train.jsonl'),
                           ('val', val_data, '/workspace/val.jsonl')]:
    with open(path, 'w') as f:
        for ex in
            formatted = format_example(ex)
            f.write(json.dumps(formatted) + '\n')
    print(f'Wrote {len(data)} examples to {path}')
EOF
```

```
sed -i 's/for ex in$/for ex in data:/' /workspace/prepare_data.py
```

- create the data set

```
python /workspace/prepare_data.py
```

- convert HF to megatron

```
python /opt/Megatron-Bridge/examples/conversion/convert_checkpoints.py import \
  --hf-model /workspace/nemotron-mini-4b \
  --megatron-path /workspace/megatron_ckpt/nemotron3_nano \
  --torch-dtype bfloat16 \
  --trust-remote-code
```

- fine tuning config

```
cat << 'EOF' > /workspace/finetune_config.yaml
train:
  train_iters: 200
  micro_batch_size: 1
  global_batch_size: 8

optimizer:
  lr: 1e-4

scheduler:
  lr_warmup_iters: 10

logger:
  wandb_project: nemotron_finetune
  wandb_exp_name: nemotron3_nano_lora_dolly500
  log_interval: 5
  tensorboard_dir: /workspace/results/tb_logs

checkpoint:
  save: /workspace/results/checkpoints
  save_interval: 100
  pretrained_checkpoint: /workspace/megatron_ckpt/nemotron3_nano
  finetune: true
EOF
```

- run the training

```
cat << 'PYEOF' > /workspace/finetune_wrapper.py
#!/usr/bin/env python3
"""Wrapper: finetune Nemotron-3-Nano with input/output JSONL data."""

from typing import Any, Optional
from megatron.bridge.data.builders.hf_dataset import ProcessExampleOutput
from megatron.bridge.training.tokenizers.tokenizer import MegatronTokenizer

def process_input_output_example(
    example: dict[str, Any], tokenizer: Optional[MegatronTokenizer] = None
) -> ProcessExampleOutput:
    return ProcessExampleOutput(input=example["input"], output=example["output"])

# Monkey-patch the squad processor
import megatron.bridge.data.hf_processors.squad as squad_mod
squad_mod.process_squad_example = process_input_output_example

# Run the original script
exec(open("/opt/Megatron-Bridge/examples/models/nemotron_3/finetune_nemotron_3_nano.py").read())
PYEOF
```

- fix columns name

```
python3 - <<'PY'
p = "/opt/Megatron-Bridge/src/megatron/bridge/data/builders/hf_dataset.py"
lines = open(p).readlines()
out, i = [], 0
while i < len(lines):
    line = lines[i]
    if line.strip() == 'if split_name == "test":':
        indent = line[:len(line) - len(line.lstrip())]
        out.append(f'{indent}if split_name == "test" and "original_answers" in processed_example:\n')
        out.append(f'{indent}    json_line["original_answers"] = processed_example["original_answers"]\n')
        j = i + 1
        while j < len(lines) and (
            lines[j].strip().startswith('if "original_answers"') or
            lines[j].strip().startswith('json_line["original_answers"]')
        ):
            j += 1
        i = j
        continue
    out.append(line)
    i += 1
open(p, "w").write("".join(out))
print("rewritten")
PY
```

- run the torch

```
TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1 \
torchrun --nproc_per_node=4 \
  /workspace/finetune_wrapper.py \
  --peft lora \
  --seq-length 2048 \
  --config-file /workspace/finetune_config.yaml \
  dataset.dataset_name=json \
  dataset.split_val_from_train=false \
  dataset.packed_sequence_specs=null \
  'dataset.hf_kwargs={data_files: {train: /workspace/train.jsonl, validation: /workspace/val.jsonl}}' \
  model.expert_model_parallel_size=4 \
  model.moe_token_dispatcher_type=alltoall \
  model.bias_activation_fusion=false
```

- now combine the model

```
python /opt/Megatron-Bridge/examples/peft/merge_lora.py \
  --lora-checkpoint /workspace/results/checkpoints \
  --hf-model-path /workspace/nemotron-mini-4b \
  --output /workspace/merged_model
```

- Now upload the huggingface

```
torchrun --nproc_per_node=1 \
  /opt/Megatron-Bridge/examples/conversion/convert_checkpoints.py export \
  --hf-model /workspace/nemotron-mini-4b \
  --megatron-path /workspace/merged_model \
  --hf-path /workspace/nemotron-finetuned-hf \
  --trust-remote-code
```