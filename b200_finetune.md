
- configure security

```
mkdir -p ~/.config/enroot
cat > ~/.config/enroot/.credentials << 'EOF'
machine gitlab-master.nvidia.com login <YOUR_USERNAME> password <GITLAB_TOKEN>
machine nvcr.io login $oauthtoken password <YOUR_NGC_TOKEN>
EOF
chmod 0600 ~/.config/enroot/.credentials
```

- get slurm account

```
sacctmgr -nP show assoc where user=$(whoami) format=account
```

- pull image

```
srun -A general_sa \
     -N 1 \
     -p batch \
     -J general_sa-finetune:interactive \
     --container-image=gitlab-master.nvidia.com/dl/dgx/pytorch:main-py3-devel \
     --mpi=pmix \
     --container-mount-home \
     --no-container-remap-root \
     --pty bash
```

- install huggingface

```
pip install --user huggingface-hub
huggingface-cli login --token <YOUR_HF_TOKEN>
```

- check gpu

```
nvidia-smi
```

- install automodel

```
pip install --user git+https://github.com/NVIDIA/NeMo.git
```

```
python3 -c "import nemo; print(nemo.__version__)"
```

```
python3 -c "
from nemo.collections.llm import finetune
# Follow NeMo's SFT/LoRA examples
"
```

- install other libraries


```
pip install --user wandb huggingface-hub peft accelerate transformers datasets
export PATH=$HOME/.local/bin:$PATH
```

- login into wand

```
wandb login
```

```
mkdir -p /lustre/fsw/general_sa/$USER/finetune
cd /lustre/fsw/general_sa/$USER/finetune
```

- now write the fine tune code


```
cat > finetune_lora.py << 'EOF'
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
MODEL_NAME = "meta-llama/Llama-3.2-1B"
HF_REPO_ID = "Balab2021/Llama-3.2-1B-lora-b200"
OUTPUT_DIR = "/lustre/fsw/general_sa/checkpoints/llama-lora"
WANDB_PROJECT = "llama-3.2-1b-lora-finetune"

# ──────────────────────────────────────────────
# 1. Initialize Weights & Biases
# ──────────────────────────────────────────────
import wandb

wandb.login()
wandb.init(
    project=WANDB_PROJECT,
    name="llama-3.2-1b-lora-squad",
    config={
        "model": MODEL_NAME,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "learning_rate": 1e-4,
        "epochs": 50,
        "batch_size": 4,
        "max_length": 512,
    },
    tags=["lora", "llama-3.2", "squad", "pre-tyche", "gb200"],
)

# ──────────────────────────────────────────────
# 2. Load tokenizer & model
# ──────────────────────────────────────────────
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# ──────────────────────────────────────────────
# 3. Apply LoRA
# ──────────────────────────────────────────────
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ──────────────────────────────────────────────
# 4. Load & preprocess dataset
# ──────────────────────────────────────────────
print("Loading SQuAD dataset...")
dataset = load_dataset("rajpurkar/squad", split="train[:1000]")

def preprocess(example):
    text = (
        f"Context: {example['context']}\n"
        f"Question: {example['question']}\n"
        f"Answer: {example['answers']['text'][0]}"
    )
    tokens = tokenizer(text, truncation=True, max_length=512, padding="max_length")
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

# ──────────────────────────────────────────────
# 5. Training arguments (with W&B reporting)
# ──────────────────────────────────────────────
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=50,
    per_device_train_batch_size=4,
    learning_rate=1e-4,
    logging_steps=10,
    save_steps=100,
    save_total_limit=3,
    bf16=True,
    gradient_accumulation_steps=4,
    warmup_ratio=0.05,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    report_to="wandb",
    run_name="llama-3.2-1b-lora-squad",
    logging_first_step=True,
)

# ──────────────────────────────────────────────
# 6. Train
# ──────────────────────────────────────────────
print("Starting fine-tuning...")
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
)
trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved locally to {OUTPUT_DIR}")

# ──────────────────────────────────────────────
# 7. Log final metrics to W&B
# ──────────────────────────────────────────────
final_metrics = trainer.state.log_history[-1] if trainer.state.log_history else {}
wandb.log({"final_train_loss": final_metrics.get("train_loss", None)})
wandb.finish()
print("W&B run finished.")

# ──────────────────────────────────────────────
# 8. Upload to HuggingFace Hub
# ──────────────────────────────────────────────
from huggingface_hub import HfApi

print(f"Uploading model to HuggingFace: {HF_REPO_ID} ...")

api = HfApi()

api.create_repo(
    repo_id=HF_REPO_ID,
    repo_type="model",
    private=True,
    exist_ok=True,
)

api.upload_folder(
    repo_id=HF_REPO_ID,
    folder_path=OUTPUT_DIR,
    commit_message="Upload Llama-3.2-1B LoRA fine-tuned on SQuAD (Pre-Tyche GB200)",
)

print(f"Upload complete! View your model at: https://huggingface.co/{HF_REPO_ID}")
print("All done!")
EOF
```

- run the code 

```
python3 finetune_lora.py
```

```
torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) finetune_lora.py
```

```
accelerate launch --multi_gpu --num_processes=$(nvidia-smi -L | wc -l) finetune_lora.py
```

- wait for execution to complete

- now vidion model

```
pip install --user qwen-vl-utils pillow
```

- code for qwnd vision finetuning

```
cat > /lustre/fsw/general_sa/bbalakreshna/finetune/finetune_qwen_vl.py << 'EOF'
import os

# ──────────────────────────────────────────────
# FIX: NCCL & env config BEFORE any torch import
# ──────────────────────────────────────────────
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
import torch

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
HF_REPO_ID = "Balab2021/Qwen2.5-VL-7B-lora-b200"
OUTPUT_DIR = "/lustre/fsw/general_sa/checkpoints/qwen-vl-lora"
WANDB_PROJECT = "qwen2.5-vl-7b-lora-finetune"
MAX_LENGTH = 2048
# FIX: Fixed image size so all samples produce identical pixel_values shapes
FIXED_IMAGE_SIZE = (448, 448)
FIXED_PIXELS = FIXED_IMAGE_SIZE[0] * FIXED_IMAGE_SIZE[1]

# ──────────────────────────────────────────────
# 1. Initialize Weights & Biases (rank 0 only)
# ──────────────────────────────────────────────
import wandb

local_rank = int(os.environ.get("LOCAL_RANK", 0))
if local_rank == 0:
    wandb.login()
    wandb.init(
        project=WANDB_PROJECT,
        name="qwen2.5-vl-7b-lora-4gpu",
        config={
            "model": MODEL_NAME,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "learning_rate": 2e-5,
            "epochs": 50,
            "batch_size": 2,
            "num_gpus": 4,
            "max_length": MAX_LENGTH,
        },
        tags=["lora", "qwen2.5-vl", "multimodal", "pre-tyche", "gb200", "4gpu"],
    )

# ──────────────────────────────────────────────
# 2. Load processor & model
# ──────────────────────────────────────────────
print(f"[Rank {local_rank}] Loading processor and model...")

# FIX: Set min_pixels == max_pixels to force identical vision token counts
processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    min_pixels=FIXED_PIXELS,
    max_pixels=FIXED_PIXELS,
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
)

# ──────────────────────────────────────────────
# 3. Apply LoRA
# ──────────────────────────────────────────────
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
if local_rank == 0:
    model.print_trainable_parameters()

# ──────────────────────────────────────────────
# 4. Custom Dataset
# ──────────────────────────────────────────────
print(f"[Rank {local_rank}] Loading multimodal dataset...")
raw_dataset = load_dataset("HuggingFaceM4/the_cauldron", "ai2d", split="train[:500]")

# FIX: Filter to only samples WITH images — mixed batches cause rank desync
raw_dataset = raw_dataset.filter(
    lambda x: x.get("images") is not None and len(x["images"]) > 0
)

# FIX: Trim to multiple of world_size * batch_size to avoid uneven last batches
world_size = int(os.environ.get("WORLD_SIZE", 4))
per_device_bs = 2
trim_to = len(raw_dataset) - (len(raw_dataset) % (world_size * per_device_bs))
raw_dataset = raw_dataset.select(range(trim_to))

print(f"[Rank {local_rank}] Loaded {len(raw_dataset)} examples (trimmed for even distribution).")


class QwenVLDataset(Dataset):
    def __init__(self, dataset, processor, max_length):
        self.dataset = dataset
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = example["images"][0]
        user_text = example["texts"][0]["user"]
        assistant_text = example["texts"][0]["assistant"]

        # FIX: Resize to FIXED square size so all images produce
        # identical pixel_values tensor shapes across all ranks
        image = image.convert("RGB").resize(FIXED_IMAGE_SIZE)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_text},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            },
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

        result = {}
        for k, v in inputs.items():
            v = v.squeeze(0)
            if v.dim() >= 1 and v.shape[0] > self.max_length:
                v = v[: self.max_length]
            result[k] = v
        result["labels"] = result["input_ids"].clone()
        return result


train_dataset = QwenVLDataset(raw_dataset, processor, MAX_LENGTH)
print(f"[Rank {local_rank}] Dataset ready.")

# ──────────────────────────────────────────────
# 5. Data collator — FIX: pad variable tensors instead of dropping
# ──────────────────────────────────────────────
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class MultimodalDataCollator:
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        batch = {}
        # Collect only keys present in ALL samples
        common_keys = set(features[0].keys())
        for f in features[1:]:
            common_keys &= set(f.keys())

        for key in common_keys:
            tensors = [f[key] for f in features]
            try:
                batch[key] = torch.stack(tensors)
            except RuntimeError:
                # Variable-length dim 0 (e.g. pixel_values) — pad to max
                max_len = max(t.shape[0] for t in tensors)
                padded = []
                for t in tensors:
                    pad_size = max_len - t.shape[0]
                    if pad_size > 0:
                        pad_shape = (pad_size,) + tuple(t.shape[1:])
                        t = torch.cat([t, torch.zeros(pad_shape, dtype=t.dtype)], dim=0)
                    padded.append(t)
                batch[key] = torch.stack(padded)
        return batch


data_collator = MultimodalDataCollator()

# ──────────────────────────────────────────────
# 6. Training arguments — ALL FIXES APPLIED
# ──────────────────────────────────────────────
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=50,
    per_device_train_batch_size=per_device_bs,
    learning_rate=2e-5,
    logging_steps=10,
    save_steps=50,
    save_total_limit=3,
    bf16=True,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    warmup_steps=50,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    dataloader_pin_memory=False,
    dataloader_num_workers=0,
    ddp_find_unused_parameters=False,
    # FIX 1: SAME report_to on all ranks (Trainer gates wandb to rank 0)
    report_to="wandb",
    run_name="qwen2.5-vl-7b-lora-4gpu",
    logging_first_step=True,
    remove_unused_columns=False,
    # FIX 2: Drop incomplete last batch
    dataloader_drop_last=True,
    # FIX 3: Explicitly disable eval
    eval_strategy="no",
    # FIX 4: Increase DDP timeout to 2 hours (default 30 min was too short)
    ddp_timeout=7200,
    # FIX 5: Set seed for reproducibility across ranks
    seed=42,
    data_seed=42,
)

# ──────────────────────────────────────────────
# 7. Sync all ranks before training
# ──────────────────────────────────────────────
print(f"[Rank {local_rank}] Starting fine-tuning...")

# FIX: Barrier to ensure all ranks are ready before training starts
if torch.distributed.is_initialized():
    torch.distributed.barrier()

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)
trainer.train()

if local_rank == 0:
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"Model saved locally to {OUTPUT_DIR}")

    final_metrics = trainer.state.log_history[-1] if trainer.state.log_history else {}
    wandb.log({"final_train_loss": final_metrics.get("train_loss", None)})
    wandb.finish()
    print("W&B run finished.")

    from huggingface_hub import HfApi

    print(f"Uploading model to HuggingFace: {HF_REPO_ID} ...")
    api = HfApi()
    api.create_repo(
        repo_id=HF_REPO_ID,
        repo_type="model",
        private=True,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=HF_REPO_ID,
        folder_path=OUTPUT_DIR,
        commit_message="Upload Qwen2.5-VL-7B LoRA fine-tuned multimodal (Pre-Tyche GB200)",
    )
    print(f"Upload complete! View your model at: https://huggingface.co/{HF_REPO_ID}")
    print("All done!")
EOF
```

- run the code

```
accelerate launch --num_processes=4 --multi_gpu finetune_qwen_vl.py
```

```
accelerate launch --num_processes=4 --multi_gpu /lustre/fsw/general_sa/bbalakreshna/finetune/finetune_qwen_vl.py
```

```
accelerate launch --num_processes=$(nvidia-smi -L | wc -l) --multi_gpu finetune_qwen_vl.py
```

- done