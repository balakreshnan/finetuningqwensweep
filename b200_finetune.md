
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

- wait for execution to complete

- now vidion model

```
pip install --user qwen-vl-utils pillow
```

- code for qwnd vision finetuning

```
cat > finetune_qwen_vl.py << 'EOF'
import os
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
MAX_LENGTH = 2048  # Increased to accommodate vision tokens

# ──────────────────────────────────────────────
# 1. Initialize Weights & Biases
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

# Limit vision tokens by constraining image size
processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    min_pixels=256 * 28 * 28,   # minimum image tokens
    max_pixels=512 * 28 * 28,   # cap at ~512 vision tokens
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
# 4. Custom Dataset (lazy, no truncation issues)
# ──────────────────────────────────────────────
print(f"[Rank {local_rank}] Loading multimodal dataset...")
raw_dataset = load_dataset("HuggingFaceM4/the_cauldron", "ai2d", split="train[:500]")
print(f"[Rank {local_rank}] Loaded {len(raw_dataset)} examples.")

class QwenVLDataset(Dataset):
    def __init__(self, dataset, processor, max_length):
        self.dataset = dataset
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = example["images"][0] if example.get("images") else None
        user_text = example["texts"][0]["user"]
        assistant_text = example["texts"][0]["assistant"]

        # Build chat messages in Qwen format
        if image is not None:
            image = image.convert("RGB")
            # Resize image to limit vision tokens (max 384px on longest side)
            image.thumbnail((384, 384))
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
        else:
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_text}],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": assistant_text}],
                },
            ]

        # Use apply_chat_template to get proper text with image placeholders
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Process WITHOUT truncation — let vision tokens expand fully
        if image is not None:
            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                # NO truncation — this was causing the crash
            )
        else:
            inputs = self.processor(
                text=[text],
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
            )

        # Squeeze batch dim and clip to max_length (safe post-processing)
        result = {}
        for k, v in inputs.items():
            v = v.squeeze(0)
            if v.dim() >= 1 and v.shape[0] > self.max_length:
                v = v[:self.max_length]
            result[k] = v
        result["labels"] = result["input_ids"].clone()
        return result

train_dataset = QwenVLDataset(raw_dataset, processor, MAX_LENGTH)
print(f"[Rank {local_rank}] Dataset ready.")

# ──────────────────────────────────────────────
# 5. Data collator (handles variable-length tensors)
# ──────────────────────────────────────────────
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class MultimodalDataCollator:
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        batch = {}
        for key in features[0].keys():
            try:
                batch[key] = torch.stack([f[key] for f in features])
            except Exception:
                # Skip tensors that can't be stacked (different shapes)
                continue
        return batch

data_collator = MultimodalDataCollator()

# ──────────────────────────────────────────────
# 6. Training arguments (4 GPU DDP)
# ──────────────────────────────────────────────
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=50,
    per_device_train_batch_size=2,
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
    report_to="wandb" if local_rank == 0 else "none",
    run_name="qwen2.5-vl-7b-lora-4gpu",
    logging_first_step=True,
    remove_unused_columns=False,
)

# ──────────────────────────────────────────────
# 7. Train
# ──────────────────────────────────────────────
print(f"[Rank {local_rank}] Starting fine-tuning...")
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

    # ──────────────────────────────────────────
    # 8. Log final metrics to W&B
    # ──────────────────────────────────────────
    final_metrics = trainer.state.log_history[-1] if trainer.state.log_history else {}
    wandb.log({"final_train_loss": final_metrics.get("train_loss", None)})
    wandb.finish()
    print("W&B run finished.")

    # ──────────────────────────────────────────
    # 9. Upload to HuggingFace Hub
    # ──────────────────────────────────────────
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