# Nemo model fine tuning

## Setup

```
pip install peft trl datasets accelerate
```

## Code

```
cat > /data/finetune_lora.py << 'EOF'
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model

# --- Config ---
model_name = "nvidia/Nemotron-Mini-4B-Instruct"
hf_repo = "Balab2021/Nemotron-Mini-4B-LoRA-OpenAssistant"
wandb_project = "nemotron-4b-lora-finetune"

# --- Wandb ---
os.environ["WANDB_PROJECT"] = wandb_project
import wandb
wandb.login()  # uses token from ~/.netrc

# --- Model + Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)

lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --- Dataset ---
dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")
val_dataset = load_dataset("timdettmers/openassistant-guanaco", split="test")

dataset = dataset.select(range(2000))
val_dataset = val_dataset.select(range(200))

MAX_LEN = 1024

def tokenize(example):
    tokens = tokenizer(example["text"], truncation=True, max_length=MAX_LEN, padding=False)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

train_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)
val_dataset = val_dataset.map(tokenize, remove_columns=val_dataset.column_names)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# --- Training ---
training_args = TrainingArguments(
    output_dir="/results/nemotron-lora",
    num_train_epochs=20,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    save_steps=100,
    eval_strategy="steps",
    eval_steps=100,
    save_total_limit=2,
    warmup_steps=20,
    lr_scheduler_type="cosine",
    report_to="wandb",
    run_name="nemotron-4b-lora-oasst-2k",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
)

trainer.train()

# --- Save locally ---
trainer.save_model("/results/nemotron-lora/final")
tokenizer.save_pretrained("/results/nemotron-lora/final")

# --- Push to Hugging Face ---
from huggingface_hub import HfApi

print(f"Pushing model to https://huggingface.co/{hf_repo} ...")
api = HfApi()
api.create_repo(repo_id=hf_repo, exist_ok=True, repo_type="model")
api.upload_folder(
    folder_path="/results/nemotron-lora/final",
    repo_id=hf_repo,
    repo_type="model",
    commit_message="LoRA fine-tune on OpenAssistant-Guanaco (2k samples, 1 epoch)",
)
print(f"Done! Model live at: https://huggingface.co/{hf_repo}")

wandb.finish()
EOF
```
