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