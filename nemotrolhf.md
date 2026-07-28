# Fine tune nemotron models from huggingface

## Pre-requiste

- Huggingface login
- wandb login

## Code

- Setup the environment

```
pip install transformers peft trl datasets accelerate bitsandbytes
```

```
hf auth login --token "xxxx"
```

```
wandb login
```

- Code

```
cat << 'EOF' > finetune_nemotron.py
import os
import torch
import wandb
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

def main():
    model_name = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
    output_dir = "./checkpoints"

    # ---- W&B Configuration ----
    wandb_project = "nemotron-finetune"
    wandb_run_name = "lora-r64-nemotron-nano-8b"

    print("=" * 60)
    print("  Nemotron Fine-Tuning (HuggingFace + PEFT + TRL + W&B)")
    print("=" * 60)
    print(f"  Model:      {model_name}")
    print(f"  GPUs:       {torch.cuda.device_count()}")
    print(f"  W&B Project: {wandb_project}")
    print(f"  W&B Run:    {wandb_run_name}")
    print("=" * 60)

    # Initialize W&B
    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config={
            "model": model_name,
            "lora_rank": 64,
            "lora_alpha": 128,
            "lora_dropout": 0.05,
            "epochs": 10,
            "batch_size": 4,
            "grad_acc_steps": 4,
            "learning_rate": 2e-4,
            "max_length": 2048,
            "dataset": "rajpurkar/squad",
            "dataset_size": 2000,
        },
    )

    # Load tokenizer
    print("\n>>> Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(">>> Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # LoRA config
    print(">>> Applying LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Log model info to W&B
    wandb.log({
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_params": sum(p.numel() for p in model.parameters()),
    })

    # Load dataset
    print(">>> Loading dataset...")
    dataset = load_dataset("rajpurkar/squad", split="train[:2000]")

    def format_squad(example):
        return {
            "text": f"### Question: {example['question']}\n### Context: {example['context']}\n### Answer: {example['answers']['text'][0]}"
        }

    dataset = dataset.map(format_squad, remove_columns=dataset.column_names)

    # Training config with W&B enabled
    print(">>> Configuring training...")
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        max_length=2048,
        dataset_text_field="text",
        # ---- W&B logging ----
        report_to="wandb",
        run_name=wandb_run_name,
    )

    # Trainer
    print(">>> Starting training...")
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    # Save
    print(f"\n>>> Saving to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Log final metrics to W&B
    wandb.log({"status": "complete"})
    wandb.finish()

    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  Checkpoint: {output_dir}")
    print(f"  W&B Dashboard: https://wandb.ai/balabala76/{wandb_project}")
    print("=" * 60)

if __name__ == "__main__":
    main()
EOF

echo ">>> finetune_nemotron.py updated with W&B logging!"
```

- run the job

```
python finetune_nemotron.py
```