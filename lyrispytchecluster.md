# Running interactive jobs with pytche or lyris and other NV72,B200, B300 Cluster

## Steps

- Log into the cluster using ssh from command line

```
sinfo --summarize
sacctmgr show associations user=$USER format=Account,Partition
squeue -u $USER
```

- summarize will list the GPU's type available
- please select one from there

```
srun -A general_sa \
     -N 1 \
     -p gb200 \
     -J general_sa-finetune:interactive \
     --container-image=gitlab-master.nvidia.com/dl/dgx/pytorch:main-py3-devel \
     --mpi=pmix \
     --container-mount-home \
     --no-container-remap-root \
     --pty bash
```

- now create the fine tuning code to test

```
cat << 'EOF' > ~/finetune_lora.py
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# ---- Config ----
model_name = "meta-llama/Llama-3.2-1B-Instruct"
output_dir = os.path.expanduser("~/finetune-results")
num_epochs = 20
batch_size = 4
gradient_accumulation = 4
learning_rate = 2e-4
max_seq_length = 1024
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05

# ---- Quantization (QLoRA) ----
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# ---- Tokenizer ----
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ---- Model ----
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    dtype=torch.bfloat16,
    device_map={"": int(os.environ.get("LOCAL_RANK", 0))},
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)

# ---- LoRA ----
lora_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ---- Dataset ----
dataset = load_dataset("rajpurkar/squad", split="train[:5000]")

def format_sample(example):
    return {
        "text": f"### Question: {example['question']}\n### Context: {example['context']}\n### Answer: {example['answers']['text'][0]}"
    }

dataset = dataset.map(format_sample, remove_columns=dataset.column_names)

# ---- Tokenize ----
def tokenize_fn(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

dataset = dataset.map(tokenize_fn, remove_columns=["text"], batched=False)

# ---- Data Collator ----
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ---- Training Args ----
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation,
    learning_rate=learning_rate,
    bf16=True,
    logging_steps=10,
    save_steps=100,
    save_total_limit=3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    optim="paged_adamw_32bit",
    gradient_checkpointing=True,
    report_to="none",
    ddp_find_unused_parameters=False,
    remove_unused_columns=False,
)

# ---- Trainer ----
trainer = Trainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    data_collator=data_collator,
)

# ---- Train ----
print("Starting training...")
trainer.train()

# ---- Save ----
print(f"Saving model to {output_dir}")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print("Done!")
EOF
```

- install hugginface and wandb as needed

```
pip install huggingface_hub
pip install wandb
```

- now lets run with multiple gpu's

```
torchrun --nproc_per_node=4 ~/finetune_lora.py
```

- now try to run a GRPO fine tuning with RL

```
cat << 'EOF' > ~/finetune_grpo.py
import os
import re
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig

# ---- WandB Login ----
wandb.login()
wandb.init(
    project="qwen3-grpo-math",
    name="grpo-qwen3-1.7b-gsm8k",
    tags=["grpo", "qwen3", "qlora", "gsm8k"],
)

# ---- Config ----
model_name = "Qwen/Qwen3-1.7B"
output_dir = os.path.expanduser("~/grpo-results")
hf_repo = "Balab2021/Qwen3-1.7B-GRPO-Math"

# ---- Quantization (QLoRA) ----
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# ---- Tokenizer ----
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# ---- Model ----
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    dtype=torch.bfloat16,
    device_map={"": int(os.environ.get("LOCAL_RANK", 0))},
    trust_remote_code=True,
)

# ---- LoRA ----
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
    bias="none",
)

# ---- Dataset: GSM8K (reduced to 500) ----
dataset = load_dataset("openai/gsm8k", "main", split="train[:500]")

SYSTEM_PROMPT = (
    "You are a helpful math assistant. "
    "Solve the problem step by step. "
    "Put your final numerical answer inside \\boxed{}."
)

def format_dataset(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ],
        "answer": example["answer"].split("####")[-1].strip(),
    }

dataset = dataset.map(format_dataset, remove_columns=["question"])

# ===========================================================
#  HELPER
# ===========================================================
def extract_text(completion):
    if isinstance(completion, str):
        return completion
    elif isinstance(completion, list):
        texts = []
        for msg in completion:
            if isinstance(msg, dict) and "content" in msg:
                texts.append(msg["content"])
            elif isinstance(msg, str):
                texts.append(msg)
        return "\n".join(texts)
    else:
        return str(completion)

# ===========================================================
#  REWARD FUNCTIONS
# ===========================================================
def correctness_reward_fn(completions, answer, **kwargs):
    rewards = []
    for completion, ans in zip(completions, answer):
        text = extract_text(completion)
        match = re.search(r'\\boxed\{([^}]*)\}', text)
        if match:
            predicted = match.group(1).strip().replace(",", "")
            expected = ans.strip().replace(",", "")
            rewards.append(2.0 if predicted == expected else -1.0)
        else:
            rewards.append(-1.0)
    return rewards

def format_reward_fn(completions, **kwargs):
    rewards = []
    for completion in completions:
        text = extract_text(completion)
        if re.search(r'\\boxed\{[^}]+\}', text):
            rewards.append(1.0)
        else:
            rewards.append(-0.5)
    return rewards

def reasoning_reward_fn(completions, **kwargs):
    rewards = []
    for completion in completions:
        text = extract_text(completion)
        lines = text.strip().split("\n")
        if len(lines) >= 3:
            rewards.append(0.5)
        elif len(lines) >= 2:
            rewards.append(0.2)
        else:
            rewards.append(-0.5)
    return rewards

# ---- GRPO Config (OPTIMIZED + WANDB) ----
grpo_config = GRPOConfig(
    output_dir=output_dir,
    num_generations=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    learning_rate=5e-6,
    bf16=True,
    logging_steps=5,
    save_steps=50,
    save_total_limit=3,
    gradient_checkpointing=True,
    report_to="wandb",
    max_completion_length=256,
    push_to_hub=True,
    hub_model_id=hf_repo,
    hub_strategy="end",
)

# ---- Trainer ----
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[correctness_reward_fn, format_reward_fn, reasoning_reward_fn],
    args=grpo_config,
    train_dataset=dataset,
    peft_config=lora_config,
    processing_class=tokenizer,
)

# ---- Train ----
print("Starting GRPO training...")
trainer.train()

# ---- Save ----
print(f"Saving model to {output_dir}")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

# ---- Push to HuggingFace Hub ----
print(f"Pushing model to HuggingFace: {hf_repo}")
trainer.push_to_hub(commit_message="GRPO fine-tuned Qwen3-1.7B on GSM8K")
print(f"Done! Model available at: https://huggingface.co/{hf_repo}")

# ---- Finish WandB ----
wandb.finish()
EOF
```

- run with multiple GPU's

```
# Login to WandB (uses your stored credentials in ~/.netrc)
pip install wandb --upgrade
wandb login

# Login to HuggingFace
huggingface-cli login

# Run
torchrun --nproc_per_node=4 ~/finetune_grpo.py
```

- in my cluster i got default 4 gpu with 5 hour time limit.