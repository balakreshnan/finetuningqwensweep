# Skill: GRPO RL Fine-Tuning of Open-Source Models (TRL + QLoRA on Slurm)

## Description
Run GRPO (Group Relative Policy Optimization) reinforcement-learning fine-tuning on open-source
causal LLMs (Qwen3, Llama, Nemotron, etc.) using HuggingFace TRL `GRPOTrainer` + QLoRA (4-bit NF4)
on a multi-GPU Slurm/enroot cluster. Covers cluster allocation, environment bootstrap, custom
reward-function design, `torchrun`/`accelerate` multi-GPU launch, WandB metrics, and HF Hub push.

Validated on two clusters:
- **Hecate** — Rubin 4-GPU nodes, partition `batch-xdr`
- **Pytche** — B200 4-GPU nodes, partition `36x2-a01r`

## When to Use
- User asks for GRPO / RLHF / RL-based / reward-model fine-tuning
- User asks to train with TRL `GRPOTrainer` or `GRPOConfig`
- User asks to fine-tune a model on GSM8K or another verifiable-reward (math/code/format) task
- User wants reward functions that score correctness, output format, or reasoning depth
- User asks to run multi-GPU RL fine-tuning on Hecate / Pytche / a Slurm + enroot cluster

## Prerequisites
- 4-GPU node (Rubin on Hecate, B200 on Pytche) — GRPO generates *k* completions per prompt, so it is
  generation-bound and wants all GPUs
- Slurm account with a batch partition (`general_sa` in the reference runs)
- Enroot credentials for `gitlab-master.nvidia.com` and `nvcr.io`
- HuggingFace token (model download + adapter push)
- WandB API key (metrics)

---

## Step 1: Check Cluster Availability

```bash
sinfo --summarize
sacctmgr show associations user=$USER format=Account,Partition
squeue -u $USER
```

Pick a partition from the association list; do not assume one is available.

---

## Step 2: Create Enroot Credentials

```bash
mkdir -p ~/.config/enroot
touch ~/.config/enroot/.credentials
```

```bash
cat > ~/.config/enroot/.credentials << 'CREDS'
machine gitlab-master.nvidia.com login <EMAIL> password <GITLAB_TOKEN>
machine nvcr.io login $oauthtoken password <NGC_API_KEY>
CREDS
```

> Never commit this file. `$oauthtoken` is a literal string for nvcr.io, not a variable.

---

## Step 3: Grab an Interactive Allocation

**Hecate (Rubin):**
```bash
srun --account=general_sa \
     --partition=batch-xdr \
     --nodes=1 \
     --ntasks-per-node=1 \
     --time=5:00:00 \
     --job-name=general_sa-finetune:interactive \
     --container-image=gitlab-master.nvidia.com/dl/dgx/pytorch:main-py3-devel \
     --container-mount-home \
     --no-container-remap-root \
     --mpi=pmix \
     --pty bash
```

**Pytche (B200):** identical, but `--partition=36x2-a01r`.

To pin a specific image tag instead of `main`, use e.g. `pytorch:26.04-py3-devel`.

| Property | Hecate | Pytche |
|----------|--------|--------|
| Partition | `batch-xdr` | `36x2-a01r` |
| GPU | Rubin x4 | B200 x4 |
| Reference wall-clock (1 epoch, 500 GSM8K rows) | ~6 min | ~15 min |

---

## Step 4: Environment Bootstrap

The DGX PyTorch devel image already has torch/CUDA. Install only what's missing:

```bash
pip install transformers datasets accelerate peft bitsandbytes trl
pip install huggingface_hub wandb
```

**Check the TRL version before doing anything else:**

```bash
python -c "from trl import GRPOTrainer, GRPOConfig; print('OK')"
nvidia-smi
python -c "import torch; print(torch.cuda.device_count(), 'GPUs available')"
```

If that import prints `OK`, TRL is new enough — **skip the rest of this step and go to Step 5.** A
current `pip install trl` normally gives you a version well past 0.18, so this is the common case.

**Only if the import fails** (`cannot import name 'GRPOConfig'`, i.e. the image pinned an older TRL),
force an upgrade with `--no-deps` so pip cannot drag in a torch that conflicts with the container's:

```bash
pip install "trl>=0.18" --no-deps
pip install "datasets>=3.0" --no-deps
```

Then re-run the import check.

---

## Step 5: Authenticate

```bash
hf auth login --token "<HF_TOKEN>"
wandb login
```

If the interactive `hf auth login` fails inside the container (no TTY / no keyring), fall back to env vars:

```bash
export HF_TOKEN="<HF_TOKEN>"
export WANDB_API_KEY="<WANDB_API_KEY>"
```

---

## Step 6: Write the Training Script

Write this to `~/finetune_grpo.py` (heredoc it in, or use an editor):

```python
import os
import re
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig

# ---- WandB ----
wandb.login()
wandb.init(
    project="qwen3-grpo-math",
    name="grpo-qwen3-1.7b-gsm8k",
    tags=["grpo", "qwen3", "qlora", "gsm8k"],
)

# ---- Config ----
model_name = "Qwen/Qwen3-1.7B"
output_dir = os.path.expanduser("~/grpo-results")
hf_repo = "<HF_USERNAME>/Qwen3-1.7B-GRPO-Math"

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
tokenizer.padding_side = "left"   # REQUIRED for generation-based RL

# ---- Model (one full replica per rank) ----
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
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
    bias="none",
)

# ---- Dataset: GSM8K ----
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
#  HELPER — completions may be a str, or a list of chat dicts
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
#  Signature: fn(completions, **kwargs) -> list[float]
#  Any extra dataset column (e.g. "answer") arrives as a kwarg
#  and is a LIST aligned with completions.
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
        rewards.append(1.0 if re.search(r'\\boxed\{[^}]+\}', text) else -0.5)
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

# ---- GRPO Config ----
grpo_config = GRPOConfig(
    output_dir=output_dir,
    num_generations=2,               # completions sampled per prompt (group size)
    per_device_train_batch_size=4,   # MUST be divisible by num_generations
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    learning_rate=5e-6,              # RL wants a much lower LR than SFT
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
    processing_class=tokenizer,      # NOT tokenizer= (deprecated)
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

wandb.finish()
```

---

## Step 7: Launch Multi-GPU

```bash
torchrun --nproc_per_node=4 ~/finetune_grpo.py
```

or, auto-detecting GPU count:

```bash
accelerate launch --num_processes=$(nvidia-smi -L | wc -l) --multi_gpu ~/finetune_grpo.py
```

Both launchers were validated. `device_map={"": LOCAL_RANK}` is what makes either one place a full
model replica per rank — do **not** use `device_map="auto"` under `torchrun`; it shards one model
across all GPUs and every rank fights for the same devices.

---

## Step 8: Monitor

The WandB run URL prints at startup:

```
https://wandb.ai/<WANDB_ENTITY>/qwen3-grpo-math/runs/<RUN_ID>
```

Key GRPO panels to watch:

| Metric | Healthy signal |
|--------|----------------|
| `train/reward` | Trending up |
| `train/reward_std` | Non-zero — zero means every completion in a group scored identically, so there is no gradient signal |
| `rewards/correctness_reward_fn` | Rising from about -1 toward positive |
| `rewards/format_reward_fn` | Saturates near +1 quickly (easiest reward to learn) |
| `train/kl` | Small and stable — a spike means the policy is drifting off the reference model |
| `train/completion_length` | Should not collapse toward 0 nor pin at `max_completion_length` |

---

## Reward Function Design Rules

1. **Signature**: `fn(completions, **kwargs) -> list[float]`, one float per completion, same length as
   `completions`. Every non-`prompt` dataset column is passed in as a same-length list kwarg.
2. **Always defend the parse.** A completion that doesn't match your regex must still yield a float —
   return a penalty, never raise. One exception kills the whole run mid-training.
3. **Stack cheap shaping rewards under one verifiable reward.** Correctness (+/-2) dominates; format
   (+/-1) and reasoning-depth (+/-0.5) are shaping terms that give gradient before the model is ever
   right. Keep shaping magnitudes strictly below the correctness magnitude, or the model learns to
   emit well-formatted nonsense.
4. **Handle both completion shapes** via `extract_text` — conversational datasets yield lists of chat
   dicts, plain-text datasets yield strings.
5. Rewards are group-normalized by GRPO, so absolute scale matters less than *relative spread within
   a group*. A reward returning the same value for every completion contributes nothing.

---

## Tuning Reference

| Goal | Lever | Notes |
|------|-------|-------|
| Better RL signal | raise `num_generations` (4-8) | Biggest quality lever; cost is linear in generation time |
| Fit memory | lower `per_device_train_batch_size` | Must stay divisible by `num_generations` |
| Fit memory | QLoRA 4-bit + `gradient_checkpointing=True` | Already on in the template |
| Faster epochs | lower `max_completion_length` | 256 is a smoke-test setting; real math runs want 512-1024 |
| Stability | keep `learning_rate` at 5e-6 to 1e-5 | SFT-scale LRs (2e-4) destabilize GRPO |
| More capacity | raise LoRA `r` to 32/64 | Also raise `lora_alpha` to 2x r |
| Throughput | more GPUs via `--nproc_per_node` | GRPO is generation-bound, scales well |

---

## Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `ImportError: cannot import name 'GRPOConfig'` | Image pinned an old TRL | `pip install "trl>=0.18" --no-deps` — only needed when the import actually fails |
| pip downgrades torch when installing TRL | Dependency resolver | Use `--no-deps` on the container image |
| `batch size must be divisible by num_generations` | Mismatched config | Make `per_device_train_batch_size` a multiple of `num_generations` |
| Generations are garbage / repeated pad tokens | `padding_side` left as `"right"` | Set `tokenizer.padding_side = "left"` |
| Hang at model load under `torchrun` | `device_map="auto"` | Use `device_map={"": int(os.environ["LOCAL_RANK"])}` |
| `reward_std == 0` at every step | Reward too coarse, or `num_generations=1` | Add shaping rewards; raise `num_generations` |
| Reward function raises mid-run | Unguarded `.group(1)` on a `None` match | Return a penalty on no-match instead |
| `tokenizer` deprecation warning in Trainer | API change | Pass `processing_class=tokenizer` |
| HF login fails in container | No TTY / no keyring | `export HF_TOKEN=...` and `export WANDB_API_KEY=...` |
| CUDA OOM during generation | group size x completion length | Lower `num_generations`, `max_completion_length`, batch size |

---

## Placeholder Reference

| Placeholder | Example Value |
|-------------|---------------|
| `<PARTITION>` | `batch-xdr` (Hecate) / `36x2-a01r` (Pytche) |
| `<ACCOUNT>` | `general_sa` |
| `<HF_TOKEN>` | `hf_xxxxxxxxxxxxxxxx` |
| `<WANDB_API_KEY>` | `xxxxxxxxxxxxxxxx` |
| `<HF_USERNAME>` | `Balab2021` |
| `<WANDB_ENTITY>` | `balabala76` |
| `<MODEL_NAME>` | `Qwen/Qwen3-1.7B` |
| `<NUM_GPUS>` | `4` |

---

## Source Runs

- [hecategrpo.md](../hecategrpo.md) — Rubin x4, `batch-xdr`, ~6 min for 1 epoch
- [pytchegrpo.md](../pytchegrpo.md) — B200 x4, `36x2-a01r`, ~15 min for 1 epoch
- Companion SFT/LoRA skill: [nemobackfinetuneLoRa-SKILL.md](nemobackfinetuneLoRa-SKILL.md)
