"""
Fine-tune Qwen2.5-3B-Instruct on OpenAssistant/oasst1 using QLoRA.

Usage:
    python finetune_qwen3b.py \
        --hf_token hf_xxxxx \
        --output_dir ./output/qwen3b-oasst1-lora

Requirements:
    pip install torch transformers datasets accelerate peft trl bitsandbytes scipy
"""

import os
import argparse
import csv
import copy
import gc
from datetime import datetime
from collections import defaultdict

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer


# =============================================================================
# ARGUMENT PARSING
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-3B with QLoRA")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                        help="HuggingFace model ID")
    parser.add_argument("--dataset_name", type=str, default="OpenAssistant/oasst1",
                        help="HuggingFace dataset ID")
    parser.add_argument("--output_dir", type=str, default="./output/qwen3b-oasst1-lora",
                        help="Directory to save checkpoints and final adapter")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace token (or set HF_TOKEN env var)")

    # Training hyperparameters
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # LoRA hyperparameters
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Options
    parser.add_argument("--use_4bit", action="store_true", default=True,
                        help="Use 4-bit quantization (QLoRA)")
    parser.add_argument("--no_4bit", action="store_true",
                        help="Disable 4-bit quantization (full precision LoRA)")
    parser.add_argument("--packing", action="store_true", default=True,
                        help="Enable sequence packing")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit training samples (for quick testing)")

    parser.add_argument("--run_experiments", action="store_true",
                        help="Run four predefined fine-tuning experiments")
    parser.add_argument("--target_loss_gap", type=float, default=0.10,
                        help="Target eval_loss - train_loss gap")
    return parser.parse_args()


# =============================================================================
# DATASET PREPARATION
# =============================================================================
def build_conversation_pairs(dataset):
    """
    Extract (instruction, response) pairs from oasst1's tree structure.
    Selects the highest-ranked assistant reply for each prompter message.
    """
    messages = {row["message_id"]: row for row in dataset}
    children = defaultdict(list)
    for row in dataset:
        if row["parent_id"]:
            children[row["parent_id"]].append(row)

    pairs = []
    for msg_id, msg in messages.items():
        if msg["role"] == "prompter" and msg_id in children:
            replies = [c for c in children[msg_id] if c["role"] == "assistant"]
            if replies:
                replies.sort(key=lambda x: x.get("rank", 999) or 999)
                best_reply = replies[0]
                pairs.append({
                    "instruction": msg["text"],
                    "response": best_reply["text"],
                })
    return pairs


def format_as_qwen_chat(sample):
    """Format instruction-response pair using Qwen's ChatML template."""
    text = (
        f"<|im_start|>system\n"
        f"You are a helpful, harmless, and honest AI assistant.<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{sample['instruction']}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"{sample['response']}<|im_end|>"
    )
    return {"text": text}


def prepare_dataset(dataset_name, max_samples=None):
    """Load and prepare the OpenAssistant dataset."""
    print(f"\n{'='*70}")
    print(f"  LOADING DATASET: {dataset_name}")
    print(f"{'='*70}")

    raw_dataset = load_dataset(dataset_name)

    # Filter English conversations
    train_data = raw_dataset["train"].filter(lambda x: x["lang"] == "en")
    val_data = raw_dataset["validation"].filter(lambda x: x["lang"] == "en")
    print(f"  English train samples: {len(train_data)}")
    print(f"  English val samples  : {len(val_data)}")

    # Build instruction-response pairs
    print("  Building instruction-response pairs...")
    train_pairs = build_conversation_pairs(train_data)
    val_pairs = build_conversation_pairs(val_data)
    print(f"  Train pairs extracted: {len(train_pairs)}")
    print(f"  Val pairs extracted  : {len(val_pairs)}")

    # Convert to HF Dataset
    train_dataset = Dataset.from_list(train_pairs)
    val_dataset = Dataset.from_list(val_pairs)

    # Limit samples if requested (useful for testing)
    if max_samples:
        train_dataset = train_dataset.select(range(min(max_samples, len(train_dataset))))
        val_dataset = val_dataset.select(range(min(max_samples // 5, len(val_dataset))))
        print(f"  Limited to: {len(train_dataset)} train, {len(val_dataset)} val")

    # Format with Qwen chat template
    train_dataset = train_dataset.map(format_as_qwen_chat)
    val_dataset = val_dataset.map(format_as_qwen_chat)

    # Print sample
    print(f"\n  Sample formatted entry:")
    print(f"  {'-'*60}")
    print(f"  {train_dataset[0]['text'][:200]}...")
    print(f"  {'-'*60}")

    return train_dataset, val_dataset


# =============================================================================
# MODEL LOADING
# =============================================================================
def load_model_and_tokenizer(model_name, use_4bit=True):
    """Load model with optional 4-bit quantization and tokenizer."""
    print(f"\n{'='*70}")
    print(f"  LOADING MODEL: {model_name}")
    print(f"{'='*70}")

    # Tokenizer
    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"  Tokenizer vocab size: {tokenizer.vocab_size}")

    # Quantization config
    if use_4bit:
        print("  Using 4-bit quantization (QLoRA)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None
        print("  Using full precision (no quantization)...")

    # Load model
    print("  Loading model weights...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    model.config.use_cache = False  # Required for gradient checkpointing

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded successfully!")
    print(f"  Total parameters: {total_params:,}")
    print(f"  GPU memory used : {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    return model, tokenizer


# =============================================================================
# LORA SETUP
# =============================================================================
def apply_lora(model, lora_r=16, lora_alpha=32, lora_dropout=0.05):
    """Apply LoRA adapters to the model."""
    print(f"\n{'='*70}")
    print(f"  APPLYING LoRA ADAPTERS")
    print(f"{'='*70}")
    print(f"  r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",   # Query projection
            "k_proj",   # Key projection
            "v_proj",   # Value projection
            "o_proj",   # Output projection
            "gate_proj",  # MLP gate
            "up_proj",    # MLP up
            "down_proj",  # MLP down
        ],
    )

    model = get_peft_model(model, lora_config)

    # Print trainable parameter info
    model.print_trainable_parameters()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model


# =============================================================================
# TRAINING
# =============================================================================
def train(model, tokenizer, train_dataset, val_dataset, args):
    """Configure and run the SFT training loop."""
    print(f"\n{'='*70}")
    print("  STARTING TRAINING")
    print(f"{'='*70}")
    print(f"  Epochs            : {args.num_epochs}")
    print(f"  Batch size        : {args.batch_size}")
    print(f"  Grad accumulation : {args.gradient_accumulation}")
    print(f"  Effective batch   : {args.batch_size * args.gradient_accumulation}")
    print(f"  Learning rate     : {args.learning_rate}")
    print(f"  Max seq length    : {args.max_seq_length}")
    print(f"  Output dir        : {args.output_dir}")

    import math
    optimizer_steps_per_epoch = math.ceil(
        len(train_dataset) /
        max(1, args.batch_size * args.gradient_accumulation)
    )
    total_optimizer_steps = max(
        1, optimizer_steps_per_epoch * args.num_epochs
    )
    warmup_steps = max(
        1, int(total_optimizer_steps * args.warmup_ratio)
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        report_to="none",
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
    )

    train_result = trainer.train()
    train_loss = float(train_result.metrics["train_loss"])
    train_runtime = float(train_result.metrics["train_runtime"])
    samples_per_second = float(
        train_result.metrics.get("train_samples_per_second", 0.0)
    )

    print("\n  Training Complete!")
    print(f"  {'─'*40}")
    print(f"  Loss           : {train_loss:.4f}")
    print(f"  Runtime        : {train_runtime:.1f}s")
    print(f"  Samples/sec    : {samples_per_second:.2f}")
    print(f"  Steps          : {trainer.state.global_step}")

    print("\n  Running evaluation...")
    eval_result = trainer.evaluate()
    eval_loss = float(eval_result["eval_loss"])
    loss_gap = eval_loss - train_loss

    print(f"  Eval loss      : {eval_loss:.4f}")
    print(f"  Loss gap       : {loss_gap:.4f}")

    metrics = {
        "train_loss": train_loss,
        "eval_loss": eval_loss,
        "loss_gap": loss_gap,
        "train_runtime": train_runtime,
        "samples_per_second": samples_per_second,
        "global_steps": int(trainer.state.global_step),
    }

    return trainer, eval_result, metrics


# =============================================================================
# SAVE MODEL
# =============================================================================
def save_adapter(trainer, tokenizer, output_dir):
    """Save the fine-tuned LoRA adapter."""
    adapter_path = os.path.join(output_dir, "final_adapter")
    print(f"\n{'='*70}")
    print(f"  SAVING ADAPTER")
    print(f"{'='*70}")
    print(f"  Path: {adapter_path}")

    os.makedirs(adapter_path, exist_ok=True)
    trainer.save_model(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    # List saved files
    saved_files = os.listdir(adapter_path)
    print(f"  Saved files:")
    for f in sorted(saved_files):
        size = os.path.getsize(os.path.join(adapter_path, f))
        print(f"    {f} ({size / 1e6:.1f} MB)")

    return adapter_path


# =============================================================================
# INFERENCE TEST
# =============================================================================
def run_inference_test(model, tokenizer):
    """Run a few test prompts through the fine-tuned model."""
    print(f"\n{'='*70}")
    print(f"  INFERENCE TEST")
    print(f"{'='*70}")

    model.eval()

    test_prompts = [
        "Explain quantum computing to a 10-year-old.",
        "Write a Python function that checks if a number is prime.",
        "What are the key differences between machine learning and deep learning?",
    ]

    for i, prompt in enumerate(test_prompts, 1):
        # Format with Qwen chat template
        formatted_input = (
            f"<|im_start|>system\n"
            f"You are a helpful, harmless, and honest AI assistant.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        inputs = tokenizer(formatted_input, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the generated tokens
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        print(f"\n  ┌─ Test {i} {'─'*55}")
        print(f"  │ Q: {prompt}")
        print(f"  │ A: {response[:400]}")
        print(f"  └{'─'*63}")


# =============================================================================
# MULTI-RUN EXPERIMENTS
# =============================================================================
EXPERIMENTS = [
    {
        "name": "run1_balanced",
        "num_epochs": 2,
        "learning_rate": 1e-4,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.10,
        "weight_decay": 0.03,
    },
    {
        "name": "run2_regularized",
        "num_epochs": 2,
        "learning_rate": 1e-4,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.10,
        "weight_decay": 0.05,
    },
    {
        "name": "run3_low_lr",
        "num_epochs": 3,
        "learning_rate": 5e-5,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.10,
        "weight_decay": 0.03,
    },
    {
        "name": "run4_strong_regularization",
        "num_epochs": 2,
        "learning_rate": 7.5e-5,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.15,
        "weight_decay": 0.05,
    },
]


def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def run_experiment_suite(args, train_dataset, val_dataset):
    timestamp = datetime.now().strftime("%d%m%y_%H%M%S")
    suite_dir = os.path.join(
        args.output_dir,
        f"experiments_{timestamp}",
    )
    os.makedirs(suite_dir, exist_ok=True)
    results = []

    for index, config in enumerate(EXPERIMENTS, start=1):
        run_args = copy.deepcopy(args)
        for key, value in config.items():
            if key != "name":
                setattr(run_args, key, value)

        run_args.output_dir = os.path.join(
            suite_dir,
            config["name"],
        )

        print(f"\n{'#'*70}")
        print(
            f"  EXPERIMENT {index}/{len(EXPERIMENTS)}: "
            f"{config['name']}"
        )
        print(f"{'#'*70}")

        model = None
        tokenizer = None
        trainer = None

        try:
            model, tokenizer = load_model_and_tokenizer(
                run_args.model_name,
                use_4bit=not run_args.no_4bit,
            )
            model = apply_lora(
                model,
                run_args.lora_r,
                run_args.lora_alpha,
                run_args.lora_dropout,
            )

            trainer, eval_result, metrics = train(
                model,
                tokenizer,
                train_dataset,
                val_dataset,
                run_args,
            )

            adapter_path = save_adapter(
                trainer,
                tokenizer,
                run_args.output_dir,
            )

            results.append({
                "run": config["name"],
                "status": "completed",
                "adapter_path": adapter_path,
                "output_dir": run_args.output_dir,
                "num_epochs": run_args.num_epochs,
                "learning_rate": run_args.learning_rate,
                "lora_r": run_args.lora_r,
                "lora_alpha": run_args.lora_alpha,
                "lora_dropout": run_args.lora_dropout,
                "weight_decay": run_args.weight_decay,
                **metrics,
            })

        except Exception as exc:
            print(f"  Experiment failed: {exc}")
            results.append({
                "run": config["name"],
                "status": "failed",
                "error": str(exc),
                "output_dir": run_args.output_dir,
            })

        finally:
            del trainer, model, tokenizer
            cleanup_gpu()

    csv_path = os.path.join(
        suite_dir,
        "experiment_results.csv",
    )
    fieldnames = sorted({
        key for row in results for key in row.keys()
    })

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(results)

    completed = [
        row for row in results
        if row["status"] == "completed"
    ]

    print(f"\n{'='*70}")
    print("  EXPERIMENT COMPARISON")
    print(f"{'='*70}")

    if completed:
        for row in sorted(
            completed,
            key=lambda item: item["eval_loss"],
        ):
            status = (
                "TARGET MET"
                if row["loss_gap"] < args.target_loss_gap
                else "above target"
            )
            print(
                f"  {row['run']:<28} "
                f"train={row['train_loss']:.4f} "
                f"eval={row['eval_loss']:.4f} "
                f"gap={row['loss_gap']:.4f} "
                f"{status}"
            )

        target_runs = [
            row for row in completed
            if row["loss_gap"] < args.target_loss_gap
        ]
        candidates = target_runs or completed
        best = min(
            candidates,
            key=lambda item: item["eval_loss"],
        )

        print(f"\n  Recommended run : {best['run']}")
        print(f"  Eval loss       : {best['eval_loss']:.4f}")
        print(f"  Loss gap        : {best['loss_gap']:.4f}")
        print(f"  Adapter path    : {best['adapter_path']}")
    else:
        print("  No experiment completed successfully.")

    print(f"  Results CSV     : {csv_path}")
    print("=" * 70)

    return results


# =============================================================================
# MAIN
# =============================================================================
def main():
    args = parse_args()

    # Print banner
    print("\n" + "=" * 70)
    print("  QWEN2.5-3B FINE-TUNING WITH QLoRA")
    print("  " + "─" * 66)
    print(f"  Model   : {args.model_name}")
    print(f"  Dataset : {args.dataset_name}")
    print(f"  Output  : {args.output_dir}")
    print(f"  GPU     : {torch.cuda.get_device_name(0)}")
    #print(f"  VRAM    : {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print(f"  PyTorch : {torch.__version__}")
    print(f"  CUDA    : {torch.version.cuda}")
    print("=" * 70)

    # Authenticate with HuggingFace
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)
        print("  ✓ Authenticated with HuggingFace")
    else:
        print("  ⚠ No HF_TOKEN provided — using cached credentials or public models")

    # Step 1: Prepare dataset
    train_dataset, val_dataset = prepare_dataset(args.dataset_name, args.max_samples)

    if args.run_experiments:
        run_experiment_suite(
            args,
            train_dataset,
            val_dataset,
        )
        return

    # Step 2: Load model
    use_4bit = not args.no_4bit
    model, tokenizer = load_model_and_tokenizer(args.model_name, use_4bit=use_4bit)

    # Step 3: Apply LoRA
    model = apply_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout)

    # Step 4: Train
    trainer, eval_result, metrics = train(model, tokenizer, train_dataset, val_dataset, args)

    # Step 5: Save
    adapter_path = save_adapter(trainer, tokenizer, args.output_dir)

    # Step 6: Inference test
    run_inference_test(model, tokenizer)

    # Final summary
    print(f"\n{'='*70}")
    print(f"  ✅ ALL DONE!")
    print(f"{'='*70}")
    print(f"  Adapter saved at: {adapter_path}")
    print(f"  Eval loss       : {eval_result['eval_loss']:.4f}")
    print(f"")
    print(f"  Load your fine-tuned model:")
    print(f"    from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f"    from peft import PeftModel")
    print(f"")
    print(f"    tokenizer = AutoTokenizer.from_pretrained('{adapter_path}')")
    print(f"    base = AutoModelForCausalLM.from_pretrained('{args.model_name}')")
    print(f"    model = PeftModel.from_pretrained(base, '{adapter_path}')")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
