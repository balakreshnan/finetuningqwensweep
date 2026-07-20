#!/usr/bin/env python3
"""
Fine-tune Qwen/Qwen3.5-2B with Hugging Face TRL using a public dataset,
log training/evaluation metrics and system statistics to Weights & Biases,
and push the trained LoRA adapter (and optionally a merged model) to the
Hugging Face Hub.

Recommended environment:
    Python 3.10+
    NVIDIA GPU with CUDA
    ~16 GB VRAM for the default 4-bit QLoRA settings

Install:
    pip install -U torch torchvision
    pip install -U "transformers @ git+https://github.com/huggingface/transformers.git@main"
    pip install -U trl peft accelerate datasets bitsandbytes wandb huggingface_hub sentencepiece pillow

Authenticate:
    export HF_TOKEN="hf_..."
    export WANDB_API_KEY="..."
    export WANDB_PROJECT="qwen35-finetuning"

Example:
    python finetune_qwen35_2b_trl_wandb.py \
        --hub-repo-id Balab2021/Qwen3.5-2B-Capybara-LoRA \
        --max-train-samples 10000 \
        --num-train-epochs 1

For a quick smoke test:
    python finetune_qwen35_2b_trl_wandb.py \
        --hub-repo-id Balab2021/Qwen3.5-2B-Capybara-LoRA-test \
        --max-train-samples 200 \
        --max-eval-samples 50 \
        --max-steps 20
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import wandb
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, login as hf_login
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from trl import SFTConfig, SFTTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen/Qwen3.5-2B with TRL, W&B, and Hugging Face Hub."
    )

    # Model and dataset
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--dataset-id", default="trl-lib/Capybara")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument(
        "--hub-repo-id",
        required=True,
        help="Target Hub repository, for example Balab2021/Qwen3.5-2B-Capybara-LoRA",
    )
    parser.add_argument("--output-dir", default="./output/qwen35-2b-capybara-lora")
    parser.add_argument("--max-train-samples", type=int, default=10000)
    parser.add_argument("--max-eval-samples", type=int, default=500)
    parser.add_argument("--eval-size", type=float, default=0.02)
    parser.add_argument("--max-length", type=int, default=2048)

    # Training
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Set to a positive value to override num_train_epochs.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_false",
        dest="gradient_checkpointing",
    )
    parser.add_argument("--packing", action="store_true", default=False)

    # LoRA / QLoRA
    parser.add_argument("--use-4bit", action="store_true", default=True)
    parser.add_argument("--no-4bit", action="store_false", dest="use_4bit")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    # W&B and Hub
    parser.add_argument(
        "--wandb-project",
        default=os.getenv("WANDB_PROJECT", "qwen35-finetuning"),
    )
    parser.add_argument("--wandb-entity", default=os.getenv("WANDB_ENTITY"))
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--push-merged-model", action="store_true")
    parser.add_argument("--private-repo", action="store_true")
    parser.add_argument(
        "--hub-strategy",
        choices=["end", "every_save", "checkpoint", "all_checkpoints"],
        default="end",
    )

    return parser.parse_args()


def require_tokens() -> tuple[str, str]:
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    wandb_key = os.getenv("WANDB_API_KEY")

    missing = []
    if not hf_token:
        missing.append("HF_TOKEN")
    if not wandb_key:
        missing.append("WANDB_API_KEY")

    if missing:
        raise RuntimeError(
            "Missing required environment variable(s): "
            + ", ".join(missing)
            + ". Set them before running this script."
        )
    return hf_token, wandb_key


def choose_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def prepare_dataset(args: argparse.Namespace) -> tuple[Dataset, Dataset]:
    dataset = load_dataset(args.dataset_id, split=args.dataset_split)

    if "messages" not in dataset.column_names:
        raise ValueError(
            f"Dataset {args.dataset_id!r} does not contain a 'messages' column. "
            f"Found columns: {dataset.column_names}"
        )

    # Cap the source data before splitting to make experiments reproducible and fast.
    requested_total = args.max_train_samples + args.max_eval_samples
    if requested_total > 0 and len(dataset) > requested_total:
        dataset = dataset.shuffle(seed=args.seed).select(range(requested_total))

    if args.max_eval_samples <= 0:
        raise ValueError("--max-eval-samples must be greater than zero.")

    # Use an explicit evaluation count when possible.
    eval_count = min(args.max_eval_samples, max(1, len(dataset) - 1))
    split = dataset.train_test_split(test_size=eval_count, seed=args.seed)

    train_dataset = split["train"]
    eval_dataset = split["test"]

    if args.max_train_samples > 0 and len(train_dataset) > args.max_train_samples:
        train_dataset = train_dataset.select(range(args.max_train_samples))
    if len(eval_dataset) > args.max_eval_samples:
        eval_dataset = eval_dataset.select(range(args.max_eval_samples))

    return train_dataset, eval_dataset


def build_model_and_tokenizer(
    args: argparse.Namespace,
    hf_token: str,
    dtype: torch.dtype,
):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        token=hf_token,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = None
    if args.use_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "4-bit QLoRA requires a CUDA GPU. Run with --no-4bit for CPU, "
                "although CPU full-precision training will be very slow."
            )
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    model_kwargs: dict[str, Any] = {
        "token": hf_token,
        "trust_remote_code": True,
        "dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", 0))}

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    model.config.use_cache = False

    return model, tokenizer


def write_run_summary(
    output_dir: Path,
    args: argparse.Namespace,
    train_metrics: dict[str, Any],
    eval_metrics: dict[str, Any],
    train_size: int,
    eval_size: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "model_id": args.model_id,
        "dataset_id": args.dataset_id,
        "hub_repo_id": args.hub_repo_id,
        "train_examples": train_size,
        "eval_examples": eval_size,
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "eval_perplexity": (
            math.exp(eval_metrics["eval_loss"])
            if "eval_loss" in eval_metrics and eval_metrics["eval_loss"] < 20
            else None
        ),
        "system": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "gpu_count": torch.cuda.device_count(),
        },
        "arguments": vars(args),
    }

    path = output_dir / "run_summary.json"
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    hf_token, wandb_key = require_tokens()
    hf_login(token=hf_token, add_to_git_credential=False)
    wandb.login(key=wandb_key, relogin=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_name = args.wandb_run_name or (
        f"qwen35-2b-capybara-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=vars(args),
        job_type="sft",
    )

    try:
        dtype = choose_dtype()
        print(f"Using dtype: {dtype}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        train_dataset, eval_dataset = prepare_dataset(args)
        print(f"Training examples: {len(train_dataset):,}")
        print(f"Evaluation examples: {len(eval_dataset):,}")

        model, tokenizer = build_model_and_tokenizer(
            args=args,
            hf_token=hf_token,
            dtype=dtype,
        )

        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )

        use_bf16 = dtype == torch.bfloat16
        use_fp16 = dtype == torch.float16

        training_args = SFTConfig(
            output_dir=str(output_dir),
            run_name=run_name,
            max_length=args.max_length,
            packing=args.packing,
            assistant_only_loss=True,
            num_train_epochs=args.num_train_epochs,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_checkpointing=args.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            lr_scheduler_type="cosine",
            optim="paged_adamw_8bit" if args.use_4bit else "adamw_torch",
            logging_strategy="steps",
            logging_steps=args.logging_steps,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            bf16=use_bf16,
            fp16=use_fp16,
            tf32=torch.cuda.is_available(),
            report_to=["wandb"],
            push_to_hub=args.hub_strategy != "end",
            hub_model_id=args.hub_repo_id,
            hub_strategy=args.hub_strategy if args.hub_strategy != "end" else "every_save",
            hub_private_repo=args.private_repo,
            hub_token=hf_token,
            #save_safetensors=True,
            seed=args.seed,
            data_seed=args.seed,
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )

        train_result = trainer.train()
        train_metrics = train_result.metrics
        trainer.log_metrics("train", train_metrics)
        trainer.save_metrics("train", train_metrics)
        trainer.save_state()

        eval_metrics = trainer.evaluate()
        if "eval_loss" in eval_metrics and eval_metrics["eval_loss"] < 20:
            eval_metrics["eval_perplexity"] = math.exp(eval_metrics["eval_loss"])
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

        # Saves the LoRA adapter and tokenizer locally.
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        summary_path = write_run_summary(
            output_dir=output_dir,
            args=args,
            train_metrics=train_metrics,
            eval_metrics=eval_metrics,
            train_size=len(train_dataset),
            eval_size=len(eval_dataset),
        )
        wandb.save(str(summary_path), base_path=str(output_dir))

        # Ensure the Hub repository exists, then upload the final adapter.
        api = HfApi(token=hf_token)
        api.create_repo(
            repo_id=args.hub_repo_id,
            repo_type="model",
            private=args.private_repo,
            exist_ok=True,
        )

        trainer.push_to_hub(
            commit_message=(
                f"Fine-tune {args.model_id} on {args.dataset_id} with TRL and LoRA"
            )
        )

        if args.push_merged_model:
            if args.use_4bit:
                print(
                    "Reloading the base model in BF16/FP16 before merging the adapter..."
                )

            from peft import AutoPeftModelForCausalLM

            merged_dir = output_dir / "merged"
            merged_dir.mkdir(parents=True, exist_ok=True)

            merge_dtype = (
                torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else torch.float16
                if torch.cuda.is_available()
                else torch.float32
            )

            adapter_model = AutoPeftModelForCausalLM.from_pretrained(
                str(output_dir),
                token=hf_token,
                trust_remote_code=True,
                dtype=merge_dtype,
                low_cpu_mem_usage=True,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            merged_model = adapter_model.merge_and_unload()
            merged_model.save_pretrained(
                merged_dir,
                safe_serialization=True,
                max_shard_size="5GB",
            )
            tokenizer.save_pretrained(merged_dir)

            merged_repo_id = args.hub_repo_id.rstrip("/") + "-merged"
            api.create_repo(
                repo_id=merged_repo_id,
                repo_type="model",
                private=args.private_repo,
                exist_ok=True,
            )
            api.upload_folder(
                repo_id=merged_repo_id,
                repo_type="model",
                folder_path=str(merged_dir),
                commit_message="Upload merged fine-tuned model",
            )
            print(f"Merged model pushed to: https://huggingface.co/{merged_repo_id}")

        print("\nTraining complete.")
        print(f"Adapter pushed to: https://huggingface.co/{args.hub_repo_id}")
        print(f"W&B run: {wandb.run.url if wandb.run else 'available in your W&B project'}")
        print(f"Local output: {output_dir.resolve()}")

    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
