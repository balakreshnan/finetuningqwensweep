#!/usr/bin/env python3
"""Fine-tune Qwen3-VL-2B-Instruct on HuggingFaceH4/llava-instruct-mix-vsft."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import wandb
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, login as hf_login
from peft import LoraConfig
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig, set_seed
from trl import SFTConfig, SFTTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multimodal Qwen3-VL SFT with TRL, QLoRA, W&B, and HF Hub")
    p.add_argument("--model-id", default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--dataset-id", default="HuggingFaceH4/llava-instruct-mix-vsft")
    p.add_argument("--train-split", default="train")
    p.add_argument("--eval-split", default="test")
    p.add_argument("--hub-repo-id", required=True)
    p.add_argument("--output-dir", default="./output/qwen3-vl-2b-llava-lora")
    p.add_argument("--max-train-samples", type=int, default=10000)
    p.add_argument("--max-eval-samples", type=int, default=500)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--max-pixels", type=int, default=1024 * 28 * 28)
    p.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    p.add_argument("--num-train-epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--per-device-eval-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=16)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--save-total-limit", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument("--no-gradient-checkpointing", action="store_false", dest="gradient_checkpointing")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--use-4bit", action="store_true", default=True)
    p.add_argument("--no-4bit", action="store_false", dest="use_4bit")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--train-vision-encoder", action="store_true")
    p.add_argument("--attn-implementation", choices=["sdpa", "flash_attention_2", "eager"], default="sdpa")
    p.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT", "qwen3-vl-finetuning"))
    p.add_argument("--wandb-entity", default=os.getenv("WANDB_ENTITY"))
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--private-repo", action="store_true")
    return p.parse_args()


def require_tokens() -> tuple[str, str]:
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    wb_token = os.getenv("WANDB_API_KEY")
    missing = [name for name, val in (("HF_TOKEN", hf_token), ("WANDB_API_KEY", wb_token)) if not val]
    if missing:
        raise RuntimeError("Missing environment variable(s): " + ", ".join(missing))
    return hf_token, wb_token


def choose_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def subset(ds: Dataset, n: int, seed: int) -> Dataset:
    if n <= 0 or n >= len(ds):
        return ds
    return ds.shuffle(seed=seed).select(range(n))


def load_data(args: argparse.Namespace) -> tuple[Dataset, Dataset]:
    train = load_dataset(args.dataset_id, split=args.train_split)
    valid = load_dataset(args.dataset_id, split=args.eval_split)
    for name, ds in (("train", train), ("eval", valid)):
        missing = {"messages", "images"} - set(ds.column_names)
        if missing:
            raise ValueError(f"{name} split missing {sorted(missing)}; found {ds.column_names}")
    return subset(train, args.max_train_samples, args.seed), subset(valid, args.max_eval_samples, args.seed + 1)


def load_model_processor(args: argparse.Namespace, token: str, dtype: torch.dtype):
    processor = AutoProcessor.from_pretrained(args.model_id, token=token, trust_remote_code=True)
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    if hasattr(processor, "image_processor"):
        if hasattr(processor.image_processor, "max_pixels"):
            processor.image_processor.max_pixels = args.max_pixels
        if hasattr(processor.image_processor, "min_pixels"):
            processor.image_processor.min_pixels = args.min_pixels

    qconf = None
    if args.use_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError("4-bit QLoRA requires CUDA")
        qconf = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_storage=dtype,
        )

    kwargs: dict[str, Any] = {
        "token": token,
        "trust_remote_code": True,
        "dtype": dtype,
        "low_cpu_mem_usage": True,
        "attn_implementation": args.attn_implementation,
    }
    if qconf is not None:
        kwargs["quantization_config"] = qconf
        kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", "0"))}

    model = AutoModelForImageTextToText.from_pretrained(args.model_id, **kwargs)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.enable_input_require_grads()
    return model, processor


def normal_images(example: dict[str, Any]) -> list[Image.Image]:
    result = []
    for img in example.get("images") or []:
        if img is not None:
            if not isinstance(img, Image.Image):
                raise TypeError(f"Expected PIL image, got {type(img).__name__}")
            result.append(img.convert("RGB"))
    if not result:
        raise ValueError("Example contains no valid image")
    return result


class MultimodalDataCollator:
    """Picklable collator compatible with Python 3.14 DataLoader workers."""

    def __init__(self, processor: Any, max_length: int):
        self.processor = processor
        self.max_length = max_length
        self.tokenizer = processor.tokenizer
        self.visual_ids: set[int] = set()

        for special in [
            "<|image_pad|>",
            "<|video_pad|>",
            "<|vision_start|>",
            "<|vision_end|>",
            "<image>",
        ]:
            token_id = self.tokenizer.convert_tokens_to_ids(special)
            if (
                token_id is not None
                and token_id != self.tokenizer.unk_token_id
                and token_id >= 0
            ):
                self.visual_ids.add(int(token_id))

    def __call__(
        self, examples: list[dict[str, Any]]
    ) -> dict[str, torch.Tensor]:
        texts = [
            self.processor.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            ).strip()
            for example in examples
        ]
        images = [normal_images(example) for example in examples]

        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        labels = batch["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        for token_id in self.visual_ids:
            labels[labels == token_id] = -100
        batch["labels"] = labels
        return batch


def make_sft_config(args: argparse.Namespace, out: Path, run_name: str, token: str, dtype: torch.dtype) -> SFTConfig:
    values: dict[str, Any] = {
        "output_dir": str(out), "run_name": run_name,
        "num_train_epochs": args.num_train_epochs, "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": args.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "warmup_ratio": args.warmup_ratio, "weight_decay": args.weight_decay,
        "lr_scheduler_type": "cosine",
        "optim": "paged_adamw_8bit" if args.use_4bit else "adamw_torch_fused",
        "logging_strategy": "steps", "logging_steps": args.logging_steps,
        "eval_strategy": "steps", "eval_steps": args.eval_steps,
        "save_strategy": "steps", "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "load_best_model_at_end": True, "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "bf16": dtype == torch.bfloat16, "fp16": dtype == torch.float16,
        "tf32": torch.cuda.is_available(), "report_to": ["wandb"],
        "push_to_hub": False, "hub_model_id": args.hub_repo_id,
        "hub_private_repo": args.private_repo, "hub_token": token,
        "seed": args.seed, "data_seed": args.seed,
        "remove_unused_columns": False,
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "dataloader_num_workers": args.num_workers,
        "dataloader_pin_memory": torch.cuda.is_available(),
    }
    sig = inspect.signature(SFTConfig)
    if "max_length" in sig.parameters:
        values["max_length"] = args.max_length
    elif "max_seq_length" in sig.parameters:
        values["max_seq_length"] = args.max_length
    supported = {k: v for k, v in values.items() if k in sig.parameters}
    ignored = sorted(set(values) - set(supported))
    if ignored:
        print("Ignoring unsupported SFTConfig fields:", ", ".join(ignored))
    return SFTConfig(**supported)


def save_summary(out: Path, args: argparse.Namespace, train_metrics: dict[str, Any], eval_metrics: dict[str, Any], ntrain: int, neval: int) -> Path:
    data = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model_id, "dataset_id": args.dataset_id,
        "hub_repo_id": args.hub_repo_id,
        "train_examples": ntrain, "eval_examples": neval,
        "train_metrics": train_metrics, "eval_metrics": eval_metrics,
        "eval_perplexity": math.exp(eval_metrics["eval_loss"]) if eval_metrics.get("eval_loss", 100) < 20 else None,
        "system": {
            "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "arguments": vars(args),
    }
    path = out / "run_summary.json"
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    hf_token, wb_key = require_tokens()
    hf_login(token=hf_token, add_to_git_credential=False)
    wandb.login(key=wb_key, relogin=True)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_name = args.wandb_run_name or f"qwen3-vl-2b-llava-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=run_name, config=vars(args), job_type="multimodal-sft")

    try:
        dtype = choose_dtype()
        print("dtype:", dtype)
        print("CUDA:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))

        train_ds, eval_ds = load_data(args)
        print(f"Train examples: {len(train_ds):,}")
        print(f"Eval examples: {len(eval_ds):,}")

        model, processor = load_model_processor(args, hf_token, dtype)
        lora_args: dict[str, Any] = {
            "r": args.lora_r, "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout, "bias": "none",
            "task_type": "CAUSAL_LM", "target_modules": "all-linear",
        }
        if not args.train_vision_encoder:
            lora_args["exclude_modules"] = [r".*visual.*", r".*vision_tower.*", r".*vision_model.*"]
        peft_config = LoraConfig(**lora_args)

        trainer = SFTTrainer(
            model=model,
            args=make_sft_config(args, out, run_name, hf_token, dtype),
            data_collator=MultimodalDataCollator(
                processor=processor,
                max_length=args.max_length,
            ),
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=processor,
            peft_config=peft_config,
        )

        train_result = trainer.train()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

        eval_metrics = trainer.evaluate()
        if eval_metrics.get("eval_loss", 100) < 20:
            eval_metrics["eval_perplexity"] = math.exp(eval_metrics["eval_loss"])
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

        trainer.save_model(str(out))
        processor.save_pretrained(str(out))
        summary = save_summary(out, args, train_result.metrics, eval_metrics, len(train_ds), len(eval_ds))
        wandb.save(str(summary), base_path=str(out))

        api = HfApi(token=hf_token)
        api.create_repo(args.hub_repo_id, repo_type="model", private=args.private_repo, exist_ok=True)
        trainer.push_to_hub(commit_message=f"Multimodal SFT of {args.model_id} on {args.dataset_id}")
        print(f"Adapter pushed to https://huggingface.co/{args.hub_repo_id}")
        print("W&B:", wandb.run.url if wandb.run else "available in project")
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
