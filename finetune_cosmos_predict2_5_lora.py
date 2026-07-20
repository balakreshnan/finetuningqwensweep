#!/usr/bin/env python3
"""Fine-tune NVIDIA Cosmos-Predict2.5-2B with the official Video2World LoRA recipe.

This is an orchestration script intended to be run from the root of the official
nvidia-cosmos/cosmos-predict2.5 repository. It performs these stages:

1. Validate the Cosmos repository and GPU environment.
2. Authenticate with Hugging Face and Weights & Biases using environment tokens.
3. Download the public nvidia/Cosmos-NeMo-Assets dataset.
4. Create prompt metadata with the repository preprocessing utility.
5. Launch the official Cosmos-Predict2.5-2B Video2World LoRA training recipe.
6. Convert the latest distributed checkpoint to consolidated PyTorch files.
7. Export W&B run history to CSV when a W&B run path is supplied.
8. Upload the consolidated checkpoint and run metadata to Hugging Face Hub.

Required environment variables:
    HF_TOKEN       Hugging Face token. Read access is needed to download gated
                   Cosmos weights; write access is needed when uploading.
    WANDB_API_KEY  Weights & Biases API key.

Typical use, from the Cosmos-Predict2.5 repository root:
    python /path/to/finetune_cosmos_predict2_5_lora.py \
      --hf-repo-id YOUR_USERNAME/cosmos-predict2.5-2b-nemo-lora \
      --wandb-project cosmos-predict25-lora \
      --nproc-per-node 8

The official default experiment is:
    predict2_video2world_lora_training_2b_cosmos_nemo_assets

Notes:
- This script does not reimplement NVIDIA's model or training loop.
- Cosmos-Predict2.5 is a large video world model. Multi-GPU data-center hardware
  is strongly recommended even when using LoRA.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

LOGGER = logging.getLogger("cosmos-lora")

DEFAULT_EXPERIMENT = "predict2_video2world_lora_training_2b_cosmos_nemo_assets"
DEFAULT_CONFIG = "cosmos_predict2/_src/predict2/configs/video2world/config.py"
DEFAULT_DATASET_REPO = "nvidia/Cosmos-NeMo-Assets"
DEFAULT_PROMPT = "A video of sks teal robot."
DEFAULT_PROJECT = "cosmos_predict_v2p5"
DEFAULT_GROUP = "video2world_lora"
DEFAULT_NAME = "2b_cosmos_nemo_assets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run official Cosmos-Predict2.5-2B Video2World LoRA post-training.",
    )
    parser.add_argument(
        "--cosmos-root",
        type=Path,
        default=Path.cwd(),
        help="Root of the cloned nvidia-cosmos/cosmos-predict2.5 repository.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("datasets/cosmos_nemo_assets"),
        help="Dataset path relative to --cosmos-root unless absolute.",
    )
    parser.add_argument("--dataset-repo", default=DEFAULT_DATASET_REPO)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--master-port", type=int, default=12341)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("./cosmos_outputs"),
        help="Value assigned to IMAGINAIRE_OUTPUT_ROOT.",
    )
    parser.add_argument("--wandb-project", default="cosmos-predict25-lora")
    parser.add_argument(
        "--wandb-entity",
        default=None,
        help="Optional W&B entity/team. Passed through WANDB_ENTITY.",
    )
    parser.add_argument(
        "--wandb-run-path",
        default=None,
        help="Optional entity/project/run_id used to export W&B history to CSV.",
    )
    parser.add_argument(
        "--hf-repo-id",
        default=None,
        help="Destination Hugging Face model repository, e.g. user/cosmos-lora.",
    )
    parser.add_argument(
        "--private-repo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create the Hugging Face destination as a private repository.",
    )
    parser.add_argument(
        "--checkpoint-project",
        default=DEFAULT_PROJECT,
        help="Project segment used by the official experiment output path.",
    )
    parser.add_argument("--checkpoint-group", default=DEFAULT_GROUP)
    parser.add_argument("--checkpoint-name", default=DEFAULT_NAME)
    parser.add_argument(
        "--extra-override",
        action="append",
        default=[],
        help="Additional Cosmos config override. Repeat this option as needed.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not download Cosmos-NeMo-Assets.",
    )
    parser.add_argument(
        "--skip-prompts",
        action="store_true",
        help="Do not generate metadata prompt files.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training and operate on an existing checkpoint.",
    )
    parser.add_argument(
        "--skip-conversion",
        action="store_true",
        help="Do not convert the latest DCP checkpoint to .pt files.",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Do not upload artifacts to Hugging Face Hub.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args()


def resolve_under(root: Path, value: Path) -> Path:
    return value.expanduser().resolve() if value.is_absolute() else (root / value).resolve()


def run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    dry_run: bool,
) -> None:
    rendered = " ".join(shlex.quote(part) for part in command)
    LOGGER.info("Running: %s", rendered)
    if dry_run:
        return
    subprocess.run(list(command), cwd=str(cwd), env=dict(env), check=True)


def require_command(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required command is not available on PATH: {name}")


def validate_repository(root: Path, config: str) -> None:
    required = [
        root / "scripts" / "train.py",
        root / "scripts" / "convert_distcp_to_pt.py",
        root / "scripts" / "create_prompts_for_nemo_assets.py",
        root / config,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        formatted = "\n  - ".join(missing)
        raise RuntimeError(
            "The selected directory does not look like a compatible "
            f"Cosmos-Predict2.5 repository. Missing:\n  - {formatted}"
        )


def validate_credentials(args: argparse.Namespace) -> None:
    if not os.getenv("HF_TOKEN"):
        raise RuntimeError(
            "HF_TOKEN is not set. The Cosmos checkpoint is gated and requires "
            "license acceptance plus Hugging Face authentication."
        )
    if not args.skip_training and not os.getenv("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY is not set.")
    if not args.skip_upload and not args.hf_repo_id:
        raise RuntimeError("--hf-repo-id is required unless --skip-upload is used.")


def gpu_count() -> int:
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        return 0


def prepare_environment(args: argparse.Namespace, output_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["IMAGINAIRE_OUTPUT_ROOT"] = str(output_root)
    env["WANDB_PROJECT"] = args.wandb_project
    env.setdefault("WANDB_MODE", "online")
    if args.wandb_entity:
        env["WANDB_ENTITY"] = args.wandb_entity
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def download_dataset(
    args: argparse.Namespace,
    *,
    root: Path,
    dataset_dir: Path,
    env: Mapping[str, str],
) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "hf",
        "download",
        args.dataset_repo,
        "--repo-type",
        "dataset",
        "--local-dir",
        str(dataset_dir),
        "--include",
        "*.mp4*",
        "--token",
        env["HF_TOKEN"],
    ]
    run_command(command, cwd=root, env=env, dry_run=args.dry_run)

    source = dataset_dir / "nemo_diffusion_example_data"
    destination = dataset_dir / "videos"
    if args.dry_run:
        LOGGER.info("Would normalize video directory: %s -> %s", source, destination)
        return
    if source.exists() and not destination.exists():
        source.rename(destination)
    if not destination.exists():
        mp4_files = list(dataset_dir.rglob("*.mp4"))
        if not mp4_files:
            raise RuntimeError(f"No MP4 files were found under {dataset_dir}")
        destination.mkdir(parents=True, exist_ok=True)
        for video in mp4_files:
            if video.parent != destination:
                target = destination / video.name
                if target.exists():
                    raise RuntimeError(f"Duplicate dataset filename: {video.name}")
                shutil.move(str(video), str(target))


def create_prompts(
    args: argparse.Namespace,
    *,
    root: Path,
    dataset_dir: Path,
    env: Mapping[str, str],
) -> None:
    command = [
        sys.executable,
        "-m",
        "scripts.create_prompts_for_nemo_assets",
        "--dataset_path",
        str(dataset_dir),
        "--prompt",
        args.prompt,
    ]
    run_command(command, cwd=root, env=env, dry_run=args.dry_run)


def train(
    args: argparse.Namespace,
    *,
    root: Path,
    env: Mapping[str, str],
) -> None:
    command = [
        "torchrun",
        f"--nproc_per_node={args.nproc_per_node}",
        f"--master_port={args.master_port}",
        "scripts/train.py",
        f"--config={args.config}",
        "--",
        f"experiment={args.experiment}",
        *args.extra_override,
    ]
    run_command(command, cwd=root, env=env, dry_run=args.dry_run)


def checkpoint_root(args: argparse.Namespace, output_root: Path) -> Path:
    return (
        output_root
        / args.checkpoint_project
        / args.checkpoint_group
        / args.checkpoint_name
        / "checkpoints"
    )


def read_latest_checkpoint(checkpoints: Path) -> Path:
    latest_file = checkpoints / "latest_checkpoint.txt"
    if not latest_file.exists():
        raise RuntimeError(f"Missing latest checkpoint marker: {latest_file}")
    latest_value = latest_file.read_text(encoding="utf-8").strip()
    if not latest_value:
        raise RuntimeError(f"Latest checkpoint marker is empty: {latest_file}")
    latest = Path(latest_value)
    checkpoint = latest if latest.is_absolute() else checkpoints / latest
    if not checkpoint.exists():
        raise RuntimeError(f"Latest checkpoint directory does not exist: {checkpoint}")
    return checkpoint.resolve()


def convert_checkpoint(
    args: argparse.Namespace,
    *,
    root: Path,
    checkpoint: Path,
    env: Mapping[str, str],
) -> None:
    model_dir = checkpoint / "model"
    if not model_dir.exists():
        raise RuntimeError(f"DCP model directory does not exist: {model_dir}")
    command = [
        sys.executable,
        "scripts/convert_distcp_to_pt.py",
        str(model_dir),
        str(checkpoint),
    ]
    run_command(command, cwd=root, env=env, dry_run=args.dry_run)


def export_wandb_history(run_path: str, destination: Path) -> Path:
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("Install wandb to export run history.") from exc

    LOGGER.info("Exporting W&B history from %s", run_path)
    run = wandb.Api().run(run_path)
    rows = list(run.scan_history())
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        destination.write_text("", encoding="utf-8")
        return destination

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return destination


def write_run_manifest(
    args: argparse.Namespace,
    *,
    destination: Path,
    root: Path,
    dataset_dir: Path,
    output_root: Path,
    checkpoint: Path | None,
) -> Path:
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cosmos_root": str(root),
        "dataset_repo": args.dataset_repo,
        "dataset_dir": str(dataset_dir),
        "prompt": args.prompt,
        "experiment": args.experiment,
        "config": args.config,
        "nproc_per_node": args.nproc_per_node,
        "output_root": str(output_root),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "wandb_project": args.wandb_project,
        "wandb_run_path": args.wandb_run_path,
        "hf_repo_id": args.hf_repo_id,
        "extra_overrides": args.extra_override,
    }
    destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return destination


def iter_upload_files(checkpoint: Path, extras: Iterable[Path]) -> list[Path]:
    files = [
        checkpoint / "model.pt",
        checkpoint / "model_ema_fp32.pt",
        checkpoint / "model_ema_bf16.pt",
        *extras,
    ]
    return [path for path in files if path.exists()]


def upload_to_hub(
    args: argparse.Namespace,
    *,
    files: Sequence[Path],
    token: str,
) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub to upload artifacts.") from exc

    if not files:
        raise RuntimeError("No files are available for upload.")

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.hf_repo_id,
        repo_type="model",
        private=args.private_repo,
        exist_ok=True,
    )
    for path in files:
        LOGGER.info("Uploading %s", path)
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=args.hf_repo_id,
            repo_type="model",
            commit_message=f"Upload {path.name} from Cosmos-Predict2.5 LoRA run",
        )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    root = args.cosmos_root.expanduser().resolve()
    dataset_dir = resolve_under(root, args.dataset_dir)
    output_root = resolve_under(root, args.output_root)

    validate_repository(root, args.config)
    validate_credentials(args)
    require_command("torchrun")
    if not args.skip_download:
        require_command("hf")

    visible_gpus = gpu_count()
    LOGGER.info("Cosmos repository: %s", root)
    LOGGER.info("Detected CUDA devices: %d", visible_gpus)
    if not args.skip_training and visible_gpus and args.nproc_per_node > visible_gpus:
        raise RuntimeError(
            f"--nproc-per-node={args.nproc_per_node}, but only {visible_gpus} CUDA "
            "device(s) are visible."
        )
    if not args.skip_training and visible_gpus == 0:
        LOGGER.warning("No CUDA devices were detected by PyTorch.")

    output_root.mkdir(parents=True, exist_ok=True)
    env = prepare_environment(args, output_root)

    if not args.skip_download:
        download_dataset(args, root=root, dataset_dir=dataset_dir, env=env)
    if not args.skip_prompts:
        create_prompts(args, root=root, dataset_dir=dataset_dir, env=env)
    if not args.skip_training:
        train(args, root=root, env=env)

    checkpoint: Path | None = None
    checkpoints = checkpoint_root(args, output_root)
    if not args.dry_run:
        checkpoint = read_latest_checkpoint(checkpoints)
        LOGGER.info("Latest checkpoint: %s", checkpoint)

    if not args.skip_conversion:
        if args.dry_run:
            LOGGER.info("Would convert latest DCP checkpoint under %s", checkpoints)
        elif checkpoint is not None:
            convert_checkpoint(args, root=root, checkpoint=checkpoint, env=env)

    metadata_dir = output_root / "run_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = write_run_manifest(
        args,
        destination=metadata_dir / "run_manifest.json",
        root=root,
        dataset_dir=dataset_dir,
        output_root=output_root,
        checkpoint=checkpoint,
    )

    extra_files: list[Path] = [manifest_path]
    if args.wandb_run_path and not args.dry_run:
        metrics_path = export_wandb_history(
            args.wandb_run_path,
            metadata_dir / "wandb_metrics.csv",
        )
        extra_files.append(metrics_path)

    if not args.skip_upload:
        if args.dry_run:
            LOGGER.info("Would upload consolidated checkpoint to %s", args.hf_repo_id)
        elif checkpoint is not None:
            files = iter_upload_files(checkpoint, extra_files)
            upload_to_hub(args, files=files, token=env["HF_TOKEN"])
            LOGGER.info("Hugging Face repository: https://huggingface.co/%s", args.hf_repo_id)

    LOGGER.info("Cosmos LoRA workflow completed successfully.")


if __name__ == "__main__":
    main()
