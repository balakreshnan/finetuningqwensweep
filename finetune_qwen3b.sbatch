#!/bin/bash
#==============================================================================
# SLURM Job: Fine-tune Qwen2.5-3B-Instruct with LoRA on DLCluster
#
# Model:   Qwen/Qwen2.5-3B-Instruct (from HuggingFace)
# Dataset: OpenAssistant/oasst1 (multi-turn conversation dataset)
# Method:  QLoRA (4-bit quantization + LoRA adapters)
#
# Files:
#   finetune_qwen3b.sbatch  - This SLURM wrapper script
#   finetune_qwen3b.py      - Python fine-tuning code
#
# Submit:
#   sbatch --export=HF_TOKEN=hf_xxx finetune_qwen3b.sbatch
#
#==============================================================================

#SBATCH --job-name=qwen3b-finetune
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=qwen3b_finetune-%j.out
#SBATCH --error=qwen3b_finetune-%j.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

set -euo pipefail

echo "=============================================="
echo "  Qwen2.5-3B Fine-Tuning Job"
echo "=============================================="
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Node         : $(hostname)"
echo "Date         : $(date)"
nvidia-smi
echo "=============================================="

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------
OUTPUT_DIR="/home/${USER}/finetune_output/qwen3b-oasst1-lora"

# Use scratch if available
SCRATCH_DIR="/home/scratch.${USER}"
if [ -d "$SCRATCH_DIR" ]; then
    export HF_HOME="${SCRATCH_DIR}/.cache/huggingface"
    OUTPUT_DIR="${SCRATCH_DIR}/finetune_output/qwen3b-oasst1-lora"
fi

mkdir -p "$OUTPUT_DIR"
mkdir -p "${HF_HOME:-$HOME/.cache/huggingface}"

#------------------------------------------------------------------------------
# Install Dependencies
#------------------------------------------------------------------------------
echo ""
echo ">>> Installing dependencies..."
pip install --quiet --upgrade \
    torch \
    "transformers>=4.43.0" \
    datasets \
    accelerate \
    peft \
    trl \
    bitsandbytes \
    scipy \
    huggingface_hub

# Flash Attention (optional — fails gracefully)
pip install --quiet flash-attn --no-build-isolation 2>/dev/null || echo "Flash Attention not installed (optional)"

echo ">>> Dependencies ready."

#------------------------------------------------------------------------------
# Validate Token
#------------------------------------------------------------------------------
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set!"
    echo "Submit with: sbatch --export=HF_TOKEN=hf_xxx finetune_qwen3b.sbatch"
    exit 1
fi

#------------------------------------------------------------------------------
# Run Fine-Tuning Python Script
#------------------------------------------------------------------------------
echo ""
echo ">>> Launching fine-tuning..."

python3 finetune_qwen3b.py \
    --model_name "Qwen/Qwen2.5-3B-Instruct" \
    --dataset_name "OpenAssistant/oasst1" \
    --output_dir "${OUTPUT_DIR}" \
    --hf_token "${HF_TOKEN}" \
    --num_epochs 3 \
    --batch_size 4 \
    --gradient_accumulation 4 \
    --learning_rate 2e-4 \
    --max_seq_length 2048 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05

echo ""
echo "=============================================="
echo "  SLURM Job Complete!"
echo "  Output: ${OUTPUT_DIR}"
echo "  End time: $(date)"
echo "=============================================="
