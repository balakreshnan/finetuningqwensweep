#!/usr/bin/env bash
#
# GH200 / NVIDIA AI Preflight Check
#
# Validates:
#   - OS
#   - NVIDIA GPU
#   - NVIDIA Driver
#   - GH200 memory hotplug
#   - CUDA Toolkit
#   - nvcc
#   - CUDA_HOME
#   - Compiler
#   - Disk
#   - RAM
#
# Safe to run multiple times.
#

set -euo pipefail

GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[1;33m"
BLUE="\033[0;34m"
NC="\033[0m"

ok() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

fail() {
    echo -e "${RED}[FAIL]${NC} $1"
}

section() {
    echo
    echo -e "${BLUE}=========================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}=========================================================${NC}"
}

section "Operating System"

cat /etc/os-release | grep PRETTY_NAME || true
uname -r
uname -m

section "CPU"

lscpu | grep "Model name" || true
echo "CPU cores: $(nproc)"

section "Memory"

free -h

section "Disk"

df -h /

section "NVIDIA PCI Devices"

if lspci | grep -qi nvidia; then
    ok "NVIDIA device detected"
    lspci | grep -i nvidia
else
    fail "No NVIDIA GPU detected"
    exit 1
fi

section "NVIDIA Driver"

if command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi >/dev/null 2>&1; then
        ok "nvidia-smi working"
        nvidia-smi
    else
        warn "nvidia-smi installed but GPU not initialized"
    fi
else
    fail "nvidia-smi not installed"
fi

section "Kernel Module"

if [ -f /proc/driver/nvidia/version ]; then
    cat /proc/driver/nvidia/version
    ok "Kernel module loaded"
else
    fail "NVIDIA kernel module not loaded"
fi

echo
modinfo nvidia 2>/dev/null | grep -E "^license:|^version:" || true

section "GH200 Memory Hotplug"

if [ -f /sys/devices/system/memory/auto_online_blocks ]; then

    MODE=$(cat /sys/devices/system/memory/auto_online_blocks)

    echo "Current mode: $MODE"

    if [ "$MODE" != "online_movable" ]; then

        warn "GH200 requires online_movable"

        read -p "Automatically enable it? (y/N): " ANS

        if [[ "$ANS" =~ ^[Yy]$ ]]; then

            echo online_movable | sudo tee /sys/devices/system/memory/auto_online_blocks

            sudo modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia || true
            sudo modprobe nvidia
            sudo modprobe nvidia_uvm

            ok "online_movable enabled"

        fi
    else
        ok "online_movable already configured"
    fi
fi

section "CUDA Toolkit"

if command -v nvcc >/dev/null 2>&1; then

    ok "nvcc found"

    which nvcc
    nvcc --version

else

    fail "nvcc NOT installed"

fi

section "CUDA_HOME"

echo "CUDA_HOME=${CUDA_HOME:-NOT SET}"

if [ -n "${CUDA_HOME:-}" ]; then

    if [ -x "$CUDA_HOME/bin/nvcc" ]; then
        ok "CUDA_HOME is valid"
    else
        warn "CUDA_HOME does not contain nvcc"
    fi

else

    warn "CUDA_HOME not set"

fi

section "Compiler"

gcc --version | head -1 || true
g++ --version | head -1 || true

section "Summary"

echo

echo "Checklist"

echo "---------"

if command -v nvidia-smi >/dev/null; then
    echo "✔ NVIDIA Driver"
else
    echo "✘ NVIDIA Driver"
fi

if command -v nvcc >/dev/null; then
    echo "✔ CUDA Toolkit"
else
    echo "✘ CUDA Toolkit"
fi

if [ -f /proc/driver/nvidia/version ]; then
    echo "✔ Kernel Module"
else
    echo "✘ Kernel Module"
fi

if [ -f /sys/devices/system/memory/auto_online_blocks ]; then
    echo "GH200 Memory Mode: $(cat /sys/devices/system/memory/auto_online_blocks)"
fi

echo
echo "Preflight complete."