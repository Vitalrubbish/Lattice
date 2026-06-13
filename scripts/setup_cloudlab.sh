#!/usr/bin/env bash
# ==============================================================================
# Setup Script for CloudLab A30 / Ubuntu 22.04
# Purpose: Prepare bare-metal A30 server for LLM OS benchmarks (Step 3)
#
# Installs:
#   - System packages, kernel headers, bpftrace
#   - CUDA 13.0 + NVIDIA driver 580
#   - Rust 1.78.0
#   - Miniconda3 + vLLM benchmark environment
#   - FlashInfer (pre-compiled for A30 sm_80)
#   - TinyLlama model
#   - Latttice project (this repo)
#
# Usage (as root or with sudo on CloudLab A30):
#   sudo bash setup_cloudlab.sh
#
# After setup:
#   1. reboot (or reload modules: rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia && modprobe nvidia nvidia_modeset nvidia_drm nvidia_uvm)
#   2. Verify: nvidia-smi
#   3. Run benchmark: bash scripts/step3_test_baremetal.sh compare
# ==============================================================================
set -euo pipefail

# ── Configuration ──
# Detect actual home directory (works for both /root and /users/username)
if [ "$(id -u)" -eq 0 ]; then
    REAL_HOME="${HOME:-/root}"
else
    REAL_HOME="${HOME:-/home/$USER}"
fi
MODEL_DIR="${MODEL_DIR:-$REAL_HOME/models}"
REPO_DIR="${REPO_DIR:-$REAL_HOME/llm-rust-ebpf}"
REPO_URL="${REPO_URL:-https://github.com/Vitalrubbish/Lattice.git}"
RUST_VERSION="${RUST_VERSION:-stable}"  # 1.78.0 too old for edition2024 crates
CUDA_VERSION="${CUDA_VERSION:-13.0}"
NV_DRIVER_VERSION="${NV_DRIVER_VERSION:-580}"

echo "======================================================================"
echo " Latttice CloudLab Setup — A30 / Ubuntu 22.04"
echo "======================================================================"
echo " Home dir:     $REAL_HOME"
echo " Model dir:    $MODEL_DIR"
echo " Repo dir:     $REPO_DIR"
echo " CUDA:         $CUDA_VERSION"
echo " Driver:       $NV_DRIVER_VERSION"
echo " Rust:         $RUST_VERSION"
echo "======================================================================"

# ── 1. System packages ──
echo ""
echo ">>> [1/9] Installing system dependencies..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    git \
    gnupg \
    htop \
    iotop \
    jq \
    pkg-config \
    python3 \
    python3-pip \
    software-properties-common \
    unzip \
    wget \
    libelf-dev \
    clang \
    llvm \
    libssl-dev \
    linux-headers-$(uname -r) \
    linux-tools-common \
    linux-tools-$(uname -r) \
    bpftrace

# Verify bpftrace
if command -v bpftrace &>/dev/null; then
    echo "   bpftrace: $(bpftrace --version 2>&1 | head -1)"
else
    echo "   WARNING: bpftrace not found — eBPF tracing won't work"
fi

# ── 2. CUDA + NVIDIA driver ──
echo ""
echo ">>> [2/9] Installing CUDA $CUDA_VERSION + NVIDIA driver $NV_DRIVER_VERSION..."
if ! command -v nvcc &>/dev/null; then
    # Remove any existing keyring file to avoid permission issues
    rm -f /tmp/cuda-keyring.deb
    wget -qO /tmp/cuda-keyring.deb \
        "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
    dpkg -i /tmp/cuda-keyring.deb
    apt-get update -y
    # CUDA 13.0 + driver 580 requires dkms >= 3.1.8; upgrade from CUDA repo first
    echo "   Upgrading dkms from CUDA repo..."
    apt-get install -y --no-install-recommends --allow-change-held-packages dkms
    apt-get install -y --no-install-recommends --allow-change-held-packages \
        "cuda-toolkit-${CUDA_VERSION/./-}" \
        "nvidia-driver-${NV_DRIVER_VERSION}" \
        "cuda-drivers-${NV_DRIVER_VERSION%%.*}"
    rm -f /tmp/cuda-keyring.deb
else
    echo "   CUDA already installed: $(nvcc --version 2>/dev/null | grep 'release' || echo 'unknown')"
fi

# Set environment variables
cuda_profile_line="# CUDA ${CUDA_VERSION}"
if ! grep -q "cuda-${CUDA_VERSION}" /etc/profile 2>/dev/null; then
    cat >> /etc/profile << EOF

${cuda_profile_line}
export PATH=/usr/local/cuda-${CUDA_VERSION}/bin:\$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-${CUDA_VERSION}/lib64:\$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-${CUDA_VERSION}
EOF
fi
export PATH="/usr/local/cuda-${CUDA_VERSION}/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-${CUDA_VERSION}/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_HOME="/usr/local/cuda-${CUDA_VERSION}"

# ── 3. Rust ──
echo ""
echo ">>> [3/9] Installing Rust $RUST_VERSION..."
if ! command -v cargo &>/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
        sh -s -- -y --default-toolchain "$RUST_VERSION"
    source "$HOME/.cargo/env"
else
    echo "   Rust already installed: $(rustc --version)"
fi
source "$HOME/.cargo/env" 2>/dev/null || true
rustup component add clippy rustfmt 2>/dev/null || true

# ── 4. Miniconda3 ──
echo ""
echo ">>> [4/9] Installing Miniconda3..."
if [ ! -d "/opt/miniconda3" ]; then
    wget -qO /tmp/miniconda.sh \
        "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    bash /tmp/miniconda.sh -b -p /opt/miniconda3
    rm -f /tmp/miniconda.sh
    # Fix envs directory permissions for non-root users
    if [ "$(id -u)" -eq 0 ] && [ -n "${SUDO_USER:-}" ]; then
        chown -R "$SUDO_USER:$(id -gn "$SUDO_USER")" /opt/miniconda3/envs 2>/dev/null || true
        chown -R "$SUDO_USER:$(id -gn "$SUDO_USER")" "$HOME/.conda" 2>/dev/null || true
    fi
fi
eval "$(/opt/miniconda3/bin/conda shell.bash hook)"

# Accept conda Terms of Service (needed for newer conda versions)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

conda init bash 2>/dev/null || true

# ── 5. vLLM benchmark environment ──
echo ""
echo ">>> [5/9] Creating vLLM benchmark environment..."
eval "$(/opt/miniconda3/bin/conda shell.bash hook)"

if ! conda env list 2>/dev/null | grep -q "^vllm-bench "; then
    conda create -n vllm-bench python=3.12 -y
fi
conda activate vllm-bench

# SKIPPED: vLLM, FlashInfer, and TinyLlama installation (Step 6-8)
echo ""
echo ">>> [6/9] SKIPPED: vLLM installation"
echo ">>> [7/9] SKIPPED: FlashInfer configuration"
echo ">>> [8/9] SKIPPED: TinyLlama model download"

# ── Verification ──
echo ""
echo "======================================================================"
echo " SETUP COMPLETE — Verification Checklist"
echo "======================================================================"

echo ""
echo "  [ ] 1. REBOOT or reload NVIDIA modules:"
echo "         sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia"
echo "         sudo modprobe nvidia nvidia_modeset nvidia_drm nvidia_uvm"
echo "  [ ] 2. Verify GPU: nvidia-smi"
echo "         Expected: 'NVIDIA A30' with ~24GB VRAM"
echo "  [ ] 3. CUDA: nvcc --version"
echo "         Expected: 'release ${CUDA_VERSION}'"
echo "  [ ] 4. Rust: cargo --version"
echo "         Expected: 'cargo 1.78.0'"
echo "  [ ] 5. Conda env: conda activate vllm-bench"
echo "         (vLLM/FlashInfer/TinyLlama installation skipped per user request)"
echo ""

echo "======================================================================"
echo " Quick Start After Reboot:"
echo "======================================================================"
echo ""
echo "  # To install vLLM later:"
echo "  conda activate vllm-bench"
echo "  pip install vllm"
echo ""
echo "======================================================================"
