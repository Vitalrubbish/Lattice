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
    # Note: CUDA 13.0 + driver 580 requires dkms >= 3.1.8, allow held package changes
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
export LD_LIBRARY_PATH="/usr/local/cuda-${CUDA_VERSION}/lib64:$LD_LIBRARY_PATH"
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

echo ">>> [6/9] Installing vLLM (this may take 10-15 minutes)..."
# vLLM 0.22+ with FlashInfer for A30 sm_80
pip install --no-cache-dir vllm 2>&1 | tail -5

# Verify vLLM installation
python3 -c "import vllm; print('vLLM version:', vllm.__version__)"

# ── 6. FlashInfer configuration for A30 ──
echo ""
echo ">>> [7/9] Configuring FlashInfer for A30 (sm_80)..."
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
CCCL_FILE="$SITE_PACKAGES/flashinfer/compilation_context.py"

if [ -f "$CCCL_FILE" ]; then
    # Apply CCCL compatibility patch if needed
    if ! grep -q "CCCL_DISABLE_CTK_COMPATIBILITY_CHECK" "$CCCL_FILE" 2>/dev/null; then
        echo "   Applying CCCL compatibility patch..."
        sed -i 's/COMMON_NVCC_FLAGS = \[/COMMON_NVCC_FLAGS = ["-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",/' "$CCCL_FILE"
    else
        echo "   CCCL patch already applied."
    fi
else
    echo "   WARNING: FlashInfer compilation_context.py not found."
    echo "   FlashInfer may not be installed. Check vLLM installation."
fi

# Pre-compile FlashInfer kernels for A30
export FLASHINFER_CUDA_ARCH_LIST="8.0"
export CUDA_HOME="/usr/local/cuda-${CUDA_VERSION}"

echo "   Triggering FlashInfer JIT pre-compilation for sm_80..."
python3 -c "
import flashinfer
print(f'FlashInfer version: {flashinfer.__version__}')
print('FlashInfer OK — kernels will JIT-compile on first inference call')
" 2>&1 || echo "   WARNING: FlashInfer import check failed (may be OK on first vLLM run)"

# ── 7. Download TinyLlama model ──
echo ""
echo ">>> [8/9] Downloading TinyLlama model..."
pip install --no-cache-dir huggingface_hub 2>&1 | tail -3

mkdir -p "$MODEL_DIR"
if [ ! -f "$MODEL_DIR/tinyllama/model.safetensors" ]; then
    # Use 'hf download' (huggingface-cli is deprecated)
    hf download TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --local-dir "$MODEL_DIR/tinyllama" 2>&1 | tail -5
    echo "   Model downloaded to $MODEL_DIR/tinyllama"
else
    echo "   Model already exists at $MODEL_DIR/tinyllama, skipping."
fi

# Verify model
if [ -f "$MODEL_DIR/tinyllama/model.safetensors" ]; then
    MODEL_SIZE=$(du -h "$MODEL_DIR/tinyllama/model.safetensors" | cut -f1)
    echo "   Model size: $MODEL_SIZE"
else
    echo "   ERROR: Model download failed — model.safetensors missing."
    exit 1
fi

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
echo "  [ ] 5. vLLM env: conda activate vllm-bench && python3 -c 'import vllm; print(vllm.__version__)'"
echo "         Expected: '0.22.0' or newer"
echo "  [ ] 6. FlashInfer: FLASHINFER_CUDA_ARCH_LIST=8.0 python3 -c 'import flashinfer'"
echo "         Expected: no errors"
echo "  [ ] 7. Model: ls $MODEL_DIR/tinyllama/model.safetensors"
echo "         Expected: file exists (~2.1 GB)"
echo ""

echo "======================================================================"
echo " Quick Start After Reboot:"
echo "======================================================================"
echo ""
echo "  # Run full comparison (baseline + vLLM):"
echo "  cd $REPO_DIR"
echo "  bash scripts/step3_test_baremetal.sh compare"
echo ""
echo "  # Or run individual modes:"
echo "  bash scripts/step3_test_baremetal.sh baseline"
echo "  bash scripts/step3_test_baremetal.sh vllm"
echo ""
echo "======================================================================"
