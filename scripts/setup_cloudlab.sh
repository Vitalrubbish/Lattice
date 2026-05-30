#!/usr/bin/env bash
# ==============================================================================
# Setup Script for CloudLab A30 / Ubuntu 22.04 (Root)
# Purpose: Prepare bare-metal A30 server for LLM OS benchmarks (Step 3)
#
# Installs:
#   - System packages, kernel headers, bpftrace
#   - CUDA 12.2 + NVIDIA driver 535
#   - Rust 1.78.0
#   - Miniconda3 + vLLM benchmark environment
#   - FlashInfer (pre-compiled for A30 sm_80)
#   - TinyLlama model
#   - Latttice project (this repo)
#
# Usage (as root on CloudLab A30):
#   bash setup_cloudlab.sh
#
# After setup:
#   1. reboot
#   2. Verify: nvidia-smi
#   3. Run benchmark: bash scripts/step3_test_baremetal.sh compare
# ==============================================================================
set -euo pipefail

# ── Configuration ──
MODEL_DIR="${MODEL_DIR:-/root/models}"
REPO_DIR="${REPO_DIR:-/root/llm-rust-ebpf}"
REPO_URL="${REPO_URL:-https://github.com/Vitalrubbish/Lattice.git}"
RUST_VERSION="${RUST_VERSION:-1.78.0}"
CUDA_VERSION="${CUDA_VERSION:-12.2}"
NV_DRIVER_VERSION="${NV_DRIVER_VERSION:-535-server}"

echo "======================================================================"
echo " Latttice CloudLab Setup — A30 / Ubuntu 22.04"
echo "======================================================================"
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

# ── 2. CUDA 12.2 + NVIDIA driver ──
echo ""
echo ">>> [2/9] Installing CUDA $CUDA_VERSION + NVIDIA driver $NV_DRIVER_VERSION..."
if ! command -v nvcc &>/dev/null; then
    wget -qO /tmp/cuda-keyring.deb \
        "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
    dpkg -i /tmp/cuda-keyring.deb
    apt-get update -y
    apt-get install -y --no-install-recommends \
        "cuda-toolkit-${CUDA_VERSION/./-}" \
        "nvidia-driver-${NV_DRIVER_VERSION}" \
        "cuda-drivers-${NV_DRIVER_VERSION%%.*}"
    rm -f /tmp/cuda-keyring.deb
else
    echo "   CUDA already installed: $(nvcc --version 2>/dev/null | grep 'release' || echo 'unknown')"
fi

# Set environment variables
if ! grep -q "cuda-${CUDA_VERSION}" /etc/profile 2>/dev/null; then
    cat >> /etc/profile << EOF

# CUDA ${CUDA_VERSION}
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
fi
eval "$(/opt/miniconda3/bin/conda shell.bash hook)"
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
    huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --local-dir "$MODEL_DIR/tinyllama" \
        --exclude "*.bin" "*.pt" "*.msgpack" "*.h5" 2>&1 | tail -5
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

# ── 8. Clone and build project ──
echo ""
echo ">>> [9/9] Cloning and building Latttice..."
if [ ! -d "$REPO_DIR" ]; then
    git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"
echo "   Building release binaries..."
cargo build --release 2>&1 | tail -5
cargo build --release --example bench_throughput 2>&1 | tail -3
echo "   Build complete."

# ── Verification ──
echo ""
echo "======================================================================"
echo " SETUP COMPLETE — Verification Checklist"
echo "======================================================================"

echo ""
echo "  [ ] 1. REBOOT REQUIRED: run 'reboot' to activate NVIDIA drivers"
echo "  [ ] 2. After reboot: nvidia-smi"
echo "         Expected: 'NVIDIA A30' with ~24GB VRAM"
echo "  [ ] 3. CUDA: nvcc --version"
echo "         Expected: 'release ${CUDA_VERSION}'"
echo "  [ ] 4. Rust: cargo --version"
echo "         Expected: 'cargo 1.78.0'"
echo "  [ ] 5. vLLM env: conda activate vllm-bench && python3 -c 'import vllm; print(vllm.__version__)'"
echo "         Expected: '0.22.0' or newer"
echo "  [ ] 6. FlashInfer: FLASHINFER_CUDA_ARCH_LIST=8.0 python3 -c 'import flashinfer'"
echo "         Expected: no errors"
echo "  [ ] 7. Model: ls /root/models/tinyllama/model.safetensors"
echo "         Expected: file exists (~2.2 GB)"
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
