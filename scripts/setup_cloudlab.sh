#!/usr/bin/env bash

# ==============================================================================
# Setup Script for CloudLab A30 / Ubuntu 22.04 (Root Version)
# Purpose: Support LLM I/O benchmarks with Rust and eBPF
# ==============================================================================

set -euo pipefail

# Configuration
MODEL_DIR="/root/models"
REPO_DIR="/root/llm-rust-ebpf"
REPO_URL="https://github.com/Vitalrubbish/Lattice.git"
RUST_VERSION="1.78.0"

echo ">>> Starting system setup as root..."

# 1. Update and Install Essential System Packages
echo ">>> Installing system dependencies..."
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
    llvm

# 2. Install Kernel Headers and eBPF Tools
# These are critical for the eBPF components of the project
echo ">>> Installing kernel headers and eBPF tools..."
apt-get install -y --no-install-recommends \
    linux-headers-$(uname -r) \
    linux-tools-common \
    linux-tools-$(uname -r) \
    bpftrace

# 3. Install CUDA 12.2 and NVIDIA Driver 535
# A30 is best supported by the 535-server driver series
echo ">>> Configuring NVIDIA repository and installing CUDA 12.2..."
wget -qO /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i /tmp/cuda-keyring.deb
apt-get update -y

# Installing Toolkit and Server-grade drivers for A30
apt-get install -y --no-install-recommends \
    cuda-toolkit-12-2 \
    nvidia-driver-535-server \
    cuda-drivers-535

# Set System-wide Environment Variables for CUDA
if ! grep -q "cuda-12.2" /etc/profile; then
    echo 'export PATH=/usr/local/cuda-12.2/bin:$PATH' >> /etc/profile
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH' >> /etc/profile
    echo 'export CUDA_HOME=/usr/local/cuda-12.2' >> /etc/profile
fi

# Export for the current shell session
export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.2

# 4. Install Rust Toolchain
echo ">>> Installing Rust $RUST_VERSION..."
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain "$RUST_VERSION"
# Source cargo for the current session
source "$HOME/.cargo/env"
rustup component add clippy rustfmt

# 5. Install Miniconda3 for Python Environment Management
echo ">>> Installing Miniconda3..."
if [ ! -d "/opt/miniconda3" ]; then
    wget -qO /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash /tmp/miniconda.sh -b -p /opt/miniconda3
    rm -f /tmp/miniconda.sh
fi

# Load conda for current session
eval "$(/opt/miniconda3/bin/conda shell.bash hook)"
conda init bash 2>/dev/null || true

# 6. Create vLLM Benchmark Environment
echo ">>> Creating vLLM benchmark environment..."
if ! conda env list 2>/dev/null | grep -q "^vllm-bench "; then
    conda create -n vllm-bench python=3.12 -y
fi
conda activate vllm-bench

echo ">>> Installing vLLM with FlashInfer..."
pip install --no-cache-dir vllm

# 7. Configure FlashInfer for A30 (Compute Capability 8.0)
echo ">>> Configuring FlashInfer for A30 (sm_80)..."
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
CCCL_FILE="$SITE_PACKAGES/flashinfer/compilation_context.py"
if [ -f "$CCCL_FILE" ] && ! grep -q "CCCL_DISABLE_CTK_COMPATIBILITY_CHECK" "$CCCL_FILE" 2>/dev/null; then
    echo "    Applying CCCL compatibility patch..."
    sed -i 's/COMMON_NVCC_FLAGS = \[/COMMON_NVCC_FLAGS = ["-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",/' "$CCCL_FILE"
fi

# Pre-compile FlashInfer for A30 sm_80
export FLASHINFER_CUDA_ARCH_LIST="8.0"
export CUDA_HOME="/usr/local/cuda-12.2"

# Trigger FlashInfer JIT compilation
echo ">>> Pre-compiling FlashInfer kernels for A30..."
python3 -c "import flashinfer; print('FlashInfer version:', flashinfer.__version__)" 2>&1 || true

# 8. Download TinyLlama Model
echo ">>> Downloading TinyLlama model..."
pip install --no-cache-dir huggingface_hub

mkdir -p "$MODEL_DIR"
if [ ! -f "$MODEL_DIR/tinyllama/model.safetensors" ] && [ ! -f "$MODEL_DIR/tinyllama/model.safetensors.index.json" ]; then
    huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --local-dir "$MODEL_DIR/tinyllama" \
        --exclude "*.bin" "*.pt" "*.msgpack" "*.h5"
else
    echo "    Model already exists, skipping download."
fi

# 9. Clone and Build Project
echo ">>> Cloning and building the benchmark project..."
if [ ! -d "$REPO_DIR" ]; then
    git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"
cargo build --release
cargo build --release --example bench_throughput

echo ""
echo "======================================================================"
echo " SETUP COMPLETE"
echo "======================================================================"
echo " 1. ACTION REQUIRED: Run 'reboot' to activate the NVIDIA drivers."
echo " 2. After reboot, verify the GPU using 'nvidia-smi'."
echo " 3. Verify CUDA: 'nvcc --version' should show 12.2."
echo " 4. Activate vLLM env:  conda activate vllm-bench"
echo " 5. Project Directory:  $REPO_DIR"
echo " 6. Model Directory:    $MODEL_DIR/tinyllama"
echo " 7. Run benchmark:      $REPO_DIR/scripts/run_step3_bench_baremetal.sh {baseline|vllm|compare}"
echo "======================================================================"