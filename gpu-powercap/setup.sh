#!/usr/bin/env bash
# Setup for a fresh Ubuntu GPU VM (Lambda or any provider that gives you root). Fine to run twice.
#   echo 'export HF_HOME=/<persistent-volume>/hf' >> ~/.bashrc && source ~/.bashrc   # optional
#   bash setup.sh                  # FLASHINFER=1 bash setup.sh to also install FlashInfer (see below)
# Put HF_HOME in ~/.bashrc rather than exporting it in one shell: every later shell (tmux, a new SSH
# login) has to see the same cache, because the sweep only ever uses the cached model.
set -euo pipefail
cd "$(dirname "$0")"

# use the same vLLM version on both GPUs. VLLM_VERSION=x.y.z bash setup.sh to change it
VLLM_VERSION="${VLLM_VERSION:-0.11.0}"
# Install every package as it was on PyPI on this date, so the H100 box and the B200 box get exactly
# the same stack however far apart they're rented. Use the same value on both if you change it.
PIP_CUTOFF="${PIP_CUTOFF:-2025-10-20T00:00:00Z}"

SUDO=sudo; [ "$(id -u)" = 0 ] && SUDO=""  # some providers log you in as root, sometimes without sudo
$SUDO apt-get update -y
# python3-dev + build-essential: Triton (under torch.compile, in vLLM and the training probe) compiles C
# helpers against Python.h at runtime
$SUDO apt-get install -y python3-venv python3-dev build-essential sysstat tmux

if [ ! -d .venv ]; then python3 -m venv .venv; fi
source .venv/bin/activate
pip install -U pip  # new enough for --uploaded-prior-to
# transformers 5 breaks vLLM 0.11.0's tokenizer setup (0.11.1 added the same cap). uvloop speeds up
# the load generator, psutil measures vLLM's API server CPU
PKGS=("vllm==${VLLM_VERSION}" "transformers>=4.55.2,<5" nvidia-ml-py aiohttp uvloop psutil)
# FlashInfer is what vLLM wants for attention on Blackwell; without it a B200 falls back to
# FlashAttention 2. (Sampling doesn't change: the load generator pins plain sampling.) Version 0.3.1
# compiles its kernels on first use, so it needs nvcc from CUDA 12.8+. Check the smoke test's
# vllm_server.log says "Using FlashInfer backend" before relying on it.
if [ "${FLASHINFER:-0}" = "1" ]; then PKGS[0]="vllm[flashinfer]==${VLLM_VERSION}"; fi
pip install --uploaded-prior-to "$PIP_CUTOFF" "${PKGS[@]}"

# DCGM is nice to have but not required. DCGM 4 first: the legacy 3.x package can't see a B200
if ! command -v dcgmi >/dev/null; then
  CUDA_MAJOR=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+' || true)
  $SUDO apt-get install -y --install-recommends "datacenter-gpu-manager-4-cuda${CUDA_MAJOR}" \
    || $SUDO apt-get install -y datacenter-gpu-manager || echo "couldn't install DCGM, skipping it"
fi
if command -v dcgmi >/dev/null; then
  $SUDO systemctl enable --now nvidia-dcgm 2>/dev/null || $SUDO systemctl enable --now dcgm 2>/dev/null || true
fi

# grab the model
python -c "import config; from huggingface_hub import snapshot_download; print(snapshot_download(config.MODEL))"

python -c "import vllm, torch, transformers; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__)"
echo "All set. Next: source .venv/bin/activate && python preflight.py"
