#!/usr/bin/env bash
set -e
# Python 3.14 + a CUDA 12.6-compatible GPU required for inference.
python -c 'import sys; assert sys.version_info[:2] == (3, 14), f"Python 3.14 required, got {sys.version.split()[0]}"'
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
# NVIDIA BigVGAN is a repo, not a pip package (no setup.py). Clone it and add to PYTHONPATH so
# `import bigvgan` resolves (we use the torch path / use_cuda_kernel=False, so nothing is compiled).
[ -d BigVGAN/.git ] || git clone --depth 1 https://github.com/NVIDIA/BigVGAN.git BigVGAN
export PYTHONPATH="$PWD/BigVGAN:$PYTHONPATH"   # add this to your shell rc for persistent use
python scripts/download_weights.py                        # our weights -> models/ + IndicF5 base (vocab)
echo "✓ setup complete — see Quickstart in README.md"
