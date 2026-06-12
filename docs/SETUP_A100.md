# Environment setup (A100, Ubuntu 22.04+)

## 0. Sanity
nvidia-smi                      # driver visible, A100 listed
# CUDA 12.x driver assumed; PyTorch wheels bundle their own CUDA runtime.

## 1. System packages
sudo apt update && sudo apt install -y build-essential git tesseract-ocr tesseract-ocr-hin

## 2. Python env (uv)
curl -LsSf https://astral.sh/uv/install.sh | sh
cd nyaya-rag
uv venv --python 3.11
source .venv/bin/activate

## 3. Install (order matters: torch first with the CUDA index)
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
uv pip install -e ".[dev,rag]"
pre-commit install

## 4. vLLM (local generation server — Day 6 onward)
uv pip install vllm
# serve (one terminal, stays up):
#   vllm serve Qwen/Qwen2.5-14B-Instruct --max-model-len 8192 --gpu-memory-utilization 0.85
# 14B bf16 ≈ 28 GB weights + KV cache: comfortable on 40 GB, trivial on 80 GB.
# Smoke test:
#   curl http://127.0.0.1:8000/v1/models

## 5. Verify
python - << 'PY'
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
PY
make smoke
