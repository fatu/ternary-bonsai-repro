#!/usr/bin/env bash
# One-time environment on the RTX box: uv venv, vLLM first (it pins torch), then this package + eval.
#   bash scripts/setup_env.sh
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || { echo "install uv first: https://docs.astral.sh/uv/getting-started/installation/"; exit 1; }
uv venv --python 3.12 .venv
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install vllm
uv pip install -e ".[eval,dev]"
python - <<'PY'
import torch, transformers, vllm
print(f"torch {torch.__version__} · cuda {torch.cuda.is_available()} · {torch.cuda.device_count()} GPUs"
      f" · vllm {vllm.__version__} · transformers {transformers.__version__}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  gpu{i} {p.name} {p.total_memory / 2**30:.0f} GiB")
PY
python -m pytest tests -q
