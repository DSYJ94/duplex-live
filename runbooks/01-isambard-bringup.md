# Runbook 01 — Isambard-AI bring-up (Issue #1)

Target: one interactive GH200 node (4× GH200, aarch64) running our three
candidate models. Allocation: Isambard-AI Phase 2 (terminates 2026-10-21).

## 0. Access

```bash
ssh <user>@isambard.ai   # per BriCS onboarding; check portal for exact host
```

## 1. Get a node

```bash
# interactive dev session, 1 node, 4h
salloc -p <partition> -N 1 -t 04:00:00    # confirm partition name on portal
srun --pty bash
```

## 2. ARM-native environment (container preferred)

Isambard is aarch64: **no x86 binaries, no x86 Conda envs.** Use the site
Apptainer/Singularity with an ARM64 base, or a venv with ARM wheels.

```bash
# check what the site provides
module avail 2>&1 | grep -i -E 'pytorch|cuda|python'
# NVIDIA NGC PyTorch containers ship linux/arm64 (SBSA) variants — preferred
# e.g. nvcr.io/nvidia/pytorch:26.xx-py3 (verify ARM64 tag before pulling)
apptainer pull pytorch.sif docker://nvcr.io/nvidia/pytorch:<arm64-tag>
```

Fallback: venv + pip ARM wheels (`torch` aarch64+cuda wheels from NVIDIA
index; llama.cpp builds natively on Grace with CUDA enabled).

## 3. Smoke tests (each module individually — see docs/09 §4 risks)

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
# expect: True 4

# Qwen3-Omni 30B-A3B (needs ~65 GB bf16 — fits on one 96 GB GH200, or shard)
python -c "from transformers import AutoModel; print('transformers ok')"

# llama.cpp ARM sanity path (MiniCPM-o)
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build -j
```

## 4. Record results

Post environment definition (container/venv + versions) to Issue #1 and
commit it under `training/`.
