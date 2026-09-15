# Environment definition — Isambard-AI Phase 2 (verified 2026-09-15)

Issue #1. ARM-native stack for all duplex-live work on GH200 nodes.

## Cluster

- Login: `ssh u6si.aip2.isambard` (user `dsyj94.u6si`), login node `login45`
- Compute: 1,320 nodes × **4× NVIDIA GH200 120GB** (Grace-Hopper, aarch64,
  288 cores, ~460 GB RAM/node), SLURM partition `workq`, account `brics.u6si`
- Workspace: `/projects/u6si/duplex-live/` (Lustre; subdirs `data/ models/
  containers/ runs/ logs/ pylibs/ repo/`)
- Repo clone on Isambard: `/projects/u6si/duplex-live/repo` (read-write via
  repo deploy key `~/.ssh/github_duplex_live`)

## Container

`/projects/u6si/containers/latentsprites-pytorch-25.05.sif`
(NGC PyTorch 25.05, aarch64/SBSA — reused from prior project)

- Python 3.12.3, torch 2.8.0a0+5228986c39.nv25.05, CUDA 12.9, flash_attn 2.7.3

## Python overlay (`pylibs`, PYTHONPATH-injected)

`/projects/u6si/duplex-live/pylibs` — installed via
`pip install --target=...` from inside the container:

- transformers 5.17.0, accelerate 1.15.0
- soundfile 0.14.0, librosa 0.11.0, qwen-omni-utils 0.0.9
- numpy 1.26.4, scipy 1.17.1

**Gotcha 1:** pip `--target` pulled a generic torch 2.14.0 + CUDA-13 wheels that
shadow the NGC build — these were manually removed. Re-installs into `pylibs`
must use `--no-deps` or be followed by `rm -rf pylibs/torch* pylibs/nvidia_*`.

**Gotcha 2 (MoE on GH200):** torch 2.8's native `torch._grouped_mm` kernel
asserts on GH200 (`prepare_grouped_gemm_data: delta % align == 0`). Any MoE
model (Qwen3-Omni thinker, A3B experts) must load with
`experts_implementation="batched_mm"` in `from_pretrained`. Applies to Layer-A
fine-tune too — put it in the training config, not just ad-hoc scripts.

## Usage pattern

```bash
SIF=/projects/u6si/containers/latentsprites-pytorch-25.05.sif
DL=/projects/u6si/duplex-live
srun --nodes=1 --gpus=1 --cpus-per-gpu=8 --time=04:00:00 \
  apptainer exec --nv --bind /projects,/lus,/home \
  --env PYTHONPATH=$DL/pylibs \
  --env HF_HOME=$DL/models/hf-cache \
  --env PATH=$DL/pylibs/bin:/usr/local/bin:/usr/bin:/bin \
  $SIF python <script>
```

Verified: `torch.cuda.is_available()=True`, GH200 120GB visible, all overlay
imports resolve (torch from NGC, transformers from pylibs).

## Smoke test results (2026-09-15)

Qwen3-Omni-30B-A3B thinker, one GH200 120GB (`runs/qwen_smoke.py` on Isambard):

- load 24.7s from Lustre | generate 16 tok in 1.98s (~8 tok/s, single stream)
- peak memory 64.6 GB — fits with ~55 GB headroom for talker/code2wav/KV
- reply correct ("duplex online"); requires Gotcha-2 workaround
- PersonaPlex-7B-v1 weights: pending user's HF token (NVIDIA OML gate)

## Model weights (HF cache: $DL/models/hf-cache)

- `Qwen/Qwen3-Omni-30B-A3B-Instruct` — Apache-2.0, ungated
- `openbmb/MiniCPM-o-4_5` — Apache-2.0, ungated
- `nvidia/personaplex-7b-v1` — **gated**: requires HF account + NVIDIA OML
  acceptance at huggingface.co/nvidia/personaplex-7b-v1, then `HF_TOKEN`

## Notes

- MiniCPM-o 4.5 duplex path is transformers-native (`trust_remote_code=True`,
  `.as_duplex()`); llama.cpp is not required (Runbook 01 §3 amended).
- Remote code in MiniCPM-o may pin older transformers — test under 5.17.0
  first, fall back to a pinned 4.x overlay if it breaks.
- Downloads run on the login node (no allocation cost); compute via `srun`
  only.
