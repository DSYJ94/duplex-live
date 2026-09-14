# training/

Training code (see `docs/03-training-program.md` and `docs/08-duplex-finetune-recipe.md`).

- `finetune/` — Track 1: Qwen3-Omni duplex fine-tune (Phase 1 frozen backbone + new embeddings; Phase 2 full FT + text replay)
- `preference/` — timing-aware reward model + preference optimization
- `pretrain/` — Track 2 (gated): 1B pilot -> 3B scratch, native micro-turn format
- `scaling/` — small-scale mixture experiments

Hardware: Isambard-AI GH200, **aarch64 only** — containers must be ARM-native.
