# tokenizer/

Canonical micro-turn tokenizer/streamer — the single implementation shared by training and serving.

- Chunks all streams (audio, video patches, text) onto the **200 ms micro-turn grid**.
- Produces the interleaved `in0 out0 in1 out1 ...` token sequence exactly as the model sees it.
- Emits timing/silence tokens so elapsed time is explicit in the representation.
- One codebase, two consumers: training sampler and streaming-session server. Divergence here = trainer-sampler misalignment.

First task (Q1 W1–2): implement the grid for Qwen3-Omni audio tokens. See docs/08.
