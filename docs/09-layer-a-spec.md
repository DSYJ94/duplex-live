# 09 — Layer-A Spec: The Surgical Full-Duplex Modification to Qwen3-Omni

**Status:** draft spec (2026-09-14) · implements Issue #2's Layer-A sub-task · feeds Issue #7

## 0. What we're modifying (ground truth from the technical report)

Qwen3-Omni-30B-A3B components:

| Module | Params | Native rate | Streaming? |
|---|---|---|---|
| AuT audio encoder | 650M | **12.5 Hz** (1 audio token = 80 ms) | ✓ chunked |
| Thinker | 30B-A3B MoE | text tokens | ✓ chunked prefill |
| Talker | 3B-A0.3B MoE | **12.5 Hz** (1 codec frame/step = 80 ms audio) | ✓ |
| MTP (residual codebooks) | 80M | per Talker step | ✓ |
| Code2Wav | 200M ConvNet (causal) | frame-by-frame | ✓ |

Current inference flow (turn-based):

```
user audio → AuT streams chunks → Thinker async-prefills chunks
  → [endpoint detected: VAD/external] → Thinker decodes text
  → Talker decodes codec frames (12.5 Hz) → Code2Wav → speaker

during Thinker/Talker generation: new user audio is NOT in context.
Perception is frozen mid-response → half-duplex.
```

**The single root cause of half-duplex:** the endpoint gate. Generation only
starts after the user "finishes", and no new AuT frames enter the Thinker's
context until generation ends. Layer A removes that gate.

## 1. The core trick: serialize the two streams into one causal sequence

We do NOT need parallel attention streams or exotic masks. Following TML's
micro-turn design: interleave input and output into **a single causal token
sequence on a fixed time grid**. Causality is preserved; the only change is
*what gets appended when*.

**Grid: 80 ms ticks (12.5 Hz)** — the model's native rate on both sides (AuT
frame = Talker frame). No resampling, no rate mismatch.

Sequence layout:

```
… [A_in t] [x_t] [A_in t+1] [x_t+1] [A_in t+2] [x_t+2] …

A_in t : AuT embedding of user audio frame t  (always present, even silence)
x_t    : Thinker emission for tick t — exactly one of:
           text token      → forwarded to Talker (speech continues)
           <sil>           → Talker emits silence frame
           <delegate>      → fire async RPC (serving layer intercepts)
           <yield>         → stop speaking, listen (after barge-in)
```

Every tick: one 1-token prefill (audio in) + one 1-token decode (emission).
Both are trivially cheap for a 3B-active MoE. The KV cache is a standard
append-only causal cache — no re-prefill, ever.

## 2. Concrete changes, component by component

### 2a. Thinker driver loop (the main change) — serving/training code, not weights

Replace vLLM's request-response generate() with a tick loop:

```python
each 80 ms tick t:
    audio_tok = AuT.encode_frame(mic_frame(t))        # 12.5 Hz streaming
    kv.append(audio_tok)                              # 1-token prefill
    x = thinker.decode_one()                          # sampled from {text, <sil>, <delegate>, <yield>}
    kv.append(x)
    route(x)   # text→Talker | <sil>→silence frame | <delegate>→RPC | <yield>→flush Talker queue
```

This is ~200 lines of driver code around the existing model. The Thinker's
**weights need no architectural change** — appended audio tokens are just
more context. What changes is *when* tokens enter.

### 2b. Talker coupling — text-decoupled design is our friend

The report notes the Talker no longer consumes Thinker text representations
directly, and that "external modules can intervene on the Thinker's textual
output… via controlled preprocessing." We use exactly that hook:

- Talker consumes Thinker emissions tick-aligned, padding `<sil>` ticks with
  silence codec frames.
- **Retraction:** on `<yield>`, flush Talker's pending codec queue (frames not
  yet rendered by Code2Wav) and mark the truncation point in the training
  sequence. This is how interruption becomes clean instead of clipped.

### 2c. New token embeddings (weights, small)

Add to the 151,643-token vocab: `<sil>`, `<t+Ns>` timing buckets, `<delegate>`,
`<yield>`, `<barge-in>`. New rows in embed/unembed initialized from
mean-of-neighbors; everything else frozen in Phase 1.

### 2d. Explicit timestamps — already half-built

The Qwen3.5-Omni report notes "explicit timestamps inserted to improve
temporal perception." Qwen3-Omni has TM-RoPE lineage for time-aligned A/V.
Our `<t+Ns>` tokens formalize this into the tick stream; we are extending an
existing capability, not inventing one.

## 3. Training the modification (maps to docs/08 Steps 1–5)

1. Render synthetic duplex conversations onto the 80 ms interleaved grid
   (tokenizer/ + data/synthetic/ — Issues #2, #3).
2. Phase 1: freeze Thinker/Talker backbones; train AuT adapter + new
   embeddings + (light) Talker retraction behavior. Days on a few GH200s.
3. Phase 2: full FT with text replay; checkpoint-gated on interactivity ↑
   and intelligence regression < 2%.

## 4. What could bite us (risk register)

| Risk | Likelihood | Mitigation |
|---|---|---|
| Thinker attention degrades with dense interleaved audio (2× context growth per tick) | Medium | `<sil>` runs compressible via periodic summarization; monitor long-session evals |
| Talker retraction artifacts (clipped phonemes on barge-in) | Medium | retraction training data with 80–240 ms fade-out windows |
| vLLM fighting the custom tick loop | Medium | bypass vLLM scheduler for single-session demo; upstream later if worth it |
| aarch64 kernel gaps for AuT/MTP/ConvNet | Medium | Issue #1 bring-up smoke-tests each module individually |
| Per-tick Thinker decode too slow at 12.5 Hz under load | Low | 3B active, 1-token steps; batch ticks across sessions, not within |

## 5. Note: MiniCPM-o 4.5 as a possible shortcut (evaluate in Issue #4)

Per a March 2026 benchmark appendix, MiniCPM-o 4.5 (Qwen3-8B backbone, open)
ships **full-duplex streaming with timeline modeling (TDM)** and an
interleaved text–speech token decoder — i.e., someone may have already done
a Layer-A-class modification on a Qwen backbone. Action: evaluate it in the
same week as PersonaPlex. If its interleaved decoder is real and its license
is clean, it could replace the 2a driver-loop work or serve as a second
reference implementation. Its 8B dense backbone is less intelligent than
Qwen3-Omni's 30B-A3B, so the intelligence-max track still points at
Qwen3-Omni — but MiniCPM-o's code may show us exactly how they solved the
tick loop and retraction.

## 6. Milestones for this spec

- [ ] M1: tick-loop driver runs unmodified Qwen3-Omni in forced half-duplex
      (proves the driver) — week 1 after Issue #1
- [ ] M2: interleaved grid renderer produces training sequences (with #2/#3)
- [ ] M3: Phase-1 checkpoint emits sensible `<sil>`/speech decisions on
      streaming eval (Gate A→B bridge)
- [ ] M4: retraction works: barge-in mid-speech → clean yield → resume/revise
