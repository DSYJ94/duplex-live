# Duplex approach survey — literature & tech scan (2026-10-08)

Primary-source survey for the duplex-live fine-tune: what the 2024–2026 full-duplex
literature actually did, and what it implies for our plan (self-history round, text-word
emission tokens, banked-utterance demo, 80 ms arrears loop).

Method: every claim below traces to a primary source (arXiv paper, official model card /
release blog, official docs, or first-party code repo), fetched 2026-10-07. arXiv IDs were
verified against the abstract pages — several IDs floating around in secondary write-ups
(including some we started with) are wrong, so treat the links here as canonical.
Confidence note: papers dated 2026 are recent preprints with light citation counts; their
numbers may move on revision.

---

## 1. Full-duplex spoken-dialogue architectures 2024–2026

**Plain-language summary.** "Full duplex" here means one model that keeps listening while
it talks, so it can be interrupted, backchannel ("mm-hm"), and overlap, instead of the
classic pipeline (recognize speech → think → synthesize speech, one turn at a time). There
are two big architectural camps: (1) **joint audio generation** — the model emits audio
codec tokens directly (Moshi, dGSLM, OmniFlatten, NTPP); and (2) **text-mediated** — the
model emits text (or text-like control tokens) and a separate component turns that into
speech (Freeze-Omni, LLaMA-Omni 1/2, GLM-4-Voice, Full-Duplex-1, and essentially our
design). Turn-taking is modeled one of three ways: as emergent behavior of two parallel
audio streams (Moshi, dGSLM), as special state/timing tokens predicted by the LLM
(Freeze-Omni, Full-Duplex-1, OmniFlatten, SyncLLM, us), or by an external VAD-plus-rules
harness (OpenAI's Realtime API offers both). Published decision cadences sit at 160–240 ms
(SyncLLM) or 200 ms (Thinking Machines); 80 ms appears only as a *codec frame* rate
(Moshi's Mimi, GLM-4-Voice tokenizer), not as a decision rate. Every system that handles
barge-in well trained on at least some **real two-channel conversational audio**;
synthetic-only systems got assistant-side timing but weak interruption handling.

### Comparison table

| System (source) | Params / base | Speech output path | Turn-taking / barge-in model | Frame rate / latency | Training data (timing-critical) |
|---|---|---|---|---|---|
| **Moshi** ([arXiv 2410.00037](https://arxiv.org/abs/2410.00037), [full text](https://arxiv.org/html/2410.00037v2), [code](https://github.com/kyutai-labs/moshi)) | 7B Helium text backbone + small depth transformer | Joint: 17 parallel streams (1 text "inner monologue" + 2 × 8-codebook Mimi audio) | Emergent: model generates its own and the user's streams; EPAD tokens; interruption = overlap in streams | Mimi 12.5 Hz (80 ms frames); **160 ms theoretical, 200 ms in practice** | 7M h unsupervised audio (Whisper-transcribed) → **2,000 h Fisher** (real overlap finetune) → **>20k h synthetic** multi-stream TTS with noise/echo/reverb augmentation |
| **dGSLM** ([arXiv 2203.16502](https://arxiv.org/abs/2203.16502), [demo](https://speechbot.github.io/dgslm/)) | ~1B-scale textless unit LM, dual-tower transformer + cross-attention | Joint: HuBERT-style units, two streams, vocoder after | Emergent from interleaved unit streams | Unit-rate generation; not real-time | **2,000 h Fisher**, no text/labels |
| **GPT-4o realtime / Realtime API** ([docs](https://developers.openai.com/api/docs/guides/realtime), [VAD guide](https://developers.openai.com/api/docs/guides/realtime-vad)) | closed (`gpt-realtime-2.1` et al.) | Speech-to-speech streaming with `response.output_audio.delta` events | External harness: `server_vad` (silence threshold/padding) or **`semantic_vad`** (classifier scores whether the user is done; trails off → longer timeout) | Not published; FD-Bench V1 turn-taking latency measured at 0.57 s (gpt-realtime-1.5) / 1.18 s (gpt-realtime-2.0) by Thinking Machines ([blog](https://thinkingmachines.ai/blog/interaction-models)) | Not published |
| **Thinking Machines TML-Interaction-Small** ([blog](https://thinkingmachines.ai/blog/interaction-models)) | 276B MoE, 12B active, trained from scratch | Joint: dMel audio tokens in and out; flow head for audio output | **In-model, implicit** — no VAD harness; dialog state (user thinking/yielding/self-correcting) tracked by the model itself | **200 ms micro-turns** (200 ms in ↔ 200 ms out interleaved); FD-Bench V1 turn-taking latency 0.40 s | Not published (co-trained from scratch; safety data via TTS + red-teaming harness) |
| **Freeze-Omni** ([arXiv 2411.00774](https://arxiv.org/abs/2411.00774), [full text](https://arxiv.org/html/2411.00774v2), [code](https://github.com/VITA-MLLM/Freeze-Omni)) | Qwen2-7B-Instruct **frozen** + speech encoder + codec speech decoder | Cascaded: LLM text → AR/NAR speech-token decoder (40 Hz TiCodec) | **Chunk-level state classifier on the LLM's last-layer hidden state**: state 0 keep listening, 1 user interrupting, 2 speech ended; silero-VAD only gates onset | Non-statistical detection ~160–320 ms (1–2 chunks); interrupt→first-PCM 745 ms avg (p90 1,020 ms) on 1×A100; ~1.2 s with network | 110k h ASR; ~3k h TTS pairs; **60k multi-round text QA** (questions TTS'd); no real conversation hours reported |
| **LLaMA-Omni** ([arXiv 2409.06666](https://arxiv.org/abs/2409.06666), [code](https://github.com/ictnlp/LLaMA-Omni)) | Llama-3.1-8B + Whisper-large-v3 encoder | Cascaded: text + **non-autoregressive** speech decoder | Half-duplex (no duplex claims) | **226 ms** response latency; trained <3 days on 4 GPUs | InstructS2S-200K (200k synthetic speech instructions) |
| **LLaMA-Omni2** ([arXiv 2505.19826](https://arxiv.org/abs/2505.19826), [code](https://github.com/ictnlp/LLaMA-Omni2)) | Qwen2.5 0.5B–14B family | Cascaded: autoregressive streaming speech decoder (CosyVoice-2-style token streaming) | Half-duplex | ~600 ms | Text dialogues + ASR data (per paper's real-time recipe; no duplex training) |
| **GLM-4-Voice** ([arXiv 2412.02612](https://arxiv.org/abs/2412.02612), [code](https://github.com/THUDM/GLM-4-Voice)) | GLM-4-9B + 175 bps single-codebook tokenizer (12.5 Hz) | Speech tokens AR in the LLM + streaming flow decoder; synthesis can start after 10 tokens | Half-duplex, streaming | 80 ms per token frame; streaming decoder keeps latency low | Speech tokenizer from ASR model; instruct data (details in paper) |
| **Mini-Omni / Mini-Omni2** ([2406.14275](https://arxiv.org/abs/2406.14275) / [2410.11190](https://arxiv.org/abs/2410.11190), [code](https://github.com/gpt-omni/mini-omni2)) | small LLM + audio encoder | "Audio-to-audio" with text-then-speech parallel decoding | Mini-Omni2 adds **duplex endpoints** (interruption during voice interaction) | Real-time end-to-end claims | Synthetic instruction speech |
| **MiniCPM-o 2.6** ([model card](https://huggingface.co/openbmb/MiniCPM-o-2_6), [report](https://openbmb.vercel.app/minicpm-o-2-6-en), [code](https://github.com/OpenBMB/MiniCPM-V)) | 8B (SigLIP-400M + Qwen2.5-7B) | Integrated streaming ASR + TTS (speech out via token decoder) | **Duplex mode**: streaming chunked audio in, interruption allowed mid-generation, turn-taking steered by system prompts | ~1.8 s initial response per model card | Synthetic speech instructions; no real dyadic audio claimed |
| **MiniCPM-o 4.5** ([arXiv 2604.27393](https://arxiv.org/abs/2604.27393)) | 9B | **Omni-Flow**: all modalities aligned on a shared temporal axis; true real-time full-duplex (perceive while responding), proactive comments | In-model, time-aligned streaming | <12 GB RAM edge target; numbers in paper | Not in abstract |
| **SyncLLM** ([arXiv 2409.15594](https://arxiv.org/abs/2409.15594), [full text](https://arxiv.org/html/2409.15594), [page](https://syncllm.cs.washington.edu)) | Llama3-8B + HuBERT units (25 Hz, 501-token vocab) | Joint unit generation (speech units; text chunks time-aligned) | Time-aligned chunks for both speakers; model **generates an estimated user chunk ahead**, then replaces it with real audio | Chunks of **160–240 ms** (4–6 units); robust to ≤200 ms transport latency, degrades at 240 ms | **212k h synthetic** spoken dialogue (from text dialogues) + **~2k h real Fisher** |
| **Parrot** ([OpenReview 73EDGbG6mB](https://openreview.net/forum?id=73EDGbG6mB), NeurIPS 2024) | off-the-shelf audio LLM, fine-tuned | Joint: dual-channel audio tokens, **next-token-pair prediction** (one token per channel per step) | Turn-taking emerges from time-aligned channel pairs; metrics include latency-after-interruption | Low-latency claims in paper | Large single-channel pretraining, then "moderate-scale" double-channel SFT |
| **NTPP** ([arXiv 2506.00975](https://arxiv.org/abs/2506.00975), ICML 2025, [code](https://github.com/Chaos96/NTPP)) | speech LM (ByteDance) | Joint: dual-channel speech tokens via next-token-pair prediction | Same paradigm as Parrot, formalized | — | **2,200 h Fisher** phone conversations |
| **OmniFlatten** ([arXiv 2410.17799](https://arxiv.org/abs/2410.17799), [full text](https://arxiv.org/html/2410.17799v1)) | **Qwen2-0.5B**, NanoGPT base | Joint: flattened interleaved streams (CosyVoice tokenizer), `silent_speech_token` / `silent_text_token` as padding | Turn-taking = emitting non-silence (take turn) vs silence (yield); **synthetic data has no true interruptions → user-turn detection weak (~30% by token 25)** | Assistant turn-taking ~160 ms avg response; user-turn ~805 ms | ~100k h ASR/TTS alignment + **2,000 h synthetic two-channel dialogue** (CosyVoice TTS of text chats, MUSAN noise) — zero real conversation |
| **Full-Duplex-1 / TDM scheme** ([arXiv 2405.19487](https://arxiv.org/abs/2405.19487), [arXiv 2406.15718](https://arxiv.org/abs/2406.15718)) | LLM (Qwen-based line) | **Text tokens + streaming XTTS-v2** (cascaded) | Duplex behavior as learned token patterns in the LLM; TDM time-slicing pseudo-parallelizes listen/speak | >3× lower response latency vs half-duplex LLM systems | Time-sliced query/response SFT data; released under CC BY-NC-SA |
| **LSLM** ([arXiv 2408.02622](https://arxiv.org/abs/2408.02622)) | token-TTS decoder + streaming SSL encoder | Joint-ish: listening tokens fused into a decoder-only TTS (early/mid/late fusion; middle best) | Real-time barge-in detection from fused listening channel (command- and voice-based) | Real-time claims | Speech + text corpora (details in paper) |
| **MinMo** ([arXiv 2501.06282](https://arxiv.org/abs/2501.06282)) | ~8B aligned multimodal | Voice decoder + duplex interaction-alignment stage | Fourth training stage aligns duplex interaction | S2T ~100 ms; **full-duplex ~600 ms theoretical / ~800 ms practice** | **1.4M h** diverse speech across 4 stages |
| **SALMONN-omni** ([arXiv 2505.17060](https://arxiv.org/abs/2505.17060), [code](https://github.com/bytedance/SALMONN)) | LLM backbone, **no codec injection** | LLM text/prosody out; speech via its own path | **Dynamic thinking mechanism** in the LLM learns when to speak vs listen; **+ RL** for turn-taking, backchanneling, echo cancellation, barge-in | Not in abstract | "Substantially less training data" than baselines; 30%+ relative gains over open-source duplex baselines |
| **SALM-Duplex** ([arXiv 2505.15670](https://arxiv.org/abs/2505.15670), NVIDIA) | small SALM model | Continuous user-speech input + codec agent outputs, **channel fusion** | Directly models simultaneous streams; no time-slicing | Outperforms prior duplex models (incl. Moshi) on reasoning / turn-taking / barge-in at small scale | Spoken-QA-derived (details in NeMo) |
| **Qwen3-Omni (our base)** ([TR: arXiv 2509.17765](https://arxiv.org/abs/2509.17765)) | 30B-A3B MoE, Thinker + Talker | Talker predicts codec tokens with a **multi-token-prediction (MTP) module**; causal ConvNet decoder, streams from first frame | Half-duplex streaming natively; duplex is what we are adding | **234 ms theoretical first-packet latency** (cold start) | See TR; Apache-2.0 weights |
| **Qwen3.5-Omni** ([TR html](https://arxiv.org/html/2604.15804v1)) | MoE | Inherits Qwen3-Omni Talker/MTP + ConvNet design for streaming & high-concurrency serving | — | — | — |

2026 wave (existence proofs multiplying; IDs from the [Awesome-Full-Duplex-SDM list](https://github.com/Ruiqi-Yan/Awesome-Full-Duplex-SDM)):
DuplexMamba [2502.11123](https://arxiv.org/abs/2502.11123), MoshiRAG [2604.12928](https://arxiv.org/abs/2604.12928)
(Kyutai, announced 2026-04-29), Personaplex [2602.06053](https://arxiv.org/abs/2602.06053),
BayLing-Duplex [2606.14528](https://arxiv.org/abs/2606.14528), DuplexOmni [2606.09186](https://arxiv.org/abs/2606.09186),
Lychee-FD [2607.06540](https://arxiv.org/abs/2607.06540), SteerDuplex [2609.12623](https://arxiv.org/abs/2609.12623),
Realtime-Venus [2609.13814](https://arxiv.org/abs/2609.13814). Kyutai's own post-training RL paper is
"Multi-Faceted Interactivity Alignment in Full-Duplex Speech Models" [2606.11167](https://arxiv.org/html/2606.11167v1)
([blog](https://kyutai.org/blog/2026-06-10-interactivity), [samples](https://huggingface.co/datasets/kyutai/interactivity-alignment-samples)).

### What actually works at small scale (billions of params, synthetic-heavy data)

- **OmniFlatten is the closest published existence proof to us**: a 0.5B model, no
  architecture changes, progressive SFT on **synthetic two-channel dialogue only**, and it
  reached ~160 ms assistant-side turn-taking ([arXiv 2410.17799](https://arxiv.org/html/2410.17799v1)).
  But the same paper is the cautionary tale: user-turn / interruption accuracy stayed
  ~30% because simulated data contained no true interruptions.
- **Freeze-Omni** shows a *frozen* ~7B LLM + a small hidden-state classifier can run a
  duplex state machine (745 ms interrupt latency) with 60k QA and zero reported real
  conversational audio ([arXiv 2411.00774](https://arxiv.org/html/2411.00774v2)).
- **SALMONN-omni** and **SALM-Duplex** explicitly target efficiency and "substantially
  less training data", using RL post-training rather than more data
  ([2505.17060](https://arxiv.org/abs/2505.17060), [2505.15670](https://arxiv.org/abs/2505.15670)).
- Scale counterexamples: SyncLLM needed 212k synthetic hours + 2k real
  ([2409.15594](https://arxiv.org/abs/2409.15594)); Moshi needed 7M h pretraining + 2k h
  Fisher + 20k h synthetic to be robust ([2410.00037](https://arxiv.org/html/2410.00037v2));
  MinMo used 1.4M h ([2501.06282](https://arxiv.org/abs/2501.06282)).
  **Nobody in this table trained timing behavior on ~1,000 synthetic conversations.**
  Our bet only makes sense because our scope is narrower than any of theirs: timing
  tokens on a frozen trunk, no audio generation in the loop, whisper-verified onsets.

---

## 2. Turn-taking modeling: the VAP line vs our hidden-state gate

**Plain-language summary.** "Voice Activity Projection" (VAP) is a small, cheap model
that listens to both speakers' audio and continuously predicts who will be speaking over
the next ~2 seconds. It trains itself on nothing but real two-speaker recordings — the
labels are just "who was speaking when", which you get for free from a stereo recording.
It never generates words; it only predicts the conversational traffic pattern. Our MLP
activity gate instead reads the *trunk's internal state* of our fine-tuned model and
classifies the current operating point. The literature says both are legitimate —
Freeze-Omni's duplex detector is literally a hidden-state classifier — but VAP is the
field's *standard instrument*: it is self-supervised, validated zero-shot, works
incrementally in real time, and plugs into benchmarks, so it can grade our gate rather
than replace it.

### The VAP line

- **Original VAP** (Ekstedt & Skantze, Interspeech 2022,
  [arXiv 2205.09812](https://arxiv.org/abs/2205.09812), [project](https://erikekstedt.github.io/VAP/),
  [code](https://github.com/ErikEkstedt/VoiceActivityProjection)): encodes stereo audio
  with a frozen CPC encoder; a small transformer predicts the joint future voice-activity
  pattern of **both** speakers over a 2 s projection window; trained **self-supervised on
  Switchboard** with no labels; outperformed prior supervised models zero-shot on
  turn-shift, backchannel and turn-hold prediction.
- **Real-time, continuous operation** (Inoue et al. 2024,
  [arXiv 2401.04868](https://arxiv.org/abs/2401.04868), [code](https://github.com/inokoj/VAP-Realtime)):
  a demo system mapping dialogue stereo audio straight to future voice activities,
  incrementally, to control spoken-dialogue behaviors online.
- **Multilingual transfer** (Inoue et al., LREC-COLING 2024,
  [arXiv 2403.06487](https://arxiv.org/abs/2403.06487)): VAP trained on English
  (Switchboard) transfers to Mandarin and Japanese — i.e. the timing signal is not
  language-bound. Follow-ups extend to multi-party (triadic) settings and prompt-guided
  control (see citations in [2401.04868](https://arxiv.org/abs/2401.04868)).
- **Standard benchmarks now exist for exactly this instrument family**:
  - **Talking Turns** ([arXiv 2503.01174](https://arxiv.org/abs/2503.01174)): benchmarks
    audio foundation models on turn-taking dynamics on real conversational audio.
  - **Full-Duplex-Bench** v1 ([arXiv 2503.04721](https://arxiv.org/abs/2503.04721),
    [code](https://github.com/DanielLin94144/Full-Duplex-Bench)): scenario-driven
    evaluation of **pause handling, backchanneling, turn-taking, interruption
    management** with automatic metrics (ASRU 2025).
  - v1.5 ([2507.23159](https://arxiv.org/abs/2507.23159), ICASSP 2026) adds **overlap
    scenarios** (user interruption, backchannel, talking to others, background speech)
    plus stop/response latency and prosodic-adaptation metrics — and found SOTA agents
    split into "responsive" (fast reaction) vs "floor-holding" (filter overlaps) strategies.
  - v2 [2510.07838](https://arxiv.org/abs/2510.07838) (multi-turn, Turn-Taking Fluency)
    and v3 [2604.04847](https://arxiv.org/abs/2604.04847); **TurnBench**
    [2608.25218](https://arxiv.org/abs/2608.25218) adds multi-domain coverage ( telephone
    vs other registers).

### VAP-style head vs our hidden-state gate

| Dimension | VAP head | Our MLP activity gate |
|---|---|---|
| Input | raw-ish audio of **both** speakers (CPC features) | trunk hidden states (already fuses user audio + our emissions) |
| Supervision | self-supervised (future voice activity) — needs **real dyadic audio** | our emission-token labels (synthetic) |
| Output | calibrated distribution over **future** 2 s voice-activity patterns (predictive) | current-state classification (reactive) |
| Portability | runs without our model; can drive/benchmark any controller | tied to the fine-tuned checkpoint |
| Comparability | directly gradeable on Talking Turns / Full-Duplex-Bench | no external yardstick |
| Precedent in duplex systems | used to control real-time SDS behavior ([2401.04868](https://arxiv.org/abs/2401.04868)) | Freeze-Omni's chunk-state classifier is exactly this and works ([2411.00774](https://arxiv.org/html/2411.00774v2)) |

Verdict from the literature: **not either/or**. Freeze-Omni validates the hidden-state
classifier as the production instrument; VAP's advantages are (a) the 2 s *predictive*
distribution (our gate fires after evidence accumulates; VAP anticipates — important
because human turn gaps are ~200 ms while generation is slower, SyncLLM's stated
motivation, [2409.15594](https://arxiv.org/html/2409.15594)); (b) zero-shot validation and
benchmark comparability; (c) independence from our checkpoint, so it can be the referee.
A hidden-state gate cannot be a referee for itself.

---

## 3. Exposure bias and free-run degeneration of token emission

**Plain-language summary.** Our free-running word content collapses into single-digit
soup because the model was only ever trained with "teacher forcing" — during training it
always saw the *correct* previous words, never its own. At inference it sees its own
outputs, which are slightly off-distribution, and the errors compound. This train/test
mismatch is called **exposure bias**. The textbook fix, "scheduled sampling" (sometimes
feed the model its own output during training), is what our planned self-history round
implements. The literature's verdict is nuanced: exposure bias is real and compounds over
long horizons, but scheduled sampling is an awkward fit for transformers and did not
consistently help in the 2019 studies; modern systems instead stabilize generation with
text-anchored decoding (Moshi's inner monologue), input augmentation, and RL post-training.
Notably, no paper we could find addresses *text-token* emission degeneration in speech
LLMs head-on — but the classic degenerate-text literature (Holtzman et al.) describes our
exact symptom and its decoding-side causes.

Key sources:

- **Scheduled sampling** (Bengio et al., NeurIPS 2015, [arXiv 1506.03099](https://arxiv.org/abs/1506.03099)):
  the original curriculum for mixing model-sampled tokens into training inputs, against
  exposure bias in RNN sequence prediction.
- **Scheduled sampling for transformers** (Mihaylova & Martins, ACL 2019 SRW,
  [aclanthology P19-2040](https://aclanthology.org/P19-2040/), [arXiv 1906.07651](https://arxiv.org/abs/1906.07651)):
  transformers lack recurrent state, so scheduled sampling needs two-pass decoding
  workarounds, and results were inconsistent across tasks — i.e. the fix is not automatic.
- **Skeptic's view** (Schmidt, 2019, [aclanthology D19-5616](https://aclanthology.org/D19-5616/)):
  argues the train/test discrepancy is better understood as a **generalization** problem;
  generalization error, not exposure bias per se, dominates observed degradation.
- **Self-recovery** (He et al., [arXiv 1905.10617](https://arxiv.org/abs/1905.10617)):
  autoregressive LMs often *recover* from distortions; free-run garbage is not proof of
  exposure bias alone.
- **DAgger** (Ross, Gordon & Bagnell, 2011, [arXiv 1011.0686](https://arxiv.org/abs/1011.0686)):
  the principled version of "train on your own rollouts" — collect states visited by the
  *student*, relabel with the expert, iterate. Our planned self-history round is exactly a
  one-iteration DAgger, and DAgger-style offline rollouts sidestep the transformer
  two-pass problem (rollouts are precomputed, not interleaved with gradient steps).
- **Professor forcing** (Lamb et al., NeurIPS 2016, [arXiv 1610.09038](https://arxiv.org/abs/1610.09038)):
  adversarial training to match teacher-forced and free-running dynamics — RNN-era,
  effective in its setting, little modern use.
- **Degenerate text** (Holtzman et al., [arXiv 1904.09751](https://arxiv.org/abs/1904.09751)):
  likelihood-maximizing decoding produces fluent-but-degenerate, repetitive, low-quality
  text (the "!!!?" / repeated-phrase pathology); nucleus sampling fixes much of it **at
  the decoder, with no retraining**. Our digit soup has the same signature — high-probability
  junk under greedy/over-confident decoding — so decoding hygiene must be ruled out
  before blaming the training objective.

What the duplex/speech systems actually do instead of scheduled sampling:

- **Moshi** never reports scheduled sampling; it stabilizes content with the **inner
  monologue** (time-aligned text tokens predicted as a prefix of audio tokens) and makes
  the user stream robust with noise/echo/reverb augmentation
  ([2410.00037](https://arxiv.org/html/2410.00037v2)).
- **SyncLLM** trains the model to work with *estimated* user chunks that are later
  replaced by real ones — a scheduled-sampling-flavored objective on the input side
  ([2409.15594](https://arxiv.org/html/2409.15594)).
- **Kyutai's interactivity alignment** ([2606.11167](https://arxiv.org/html/2606.11167v1),
  [blog](https://kyutai.org/blog/2026-06-10-interactivity)) and **SALMONN-omni**
  ([2505.17060](https://arxiv.org/abs/2505.17060)) get the last timing gains from **RL
  post-training** on interaction metrics, not from more teacher forcing.
- **TADA** ([Hume AI, code](https://github.com/HumeAI/tada)) is the nearest speech-side
  relative: 1:1 aligned text-audio streams to stop token-level hallucination in
  speech LMs. We found no paper that isolates *word-token emission degeneration in a
  speech LLM's text head* as its subject — our finding would be a small contribution,
  not a retread.

---

## 4. Data-scale reality check: what did timing actually cost?

**Plain-language summary.** Every full-duplex model that handles interruptions well
trained on real, two-channel (one mic per speaker) conversational recordings — the field's
minimum is ~2,000 hours of the Fisher telephone corpus (Moshi, SyncLLM, NTPP, dGSLM).
Synthetic TTS dialogues alone (OmniFlatten: 2,000 synthetic hours) bought assistant-side
timing but not interruption handling. The classic corpora are LDC-licensed (paid,
non-commercial); the permissive options are AMI (CC BY 4.0, 100 h), CANDOR (850 h, by
request from BetterUp), and — new in 2026 — DuplexChat, a 282k-hour English open corpus
mined from podcasts with speaker separation. Our ~1,000 synthetic conversations is 1–3
orders of magnitude below every published timing dataset, which is survivable only
because we train timing tokens on a frozen trunk, not speech generation.

What the models needed for **timing specifically**:

| System | Real conversational audio | Synthetic duplex audio | Verdict on synthetic-only |
|---|---|---|---|
| Moshi ([2410.00037](https://arxiv.org/html/2410.00037v2)) | **2,000 h Fisher** (mandatory finetune stage for real overlap; diarization-masked simulation was not enough) | >20k h multi-stream TTS for instruct style | Needed both |
| SyncLLM ([2409.15594](https://arxiv.org/abs/2409.15594)) | ~2k h Fisher | 212k h | Needed both |
| NTPP ([2506.00975](https://arxiv.org/abs/2506.00975)) | 2,200 h Fisher | — | Real only |
| dGSLM ([2203.16502](https://arxiv.org/abs/2203.16502)) | 2,000 h Fisher | — | Real only |
| OmniFlatten ([2410.17799](https://arxiv.org/html/2410.17799v1)) | 0 | 2,000 h | Assistant timing OK (~160 ms); **user-turn/interrupt weak (~30%)** — authors attribute it to missing real overlap |
| Freeze-Omni ([2411.00774](https://arxiv.org/html/2411.00774v2)) | none reported | 60k TTS'd QA | Duplex works (745 ms interrupt latency) but no published duplex metrics |
| SALMONN-omni ([2505.17060](https://arxiv.org/abs/2505.17060)) | small | — | "Substantially less data" + RL; 30%+ over open baselines |

2026 results with synthetic-heavy or mined data: **DuplexChat**
([arXiv 2607.04941](https://arxiv.org/abs/2607.04941), [data](https://huggingface.co/datasets/sarulab-speech/DuplexChat),
[code](https://github.com/sarulab-speech/DuplexChat)) builds **282,634 h English /
132,723 h Japanese** of speaker-separated, overlap-preserving two-speaker dialogue from
public podcast RSS feeds — by far the largest open corpus of its kind, and it exists
precisely because "existing resources for such data are scarce". Its follow-up
(ConversationalVoice) strengthens separation quality. Kyutai's interactivity-alignment
paper ([2606.11167](https://arxiv.org/html/2606.11167v1)) shows targeted **short audio
segments** per interaction axis (pause / turn-taking / backchannel / interruption) plus RL
move the needle without new web-scale data.

Standard corpora and licensing:

| Corpus | What it is | License reality |
|---|---|---|
| **Fisher** (Parts 1+2) ([LDC2004S13](https://catalog.ldc.upenn.edu/LDC2004S13), [LDC2005S13](https://catalog.ldc.upenn.edu/LDC2005S13)) | ~2,000 h two-channel telephone conversations (975 h in Part 2) | LDC: non-commercial linguistic research/education only; per-dataset fees ([ldc.upenn.edu](https://www.ldc.upenn.edu)) |
| **Switchboard-1** ([LDC97S62](https://catalog.ldc.upenn.edu/LDC97S62)) | ~260 h / 2,400 two-sided calls, 543 speakers | Same LDC terms; non-member fee reported ~$3,000 (ISLRN listing) |
| **CallHome** ([LDC97S42](https://catalog.ldc.upenn.edu/LDC97S42)) | 120 unscripted home phone conversations (+ parallel languages) | Same LDC terms |
| **CANDOR** ([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10065445), [TalkBank host](https://talkbank.org/ca/access/CANDOR.html), [ConvoKit](https://convokit.cornell.edu/documentation/candor.html)) | 850 h / 1,656 dual-channel video-chat conversations, >1 TB, transcripts + vocal-activity measures | Available by request from BetterUp; TalkBank-hosted; third-party turn-taking annotations on HF under CC BY 4.0 "following the CANDOR license" ([HF](https://huggingface.co/datasets/hirotakahiraki/candor-turntaking-annotations)) |
| **AMI** ([official](https://groups.inf.ed.ac.uk/ami/), [HF](https://huggingface.co/datasets/edinburghcstr/ami)) | ~100 h multi-party meetings, per-speaker close-talking mics | **CC BY 4.0** — the most permissive conversational corpus; multi-party (not dyadic telephone register) |
| **DuplexChat** ([2607.04941](https://arxiv.org/abs/2607.04941)) | 415k h speaker-separated podcast dialogue (EN+JA) | Open on Hugging Face (sarulab-speech) |
| **GigaSpeech** ([GitHub](https://github.com/SpeechColab/GigaSpeech), [HF](https://huggingface.co/datasets/speechcolab/gigaspeech)) | 10k h transcribed audio | Non-commercial research; audiobooks/podcasts/YouTube — **not** conversational overlap data |

---

## 5. Serving: MoE speculative decoding and sub-100 ms-frame duplex stacks

**Plain-language summary.** Our problem: one GH200 decodes MoE words at ~80 ms each, so a
word-dense turn blows the 80 ms tick budget ~5×. The serving literature's answer for MoE
models is **speculative decoding** — a cheap "draft" guesses several tokens, the big model
verifies them in one pass. Because MoE models are memory-bandwidth-bound (not
compute-bound), verification is nearly free, so MoEs actually benefit *more* than dense
models: published speedups reach 2.3×, and draft heads now exist for production MoEs.
The published duplex serving stacks do not run decisions faster than ~200 ms cadence;
they win by (a) keeping the sequence GPU-resident and appending in place, (b) special
low-latency MoE kernels, and (c) bounding KV-cache growth. Nobody publishes a sub-100 ms
decision cadence — our 80 ms grid is beyond every published system except as a codec
frame rate.

- **Speculative decoding works well on MoE** — "MoESD / Unveil Speculative Decoding's
  Potential for Accelerating MoE Models" (NeurIPS 2025, [arXiv 2505.19645](https://arxiv.org/abs/2505.19645),
  [poster](https://neurips.cc/virtual/2025/poster/119061)): up to **2.29× on
  Qwen2-57B-A14B**; verifying drafted tokens activates more experts, but since MoE decode
  is bandwidth-bound the extra FLOPs are nearly free; gains hold at small–medium batch.
  **Cohere's engineering write-up** ([blog](https://cohere.com/blog/mixture-of-experts-models-get-more-from-speculative-decoding))
  reaches the same conclusion for production MoEs (bandwidth-bound sweet spots + routing
  correlation between consecutive tokens).
- **Draft heads in production on MoE**: EAGLE-3 ([arXiv 2503.01840](https://arxiv.org/abs/2503.01840))
  heads exist e.g. for MiniMax-M2.5 ([thoughtworks/MiniMax-M2.5-Eagle3 on HF](https://huggingface.co/thoughtworks/MiniMax-M2.5-Eagle3));
  Medusa multi-head drafting reports up to 2.5× ([arXiv 2401.10774](https://arxiv.org/abs/2401.10774)).
  For our exact base, the Qwen3-Omni thinker can run **draft-free n-gram speculative
  decoding** (`ngram_gpu`) in the vLLM production stack (recipe circulated with
  vllm-project; see also the [Qwen3-Omni model card](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Thinking)
  vLLM usage).
- **Qwen3-Omni's own trick**: the Talker uses a **multi-token-prediction (MTP) module**
  to emit several codec tokens per step and streams from the first codec frame —
  234 ms theoretical first-packet latency ([TR 2509.17765](https://arxiv.org/abs/2509.17765));
  Qwen3.5-Omni keeps this design for streaming/high-concurrency serving
  ([TR html](https://arxiv.org/html/2604.15804v1)).
- **vLLM-Omni** ([arXiv 2602.02204](https://arxiv.org/abs/2602.02204), [code](https://github.com/vllm-project/vllm-omni),
  [docs](https://docs.vllm.ai/projects/vllm-omni/en/latest)): a stage-graph abstraction
  that decomposes any-to-any models (thinker → talker → vocoder) into independently
  schedulable, disaggregatable stages — the open-source reference for serving our base
  model family.
- **Thinking Machines' realtime serving blueprint** ([blog](https://thinkingmachines.ai/blog/interaction-models)):
  streaming sessions that **append chunks to a persistent GPU-resident sequence**
  (upstreamed to SGLang), low-latency MoE **gather+gemv kernels**, and **batch-invariant
  kernels (<5% overhead)** so the trainer and the sampler stay bitwise-aligned. The
  interaction loop itself is 200 ms micro-turns; measured FD-Bench V1 turn-taking latency
  0.40 s vs 0.57–1.18 s for gpt-realtime variants.
- **Metronome** ([arXiv 2607.02640](https://arxiv.org/abs/2607.02640), [code](https://github.com/19PINE-AI/metronome)):
  real-time serving of full-duplex models (Moshi/MiniCPM-o/Qwen-Omni class) as a
  *periodic* task whose KV cache otherwise grows without bound — "bound the cache, keep
  the beat"; the systems problem that comes after ours (long duplex sessions).
- **Latency reality check** (published duplex numbers): Moshi 160 ms theoretical / 200 ms
  practice ([2410.00037](https://arxiv.org/abs/2410.00037)); TML-Interaction-Small 0.40 s
  FD-Bench turn-taking ([blog](https://thinkingmachines.ai/blog/interaction-models));
  OmniFlatten assistant turn-taking ~160 ms ([2410.17799](https://arxiv.org/html/2410.17799v1));
  MinMo full-duplex ~600/800 ms ([2501.06282](https://arxiv.org/abs/2501.06282));
  Freeze-Omni 745 ms–1.2 s interrupt-to-audio ([2411.00774](https://arxiv.org/html/2411.00774v2));
  LLaMA-Omni2 ~600 ms ([2505.19826](https://arxiv.org/abs/2505.19826)). Decision
  *cadence*: SyncLLM 160–240 ms chunks ([2409.15594](https://arxiv.org/html/2409.15594)),
  TML 200 ms micro-turns ([blog](https://thinkingmachines.ai/blog/interaction-models)).
  80 ms exists only as codec frame rate (Mimi 12.5 Hz, GLM-4-Voice 12.5 Hz;
  [2410.00037](https://arxiv.org/html/2410.00037v2), [2412.02612](https://arxiv.org/abs/2412.02612)).

---

## WHAT THIS MEANS FOR DUPLEX-LIVE

Ranked recommendations. Each traces to findings above; "us" = current plan
(self-history round, text-word emission + ~10 special tokens, banked-utterance demo,
80 ms tick arrears loop on one GH200).

1. **Run the self-history round, but scope it as offline DAgger, and fix the decoder
   first.** The digit soup has two candidate causes and the cheap one is untested:
   degenerate decoding (greedy/over-confident sampling producing high-probability junk —
   Holtzman's exact symptom, [1904.09751](https://arxiv.org/abs/1904.09751)) vs exposure
   bias (never-trained-on-own-outputs, [1506.03099](https://arxiv.org/abs/1506.03099)).
   Before/alongside retraining: nucleus sampling + logit hygiene (suppress digit
   tokens/repetition at free-run). Then do self-history as one-iteration DAgger —
   precompute free-run rollouts on our ~1k conversations, keep the verified word
   alignments as labels, retrain, iterate ([1011.0686](https://arxiv.org/abs/1011.0686)).
   Precomputed rollouts sidestep the transformer scheduled-sampling awkwardness that made
   online schedules inconsistent ([P19-2040](https://aclanthology.org/P19-2040/)); and
   since word *onsets* (not content) are what our timing loss cares about, even a
   partially-degenerate rollout is usable supervision for the emission tokens.
2. **Keep the hidden-state gate; add a VAP head as the referee and optional lookahead.**
   Freeze-Omni is published proof a hidden-state duplex classifier works
   ([2411.00774](https://arxiv.org/html/2411.00774v2)), so no rewrite is needed. But our
   gate cannot calibrate itself against the field. A VAP model trains self-supervised on
   real two-channel audio in hours ([2205.09812](https://arxiv.org/abs/2205.09812)), runs
   incrementally in real time ([2401.04868](https://arxiv.org/abs/2401.04868)), predicts
   2 s ahead (vs our gate's reactive classification), and is directly gradeable on
   Talking Turns / Full-Duplex-Bench ([2503.01174](https://arxiv.org/abs/2503.01174),
   [2503.04721](https://arxiv.org/abs/2503.04721)). Concretely: train VAP on DuplexChat
   or AMI audio (both open — [2607.04941](https://arxiv.org/abs/2607.04941),
   [AMI](https://groups.inf.ed.ac.uk/ami/)), use it to pick our operating point and to
   distill lookahead into the gate.
3. **Buy real two-channel timing data — the synthetic ceiling is real.** OmniFlatten is
   the controlled experiment: 2,000 synthetic hours got 160 ms assistant turn-taking and
   ~30% user-turn accuracy, because synthetic data has no true interruptions
   ([2410.17799](https://arxiv.org/html/2410.17799v1)). Every system that handles
   barge-in well used ~2k h of Fisher-class real audio (Moshi, SyncLLM, NTPP, dGSLM,
   §4 table). We don't need 2k hours — we need real overlap *statistics* to fine-tune the
   gate/emission layer and to train the VAP referee. Cheapest open path: DuplexChat
   (282k h EN, mined, open) for gate/VAP training; AMI (CC BY 4.0) for a permissive
   baseline; CANDOR or Fisher if license/budget allow.
4. **Fix the tick-budget math before scaling data.** Word-dense turns at ~80 ms/word
   cannot fit an 80 ms tick — this is arithmetic, not a training problem. Ranked options,
   all with precedent: (a) **generate-ahead into a queue** — SyncLLM's core mechanism:
   the model produces its next chunk before the clock needs it, and the tick grid only
   gates *when audio is released* ([2409.15594](https://arxiv.org/html/2409.15594));
   timing tokens (`<sil>`, `<speak>`, `<yield>`) stay on the tick grid, words don't.
   (b) **Speculative decoding** on the thinker: MoE-friendly, up to 2.29×
   ([2505.19645](https://arxiv.org/abs/2505.19645)), with a draft-free `ngram_gpu`
   recipe for Qwen3-Omni specifically (vLLM stack, §5). (c) **MTP-style multi-word
   emission** per forward pass — exactly what Qwen3-Omni's Talker does for codec tokens
   ([2509.17765](https://arxiv.org/abs/2509.17765)). Option (a) is architectural and
   cheapest for us; (b) is a config change; do both.
5. **Re-examine the 80 ms decision cadence.** Published duplex decision cadences are
   160–240 ms (SyncLLM) and 200 ms (TML micro-turns); 80 ms appears only as codec frame
   rate (§5 latency table). A 160 ms grid halves trunk passes and still beats every
   published end-to-end latency except Moshi's theoretical 160 ms. Keep 80 ms for the
   audio *input* grid (matching Mimi/our ear), but consider 160 ms for decisions —
   especially since our measured 59 ms decisions show we have headroom on the trunk, and
   the word-token path is the real bottleneck.
6. **Adopt Full-Duplex-Bench metrics as our external scoreboard.** Our current numbers
   (recall ~0.9, no interruptions, onsets within 1 s) are home-grown. FDB v1/v1.5 give
   scenario-driven pause/backchannel/turn-taking/interruption + overlap metrics with
   stop/response latency ([2503.04721](https://arxiv.org/abs/2503.04721),
   [2507.23159](https://arxiv.org/abs/2507.23159), [code](https://github.com/DanielLin94144/Full-Duplex-Bench)).
   Running it (or its metric definitions) against our live loop turns "works well" into a
   comparable number, and it is the benchmark TML used for their 0.40 s claim — the
   yardstick our M2 will be judged against.
7. **Wire the Qwen3-Omni Talker for M2 instead of banked utterances.** Banked utterances
   are fine for the demo, but the ceiling is real: no mid-word interruption recovery, no
   prosodic backchannels, and the arrears loop can't duck/undo a banked clip. The Talker
   is already a streaming codec head with MTP and 234 ms theoretical first-packet
   ([2509.17765](https://arxiv.org/abs/2509.17765)), vLLM-Omni exists to serve it
   ([2602.02204](https://arxiv.org/abs/2602.02204)), and GLM-4-Voice shows the
   12.5 Hz/80 ms text-then-codec pattern working ([2412.02612](https://arxiv.org/abs/2412.02612)).
   Moshi's inner-monologue result — text tokens as a prefix stabilizing audio generation
   ([2410.00037](https://arxiv.org/html/2410.00037v2)) — is the direct argument that our
   text-word emission layer is the right thing to feed it.
8. **After self-history works, budget an RL timing round (M3), not more teacher forcing.**
   The last-mile timing gains in the field came from RL post-training on interaction
   metrics: Kyutai's interactivity alignment on four axes with short targeted segments
   ([2606.11167](https://arxiv.org/html/2606.11167v1)), SALMONN-omni's RL stage
   ([2505.17060](https://arxiv.org/abs/2505.17060)). Our setting is unusually RL-friendly
   — frozen trunk, discrete emission tokens, cheap whisper-verified reward (onset error,
   interruptions, backchannel timing) — and it attacks exactly the free-run regime where
   teacher forcing cannot supervise.

---

### Source index (all fetched 2026-10-07)

Moshi [abs](https://arxiv.org/abs/2410.00037) · [html](https://arxiv.org/html/2410.00037v2) · [code](https://github.com/kyutai-labs/moshi) —
dGSLM [abs](https://arxiv.org/abs/2203.16502) —
SyncLLM [abs](https://arxiv.org/abs/2409.15594) · [html](https://arxiv.org/html/2409.15594) —
Freeze-Omni [abs](https://arxiv.org/abs/2411.00774) · [html](https://arxiv.org/html/2411.00774v2) · [code](https://github.com/VITA-MLLM/Freeze-Omni) —
LLaMA-Omni [abs](https://arxiv.org/abs/2409.06666) · [code](https://github.com/ictnlp/LLaMA-Omni) —
LLaMA-Omni2 [abs](https://arxiv.org/abs/2505.19826) —
Mini-Omni [abs](https://arxiv.org/abs/2406.14275) / Mini-Omni2 [abs](https://arxiv.org/abs/2410.11190) · [code](https://github.com/gpt-omni/mini-omni2) —
GLM-4-Voice [abs](https://arxiv.org/abs/2412.02612) · [code](https://github.com/THUDM/GLM-4-Voice) —
VITA [abs](https://arxiv.org/abs/2408.05211) —
MiniCPM-o 2.6 [card](https://huggingface.co/openbmb/MiniCPM-o-2_6) · [report](https://openbmb.vercel.app/minicpm-o-2-6-en) · [code](https://github.com/OpenBMB/MiniCPM-V) —
MiniCPM-o 4.5 [abs](https://arxiv.org/abs/2604.27393) —
MinMo [abs](https://arxiv.org/abs/2501.06282) —
OmniFlatten [abs](https://arxiv.org/abs/2410.17799) · [html](https://arxiv.org/html/2410.17799v1) —
Parrot [OpenReview](https://openreview.net/forum?id=73EDGbG6mB) — NTPP [abs](https://arxiv.org/abs/2506.00975) —
Full-Duplex-1 [2405.19487](https://arxiv.org/abs/2405.19487) / TDM [2406.15718](https://arxiv.org/abs/2406.15718) —
LSLM [abs](https://arxiv.org/abs/2408.02622) —
SALMONN-omni [abs](https://arxiv.org/abs/2505.17060) · [code](https://github.com/bytedance/SALMONN) —
SALM-Duplex [abs](https://arxiv.org/abs/2505.15670) —
Qwen3-Omni TR [abs](https://arxiv.org/abs/2509.17765) · [card](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Thinking) — Qwen3.5-Omni TR [html](https://arxiv.org/html/2604.15804v1) —
Thinking Machines [interaction models](https://thinkingmachines.ai/blog/interaction-models) —
OpenAI Realtime [guide](https://developers.openai.com/api/docs/guides/realtime) · [VAD](https://developers.openai.com/api/docs/guides/realtime-vad) —
Kyutai interactivity alignment [html](https://arxiv.org/html/2606.11167v1) · [blog](https://kyutai.org/blog/2026-06-10-interactivity) · [samples](https://huggingface.co/datasets/kyutai/interactivity-alignment-samples) —
MoshiRAG [abs](https://arxiv.org/abs/2604.12928) · [Kyutai papers page](https://kyutai.org/papers/) —
VAP [abs](https://arxiv.org/abs/2205.09812) · [project](https://erikekstedt.github.io/VAP/) · [code](https://github.com/ErikEkstedt/VoiceActivityProjection) —
VAP real-time [abs](https://arxiv.org/abs/2401.04868) · [code](https://github.com/inokoj/VAP-Realtime) — multilingual VAP [abs](https://arxiv.org/abs/2403.06487) —
Talking Turns [abs](https://arxiv.org/abs/2503.01174) —
Full-Duplex-Bench v1 [abs](https://arxiv.org/abs/2503.04721) · v1.5 [abs](https://arxiv.org/abs/2507.23159) · v2 [abs](https://arxiv.org/abs/2510.07838) · v3 [abs](https://arxiv.org/abs/2604.04847) · [code](https://github.com/DanielLin94144/Full-Duplex-Bench) — TurnBench [abs](https://arxiv.org/abs/2608.25218) —
Scheduled sampling [abs](https://arxiv.org/abs/1506.03099) · transformers variant [P19-2040](https://aclanthology.org/P19-2040/) · exposure-bias critique [D19-5616](https://aclanthology.org/D19-5616/) · self-recovery [abs](https://arxiv.org/abs/1905.10617) · DAgger [abs](https://arxiv.org/abs/1011.0686) · professor forcing [abs](https://arxiv.org/abs/1610.09038) · neural text degeneration [abs](https://arxiv.org/abs/1904.09751) —
TADA [code](https://github.com/HumeAI/tada) —
DuplexChat [abs](https://arxiv.org/abs/2607.04941) · [data](https://huggingface.co/datasets/sarulab-speech/DuplexChat) · [code](https://github.com/sarulab-speech/DuplexChat) —
Fisher [LDC2004S13](https://catalog.ldc.upenn.edu/LDC2004S13) · [LDC2005S13](https://catalog.ldc.upenn.edu/LDC2005S13) — Switchboard [LDC97S62](https://catalog.ldc.upenn.edu/LDC97S62) — CallHome [LDC97S42](https://catalog.ldc.upenn.edu/LDC97S42) — LDC licensing [ldc.upenn.edu](https://www.ldc.upenn.edu) —
CANDOR [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10065445) · [TalkBank](https://talkbank.org/ca/access/CANDOR.html) · [ConvoKit](https://convokit.cornell.edu/documentation/candor.html) · [HF annotations](https://huggingface.co/datasets/hirotakahiraki/candor-turntaking-annotations) —
AMI [official](https://groups.inf.ed.ac.uk/ami/) · [HF](https://huggingface.co/datasets/edinburghcstr/ami) — GigaSpeech [code](https://github.com/SpeechColab/GigaSpeech) · [HF](https://huggingface.co/datasets/speechcolab/gigaspeech) —
MoESD [abs](https://arxiv.org/abs/2505.19645) · [NeurIPS](https://neurips.cc/virtual/2025/poster/119061) — Cohere [blog](https://cohere.com/blog/mixture-of-experts-models-get-more-from-speculative-decoding) —
EAGLE-3 [abs](https://arxiv.org/abs/2503.01840) · [MiniMax-M2.5 head](https://huggingface.co/thoughtworks/MiniMax-M2.5-Eagle3) — Medusa [abs](https://arxiv.org/abs/2401.10774) —
vLLM-Omni [abs](https://arxiv.org/abs/2602.02204) · [code](https://github.com/vllm-project/vllm-omni) · [docs](https://docs.vllm.ai/projects/vllm-omni/en/latest) —
Metronome [abs](https://arxiv.org/abs/2607.02640) · [code](https://github.com/19PINE-AI/metronome) —
Awesome-Full-Duplex-SDM [repo](https://github.com/Ruiqi-Yan/Awesome-Full-Duplex-SDM)
