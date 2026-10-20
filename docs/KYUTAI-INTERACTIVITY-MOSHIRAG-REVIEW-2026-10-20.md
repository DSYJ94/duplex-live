# Kyutai interactivity and MoshiRAG: relevance to duplex-live

Verified 20 October 2026 from the official posts, papers, release metadata and implementation. Research only: no model substitution, dataset import, optimizer update, or extra cluster submission was performed for this review.

Both are useful. The June work supplies a better interaction-level training and evaluation recipe for silence, pauses and interruptions. The April work supplies a released native asynchronous knowledge interface and a clean way to separate retrieval failures from speech-model failures. Neither establishes a fix for our measured BF16 streaming/batched discrepancy, nor qualifies our current candidate for extension or adoption.

## June: Multi-Faceted Interactivity Alignment

[Official post](https://kyutai.org/blog/2026-06-10-interactivity/) · [Paper, arXiv:2606.11167v1](https://arxiv.org/html/2606.11167v1)

The paper applies GRPO to sampled full-duplex trajectories, with separate pause, turn-taking, backchannel and interruption rewards. The policy objective uses **text-token probabilities**, excluding audio-token probabilities, and retains reference-policy KL. Semantic and delay rewards are normalized independently before combination. This is materially different from matching one teacher token at an exact clock. Appendix A reports 32 H100s, 16 completions per group, 100 epochs and initial LR 2e-7; reproducing it is a substantial new workflow, not another small continuation of our existing supervised job. It also reports a corrected 16 kHz VAD resampling bug in benchmark backchannel evaluation. [Paper §§3.2–3.4, Appendix A/B](https://arxiv.org/html/2606.11167v1)

The official post reports these PersonaPlex changes after Seamless RL:

| Metric | Base | RL | Direction |
|---|---:|---:|---|
| Pause inappropriate takeover rate | 0.444 | 0.356 | Lower |
| Backchannel timing JSD | 0.841 | 0.786 | Lower |
| Turn-taking takeover rate | 0.958 | 0.975 | Higher |
| Post-interruption semantic score | 4.500 | 4.533 | Higher |

These are external benchmark results, not our native ordinary4, explicit quiet/pause, or delegation qualification. [Official results](https://kyutai.org/blog/2026-06-10-interactivity/)

**Actual release:** [personaplex-rl-seamless](https://huggingface.co/kyutai/personaplex-rl-seamless) is based on `nvidia/personaplex-7b-v1` and documented as compatible with official PersonaPlex inference, including text/voice prompts. Its listed weights are 16.7 GB, but access is gated. The card requires **CC BY-NC 4.0 plus NVIDIA Open Model License** and labels it research-only. It is an interesting isolated comparison checkpoint, not an automatic production upgrade or a safe base substitution beneath our trained adapters. Public API revision checked: `3fa800309a4b743a8a6d764253eb45def0334afc`. [Model card and conditions](https://huggingface.co/kyutai/personaplex-rl-seamless), [file inventory](https://huggingface.co/kyutai/personaplex-rl-seamless/tree/main)

The companion [moshika-rl-seamless](https://huggingface.co/kyutai/moshika-rl-seamless) is also gated and CC BY-NC 4.0. The PersonaPlex card says the training corpus is 4,000-hour Seamless Interaction, from which up to 2,000 segments per axis were extracted. [Seamless official repository](https://github.com/facebookresearch/seamless_interaction) is the upstream data route; no wholesale corpus download is needed for our current diagnostic.

**Important dataset distinction:** [interactivity-alignment-samples](https://huggingface.co/datasets/kyutai/interactivity-alignment-samples) contains **generated evaluation samples**, not the extracted GRPO training set. It has about 5.56k stereo WAV rows: input/examiner on channel 0 and model output on channel 1, covering static v1 and dynamic v2. The card labels these samples CC BY 4.0. Revision checked: `aed53d86c0141245ff72775d4d0cb78e644b5f03`. It is useful for listening and measurement reproduction; treating its model outputs as approved teachers would contaminate the evidence boundary. [Dataset card](https://huggingface.co/datasets/kyutai/interactivity-alignment-samples)

I found **no released GRPO trainer or extracted per-axis training set linked by the paper/model cards**, and no dedicated interactivity trainer repository in the public Kyutai organization listing inspected. That is a bounded release inspection, not proof no such implementation exists elsewhere. Their cards direct inference to existing Moshi/PersonaPlex code. Do not equate published hyperparameters with a runnable released training pipeline.

## April: MoshiRAG

[Official post](https://kyutai.org/blog/2026-04-30-moshi-rag/) · [Paper, arXiv:2604.12928v2](https://arxiv.org/html/2604.12928v2) · [ICML publication](https://proceedings.mlr.press/v306/chien26b.html)

Moshi emits a retrieval token, keeps listening/speaking, and receives compressed reference embeddings through an ARC-Encoder path. The paper trains on approximately 1.9M synthetic conversations, with randomized delivery delays and reference dropout; all model parameters except the encoder are trainable, for 100k updates. **Reference accuracy and spoken-answer accuracy are separate columns.** With Gemma 3 27B, actual spoken answers score 80.3/67.2/69.6/36.3% on LlamaQ/WebQ/TriviaQA/HaluEval, versus vanilla Moshi 62.3/26.6/22.8/10.5%. The higher 83.0/71.5/73.7/42.0 figures measure references, not answers. GPT-4.1/Tavily experiments assume fixed 1.5-second retrieval because API timing was unstable; those are not measured live latency guarantees. [Paper §§3–5, Table 1](https://arxiv.org/html/2604.12928v2)

**Actual code/weights:** [moshi-rag](https://github.com/kyutai-labs/moshi-rag) was inspected at `8c6dfc101b7871baa428424bcdc583b74fb561d9`. It includes PyTorch/Rust inference, reference encoder service and evaluation tooling. Its recursive inventory contains no training/data-generation pipeline; do not assume the 1.9M training corpus is released. Python/client are MIT, Rust Apache, weights CC BY 4.0. The README warns response quality suffers when retrieval exceeds roughly three seconds. [Pinned README](https://github.com/kyutai-labs/moshi-rag/blob/8c6dfc101b7871baa428424bcdc583b74fb561d9/README.md)

Released [PyTorch BF16 weights](https://huggingface.co/kyutai/moshika-rag-pytorch-bf16/tree/main) include a 15.4 GB model plus codec/tokenizer, at checked revision `7135a6e3c46abb66c2cd95cb04cbfcbe8376f83d`. The [configuration](https://github.com/kyutai-labs/moshi-rag/blob/8c6dfc101b7871baa428424bcdc583b74fb561d9/moshi/config-rag.json) specifies `rag_token_id=4`, frozen `kyutai/ARC4_Encoder_Llama`, a learned 3072→4096 bridge and `streaming_sum` conditioning. Our Context Spanning path instead inserts token/audio Context Spans. These are different trained interfaces; adding the encoder or copying weights into our current adapter model is not an established compatible repair.

**Actual evaluation data:** [HaluEvalAudio_1000](https://huggingface.co/datasets/kyutai/HaluEvalAudio_1000) is a 1,000-row **test** set, with synthesized questions, transcripts, knowledge passages and answers. Its oracle-reference versus end-to-end setups can distinguish retrieval failure from integration failure. Audio is CC BY-NC 4.0; source text MIT. Checked revision `82a88135fbda632d5739ffeacfc794365f10e6bf`. The Hugging Face “models trained on this dataset” presentation does not override the dataset's documented evaluation purpose. Keep it out of TRAIN. [Dataset construction and licensing](https://huggingface.co/datasets/kyutai/HaluEvalAudio_1000#dataset-construction)

**Delegation reliability:** the inspected [RAGManager](https://github.com/kyutai-labs/moshi-rag/blob/8c6dfc101b7871baa428424bcdc583b74fb561d9/moshi/moshi/inference_utils/rag_manager.py) owns one pending task per channel, cancels/awaits it before a new trigger and at exit, and supports timeout/reset. However, timeout and generic failures return empty reference strings. [Channel handling](https://github.com/kyutai-labs/moshi-rag/blob/8c6dfc101b7871baa428424bcdc583b74fb561d9/moshi/moshi/inference_utils/channel.py) displays `[RET_FAILED]`, then launches a separate ARC-encoding task that applies its result to the slot. That path has no checked request-generation ID before applying the encoded result. This static inspection does not demonstrate a race in execution, but it also does not establish stale-result protection across overlapping encoding requests. No evidence found here establishes user-requested pause/resume, explicit cancellation semantics, typed tool errors, durable request receipts or exactly-once side effects. Generic task cancellation is insufficient evidence for our delegation gate.

## What helps our current work

Our pinned [Context Spanning README](../runs/cloud-export-20261016/tree/vendor/context-spanning-79df302fa2820cae10c61bb2b7f302c52edb6ed2/README.md) already cites MoshiRAG-sampled training delays and evaluates MoshiRAG as a separate baseline. April's architecture is therefore related prior work, rather than an overlooked turnkey module.

Our [complete49 numerical review](NATIVE-BF16-STREAM49-RECOVERY1-COMPLETE-REVIEW-2026-10-20.json) measured 36 valid-token argmax disagreements, including 9 spoken tokens, across 22,992 valid targets; all49 exceed the original 0.001 logit tolerance despite unchanged parameters. June's discussion of exposure bias concerns different generated histories at inference. It does **not** explain away arithmetic differences for matched inputs. Neither reviewed paper supplies a verified TF32/KV-cache/BF16 parity correction. Our [complete-batch child review](NATIVE-COMPLETE-BATCH-COMPLETE10-REVIEW-2026-10-20.json) remains rejected; external papers do not change that outcome.

Recommended order:

1. Finish the current same-checkpoint native numerical comparison and preserve all results. Do not extend the rejected child unchanged or substitute a newly released base beneath its adapters.
2. Use June's four-axis, waveform-level measurements as **additional diagnostics**, keeping exact explicit-silence and pause/resume task gates. In particular, a backchannel permitted by an interactivity metric must still fail an explicit “stay silent” instruction.
3. Add an oracle-versus-real-backend split when diagnosing delegation: first establish that the native model correctly uses a valid delayed result, then evaluate actual retrieval/ASR/tool delivery. Preserve request IDs, receipt causality, cancellation and late-result rejection in our own protocol.
4. Consider a separately versioned PersonaPlex-RL comparison only after its access/license conditions and runtime compatibility are established. Treat it as a comparator, not adopted production capability. A new GRPO training implementation is follow-on work after trustworthy native numerics and state restoration; it is not the fastest justified use of the remaining allocation.
