# Oruk findings and the capacity, knowledge and RL branches

12 November 2026. Owner requests that these recommendations live alongside the wider plan; OneDrive editing is now explicitly permitted again. Work remains in the recovered temp checkout for this turn, preserving unrelated OneDrive work. No new GPU allocation, API call, model purchase or weight change is made by this note.

## Capacity is a real hypothesis, not ruled out

A model can lack representational capacity for knowledge, contextual reasoning and interaction decisions. More parameters can help, but cannot by themselves add missing knowledge, correct a reward, deliver a cancelled task safely, or make GPU/network execution faster. Added dense layers also consume latency and memory. More trainable existing weights (full fine-tuning) and more total model weights (structural growth) are different interventions.

Our 698M Venus adapter established actual learning and restoreability, while its cited native speech checks remained unqualified. The completed Moshi-family 1.89M/105M adapter experiment measures adaptation capacity/coverage; it does not compare 7B versus 9B native backbones. Its trained-candidate evaluation is still incomplete, so neither unlimited adapter sufficiency nor a total-parameter bottleneck has been established.

Keep three substantive training arms in the next programme design: a broad adapter control, selective/full existing native-LM tuning, and an initially function-preserving block-growth candidate (roughly +0.5–1.5B as a planning range, exact tensors to be counted). Run a matched growth arm once the chosen foundation, qualified data and distributed implementation are ready; do not postpone it indefinitely behind unrelated repairs. Compare data exposure and report both equal-exposure results and actual GPU cost/latency. A larger arm's lower TRAIN loss alone is not proof of better reasoning or product quality. Inspect unseen tasks, ordinary retention, speech, control and real live latency.

If a fixed-size model underfits a verified broad corpus after a reasonable recipe search and larger capacity improves unseen competence, continue the expanded branch. If full tuning resolves the deficit without growth, prefer the measured better product. If failures originate in perception, missing examples, runtime state or rewards, repair those rather than attributing all failures to size.

## Pretraining versus continued pretraining

A capacity limit does not require discarding learned weights. Expand a pretrained model and train the new capacity, or use a stronger pretrained backbone with explicit speech alignment work. Full from-scratch pretraining becomes justified only if reuse/expansion cannot meet the architecture or capability needs and a real licensed corpus, throughput/token budget, speech-alignment plan and comparative benefit exist. The 20k GPU-hour allocation supports substantial experiments; it is not itself evidence that starting over is best.

Continued pretraining/mid-training is appropriate for a measured knowledge deficit: verified general/math/technical material plus rehearsal, followed by native conversational alignment and RL. Preserve the native speech/control objectives and evaluate drift. The prepared 1008 text TRAIN examples are a seed, not a pretraining corpus. Neither a text-only loss improvement nor the Chinchilla token/parameter rule is a native speech-quality guarantee.

Primary support: [continued pretraining with replay and LR re-warming/re-decay](https://arxiv.org/abs/2403.08763); [LLaMA Pro block expansion](https://arxiv.org/abs/2401.02415); [Net2Net function-preserving growth](https://arxiv.org/abs/1511.05641); [LoRA versus full tuning](https://thinkingmachines.ai/blog/lora/). These establish options in their tested settings, not Moshi streaming/KV/codec equivalence.

## Oruk findings worth using

Reviewed the [research index](https://oruk.ai/research), [OrukQwen RL study](https://oruk.ai/research/orukqwen-rl), its [public model card](https://huggingface.co/oruk/orukqwen-asr-1.7b), [JEV-Speech runtime study](https://oruk.ai/research/jev-speech) and [same-model runtime study](https://oruk.ai/research/faster-speech-without-skipping-the-evidence). These are author-reported ASR/runtime experiments, not a native duplex generator or proof of improved dialogue reasoning.

The RL study reports that aligned token credit did not establish a consistent advantage over plain RL across its controls. Transcript convention changed the apparent reward outcome. Our transfer: begin with a clear baseline native GRPO objective, retain correctness/style/timing metrics separately, then ablate credit assignment only if diagnosed. Preserve acknowledgements when appropriate and penalize them during explicit silence; globally stripping fillers from an ASR reward cannot certify natural conversation.

The public card describes a compatible-base RL soup blended back toward the base. Our transfer: preserve reference/retention controls and consider same-base interpolation or checkpoint distillation only as separately tested candidates. Do not average Venus with Moshi or assume a 50/50 blend is optimal for us. The later experimental results and some source material are not public releases; avoid treating selected exploratory gains as untouched-test evidence.

The runtime studies separate execution optimizations from weight/precision changes. Our transfer: benchmark the same weights, decoder, inputs and hardware before a speed claim; include cold/warm behavior, streaming state, retained output buffers and tail latency. Do not equate a warm ASR speedup with end-to-end native voice latency. No Oruk API or model has been installed or substituted into the fixed evaluation.

## RL action

Use the best coherent compatible seed from the full comparison. RL need not wait for every behaviour to pass. Train verifiable basic answers, calibrated direct/clarify/delegate selection, real worker completion/cancellation and waveform-grounded silence/release/interruption. Retain broad SFT rehearsal and semantic/speech quality; then add MiRL-style targeting of repeated TRAIN errors. [Published Moshi/PersonaPlex interaction RL](https://arxiv.org/html/2606.11167v2) is the more direct model-family evidence. OPAL controls recorded recipe changes; it does not replace task rewards or native rollout correctness.

## Immediate execution

All four trained196 states are backed up. Evaluation8352856 failed after both baseline32 collections/audits and partial narrow collections; no broad candidate outputs. Guard snapshot has local stamp1794503024.4430718 and original provider stamp1794502936.5135705, age87.9295seconds, expiry1794503026.5135705 (provider +90). Storage observation1794503029.1385953 remained fresh. Watcher rejected missing/future/expired evidence; the original provider timeout/retry event is not retained in these receipts, so exact query cause remains unproven. Guard failure does not establish model failure.

Prepare a bounded evaluation-only recovery with preserved existing outputs, strict complete192 coverage, actual accounting and more informative failure receipts; do not replay training or silently widen evidence TTLs. Current max10-start grant is exhausted even though GPU-hour budget remains. A new running interval needs a concrete scope amendment. Offline repair and proposal preparation continue under existing authority. This note adds no runtime permission.
