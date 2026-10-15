# Native Lychee continuation

## Current execution — 15 October, native Lychee and actual audio

Plain words: no complete model qualifies. Continue the owner's authorised work
until actual input is needed; the allocation expires in about six days. The
accepted development voices are PersonaPlex synthetic marin and Lychee. Use
[docs/LYCHEE_NEXT_STAGE.md](docs/LYCHEE_NEXT_STAGE.md) for the next actions.

The released Qwen-based Lychee model runs through its native custom-vLLM backend
with all 982 checkpoint tensors accounted for. Warm-input baseline 7451709 keeps
maximum input backlog to 18.5ms; every text, speech and control token equals the
original 7441742 baseline. Concurrent two-GPU capture 7452017 completes 70s of
paced input in 70.540s with p95 model compute 299ms/400ms, maximum backlog 72.7ms,
1258 codec units accounted for and zero within-utterance gaps under fixed 240ms
per-utterance causal playout. It actually generates LM tokens and PCM together;
the final WAV simulates playout. Browser/audio-device/microphone use is untested.
The baseline still makes factual errors and is not a qualified voice assistant.

Static fact controls 7450548/50 say both private values. Actual late-return
7451814/15 and idle/quiet delivery 7451816/17 omit both values despite verified
fresh prefixes at the actual engine request boundary. Quiet delivery adds about
20s and is rejected. More prompt/timing-only variants are not the next step.

Native capture 7452224 and HF comparison 7452225 compare all three raw heads
at three fresh prefills using identical embeddings and correct next-text merge
conditioning. All nine raw top-one predictions match; RMS logit differences are
0.0379–0.1219, KL 0.000153–0.00863. This is evidence of similar graph behavior,
not numerical identity, actual masked control agreement or incremental-KV proof.

Next: native S3 round trips and independent speech audits of the 48 approved
TRAIN scenes; causal result-ready training views with actual student history;
then a bounded native semantic-adapter proof before training scale-up. Persona
adaptations remain failed references. All source data remains immutable, no
held-out content enters training, and no Lychee optimizer run exists yet.

Code moves through Git only. New jobs use clean immutable worktrees and exact
commits. Fetch full logs before local filtering. Preserve unrelated pyc, roadmap
and cluster files. Earlier notes below record historical stages and may be stale.


## Native training design

Plain words: use the pretrained duplex model; do not pretrain a new foundation
model or revive the old Qwen row model. The current failure is late result use.
Training must include actual student history, including its prior refusal, and
actual pending/completed client context. Teacher-only history has already failed
to transfer in the Persona experiments.

The 28-layer text backbone forks into the four-layer speech branch after layer
24 and the control branch after layer 20. A first semantic adapter can affect
only main layers 24–27, leaving raw speech/control branch outputs unchanged for
identical inputs. Text-selected merge conditioning can still produce different
speech naturally when words change. This preservation needs an actual native
zero/update/reload proof; it is not yet implemented or a task-success claim.

Re-encode only verified TRAIN speech with the actual pinned 25Hz S3 tokenizer
and accepted vocoder. Keep measured source words, source hashes and real worker
receipt times. Independently transcribe each round trip under the unchanged
0.08 WER bar. A codec pass is not native training approval. Do not silently
accept missing words, truncated speech, invalid units or failed source stamps.

The 200ms first-subword check in recovery_encode is for the Moshi interleaver.
Lychee's published layout emits text ahead of delayed speech, and cannot be
called a pass on that Moshi check. Source word onsets still need independent
verification. Measure Lychee's actual text/audio relationship explicitly. In
particular, a private fact's first text token must not precede the actual worker
receipt, even if the audible word occurs after it. Native training views must
mask earlier loss and ensure causality at the native text token, not merely the
audio word. Preserve all real user audio and report any declared PCM transforms.

## Audio and runtime reproduction

Plain words: the concurrent recording is available at
http://localhost:8766/audio/voice-diagnostics/index.html#concurrent . The owner
has not rated this new capture yet; the accepted older audition remains visible.
It uses hop8, lookahead3, decoder batch1 and fixed 240ms per-utterance playout.
1258 units = 1240 consumed + 18 cancelled before vocoding. The final playout
retains 1,156,285 PCM samples and cancels 26,435 before playback at actual received
interruption controls. Local socket p95 transport delay is 0.079ms; maximum is
79.97ms. Source token arrays exactly equal the original standalone baseline.

The old paced-report vocoder_mode text is hardcoded to offline in 7452017's LM
subreport. The concurrent receiver/parent evidence is authoritative for this
run; subsequent code labels external event transport separately. Keep the old
artifact immutable. Full-transcript audit 7450830 of earlier recordings reports
WER 0.2763 and is not a pass; cancelled tails and number formatting remain to be
separated from actual text/audio disagreement.

Native vLLM runtime: envs/lychee-vllm-7441739 with CPU dependencies
lychee-vllm-deps-7441737. Separate vocoder overlay lychee-probe-7415053 plus
minicpm-recovery-7377673. Exclusive_Process requires two separately assigned
GPUs for concurrent processes. Never overwrite the container Torch or the
accepted overlays. lychee-concurrent.slurm enforces immutable job source.


**Plain words — native data stage completed:** S3 conversion 7452500 reads
48 approved TRAIN scenarios and creates 125 measured speech segments. Independent
round-trip audit 7452502 accepts 106 under WER <=0.08 and rejects 19; no threshold
is relaxed. These are codec-intelligibility results, not approved native training
windows or task success. Keep failed segments excluded. Native model histories
will be collected using actual workers and sampled control from approved private
TRAIN scenes, preserving both the ready moment and the first subsequent quiet
native listening boundary. Snapshots retain all actual sampled body tokens,
arrived audio features/embeddings/tail and actual receipt clocks. They contain
no gold future labels yet. Validation/test history is rejected before tensor
access; future/unreceived results are rejected. State collection invalidates
latency qualification. Two collection-boundary regressions pass.
