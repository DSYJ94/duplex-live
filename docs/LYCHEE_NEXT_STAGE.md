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

Native S3 conversion 7452500 yields 125 segments from 48 approved TRAIN scenes;
audit 7452502 accepts 106 and excludes 19. Actual student histories 7452624–31
support seven causal native semantic examples (7452831), with all past loss
masked and private text targets after actual results. The eighth lacks exact
native gold transcript agreement and is excluded.

First semantic proofs 7452899/7452921 reject NaN gradients before updates on the
default attention path. Matched no-cuDNN 7452937 and math 7452938 pass eight
updates: 2,883,584 trainable parameters in main layers 24–27, unchanged frozen
hashes, zero same-input speech/control drift and exact fresh-adapter reload.
Checkpoint 7452937 SHA 5c238a774042eee2596350b53fd04afd83ea689b8ed470475001e7636adf7965.
Paired TRAIN histories 7452946–58 and causal assembly 7453119 complete
15 usable examples. Both 64/128-step pilots 7453182/83 preserve frozen branches
but their cold actual-worker replays fail: they invent room temperatures, and
one value echo describes lightning instead of the record. Ordinary answers also
drift. Do not scale this recipe. Fixed-state TRAIN continuations 7453748–50
now show exact native input IDs and embeddings (maximum error 0). Both adapters
return all three supplied private values correctly; base returns one of three.
These teacher-started TRAIN component checks do not qualify live behavior.

Next diagnosis keeps the base weights during ordinary conversation, activating
the checked adapter only for one application-started result reply after a real
receipt, native listening state and two arrived quiet chunks. Every switch
aborts the old KV request, preserves history/audio/clock/RNG and rebuilds it.
Disabled-adapter 7454004 now reproduces every baseline token exactly.
Actual-worker 7454007/09/10/11 produce the correct room-marker sentence in all
four counterfactual replays, but initial access denial and 14.8–16.4s activation
delay remain. Next: concurrent two-GPU audio plus independent speech audit. This is a disclosed application policy, not
learned turn choice or delegation. Keep held-out data excluded from training.

The owner listened to concurrent capture 7452017 and said "Sounds smooth" on
15 October. Retain it as the development audio baseline; factual accuracy,
worker behavior and microphone acceptance remain separate unresolved gates.

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


**Plain words — actual student history ready for native semantic assembly:**
7452624–31 complete all eight approved private TRAIN scenes. Every run stores an
actual completed-result snapshot and a later native listening snapshot, with
all sampled student history and received audio preserved. All omit the value;
these are useful error contexts, not successful candidates. Native replay now
captures the actual complete prefill on the next arrived user chunk, then
appends a verified gold reply only where all remaining real user audio is quiet.
The quiet extension is an explicit training augmentation, never an inference
result. Native S-S/detect/sleep/K-S/S-L grammar and all S3 units must round-trip
through the native decoder exactly. Only future words and EOT receive loss;
all old history and speech/control labels are masked. Every first target is
after the actual receipt and the snapshot clock. Gold speech must have exact
independent transcript agreement (stronger than the 0.08 corpus bar) and contain
the actual returned value. Missing or failed gold segments remain excluded.
A layout regression checks future timing, native boundaries and codec retention.
Native GPU assembly and an adapter gradient/preservation/reload proof are next.


**Plain words — native semantic inputs passed:** 7452831 builds seven causal
TRAIN examples from the actual collected native listening histories. The eighth
source has no exactly transcribed native gold reply and stays excluded. Actual
prefills contain 759–979 tokens; 8–9 future text/EOT targets per example begin at
25.2–34.8s, after the corresponding real worker returns at 8.8–11.6s. Native
word/code/control decoding reproduces each gold reply and all codec units exactly.
The quiet future is explicitly an augmentation, not an inference result. No
optimizer ran in this assembly.

A bounded eight-update semantic LoRA proof is prepared on main layers 24–27
only, after speech/control fork at 24/20. It freezes every base parameter and
uses only future semantic text loss on approved native inputs. It must establish
zero-adapter identity, preserved CPU/CUDA construction RNG, finite nonzero
gradients in all intended tensors, identical frozen parameter hashes, unchanged
raw speech/control hidden outputs for the same input, changed text logits and
exact fresh-adapter reload. Every data consumer revalidates source quality and
hashes of native units, independent audits and actual student histories. A pass
would prove implementation only, not task performance or merged native deployment.


**Plain words — eight updates do not fix late results:** cold native
7452981/83 (model timing) and 7452984/85 (one disclosed response event) still
omit yellow circle and purple diamond. All native merges and strict loading
complete; wording changes but no task improvement is claimed. This was one
implementation-proof pass over seven examples, not a trained release. Do not
promote it. Paired native history collection 7452946–58 completes all 13 actual
counterfactual TRAIN worlds. The next bounded behavior pilot will use those
additional histories only where native codec/gold and causal assembly checks
pass, before comparing a finite number of training lengths. Actual ordinary
answers and speech must also be rechecked before promotion.


**Plain words — same-GPU backward cause isolated:** 7453014 uses nid010035,
GH200 120GB, Torch 2.9.0a0+145a3a7bda.nv25.10, CUDA13 and cuDNN91400. The
same zero adapters/input on that GPU give 44 invalid-gradient tensors through
aten::_scaled_dot_product_cudnn_attention_backward; efficient attention and
math give zero. CPU profiling confirms actual selected operators. Clearing
probe gradients and training with cuDNN SDPA disabled reproduces all eight
losses and checkpoint SHA exactly from 7452937. Training now defaults to the
verified efficient path. Serving still uses its separate native custom backend.

Paired assembly 7453119 passes 15 causal native examples from 21 collected
TRAIN histories; six without exact native gold transcripts remain excluded.
A bounded pilot permits only 64/128 updates after the passing implementation
proof, rechecking every original source approval and exact frozen base hash.
Both arms retain zero/frozen/branch/gradient/reload gates and cannot qualify
quality from loss. Their native actual-worker replays and ordinary answers
must decide whether this recipe improves the task. No corpus-scale run is
justified yet; no failed-gradient checkpoint is allowed.


**Plain words — both longer semantic pilots fail actual tasks:** native
7453230/38/39/40 (64) and 7453242/43/44/45 (128) invent room temperatures rather
than returning the room marker. One 64-step client-event run says "The lightning
is purple diamond", which is a literal value echo with the wrong referent and
is not a correct result. Ordinary 7453241/46 also change science answers. No
pilot qualifies; do not increase epochs or corpus scale on this recipe.

Next diagnosis replays a saved approved TRAIN prefix with its supplied future
S-S event and no supplied future word/acoustic code. It compares actual native
first-layer input embeddings with the exact training tensor, then lets native
words/speech codes run freely. Compare base, 64 and 128 on two original TRAIN
records and a paired world. This separates mismatched inputs/deployment from
on-policy trajectory drift; it cannot qualify native turn choice or live tasks.
A literal text diagnostic separately attributes private words to their FIRST
sampled subword, catches guesses that finish after result delivery, and never
promotes matching text to correct claims, actual speech or full-task success.
Two timing/echo regressions pass. The 200ms environment field is also explicitly
scoped to the older Moshi interleaver; Lychee word-offset quality is unqualified.
