# Native Lychee continuation


**15 October, result-integration diagnosis:** static reference controls
7450548/50 correctly say both private values. Actual late receipts 7450551/53
omit them, and the disclosed application response-start controls 7450827/28
also omit them. This rules out a missing response-start event as a sufficient
fix. New diagnostics audit the exact fresh system prefix at the actual vLLM
add-request boundary and compare immediate delivery with delivery when the
assistant is listening and two already-arrived 400ms user chunks are quiet.
The latter is explicitly application scheduling; all words/acoustics remain
sampled. Keep backend-ready, context-delivery and response clocks separate.
Neither variation enters training. Warm-input benchmark 7451709 is submitted.

Per-utterance playout replay is now verified: fixed 240ms buffering produces
zero within-utterance gaps for trace 7443647, preserves every played PCM sample,
and cancels 26,800 queued samples at actual native interruption times. This is
component replay only. ASR audit 7450830 gives full-transcript WER 0.2763 for
both offline and repaired playout; intentionally cancelled tails and spoken
number formatting need separate analysis. This does not pass the 0.08 bar.

15 October 2026. No full-task or microphone model qualifies.

The released custom-vLLM backend now works on the isolated CUDA13 build. Run
7441742 accounts for all 982 source tensors (488 model, 487 encoder, six adapter,
one explicitly unused merge token embedding). All 329 packed parameter/shard
groups load; side-head weight copying is not used. It generates four responses
and 1304 speech-token events from 70 seconds of paced input in 70.523 seconds.
Compute p95 is 297ms per 400ms chunk. Initial backlog is 1.280s; this is token
processing, with no concurrent PCM or microphone claim.

The existing 200ms native encoding check is alignment of each first subword to
its measured word onset. It is not a response-start latency target. Keep response
start, input backlog, PCM availability and within-utterance playback gaps separate.

Offline vocoder 7443260 decodes the generated speech in 2.450 seconds using the
accepted synthetic voice. This is quality-only placement. Streaming replay
7443262/64 failed on an incorrect adapter name; corrected immutable-source jobs
7443646 (hop25) and 7443647 (hop8) use the actual published StreamingDecoder,
lookahead-three overlap and cache rules. They must count every unique codec token
and preserve actual recorded arrivals. This remains a component replay, not
simultaneous speech-model/vocoder execution. The new tests use decoder batch one;
the application defaults are decoder batch ten and vocoder hop25. Raw default
StreamingDecoder's constructor uses batch25. Do not conflate those settings.

Actual-worker diagnostic 7443841 (yellow circle) and 7443842 (purple diamond)
use the same approved private validation question. Work is scheduled after the
arrived first user utterance, and the backend uses causal measured words rather
than ASR. Pending state and then the actual returned record enter only through
the native nonspoken system prefix. Every existing text/acoustic/control token,
relative speech mapping, arrived audio feature/tail, next input clock and RNG
must survive the prefix rebuild. Prefix KV caching is disabled because its keys
do not include audio. All output words and acoustic tokens remain sampled.
These tests cannot establish native foreground delegation choice. No validation
audio, transcript, world or generated answer enters training.

If context preservation fails, fix the exact interface before interpreting its
speech. If it passes, independently transcribe the generated audio and compare
spoken values with actual receipts in both worlds, including premature guesses.
If it actually uses both returned values correctly, use this text-context path
for the native routing/data stage. Do not assume ARC alignment or a pretrained
tool policy. Native foreground choice, clarification and cancellation still need
TRAIN-only training and fresh full-task evaluation.

For PCM, first verify the published streaming vocoder at recorded token arrivals,
then connect concurrent processes on two allocated GPUs (Exclusive_Process).
Keep the verified LM/CPU overlay separate from the accepted HF/vocoder overlay.
Warm up before real user input, close/reset the warm session and keep prefix
caching disabled. Preserve all actual user samples and measure both LM backlog
and audio availability. No offline vocoder or concatenated quality WAV counts as
an end-to-end real-time pass.

PersonaPlex remains a voice/speed reference. Its small adaptation runs, causal
state feedback and paired-grounding recipes fail actual private-result use.
Its native wrapper now preserves RNG across adapter construction; legacy seeded
replays consumed extra CUDA random draws and are not matched raw-model controls.
Sampling-preserved 7441744–53 still fails task speech. Do not scale those recipes
merely because frozen hashes, reload, teacher loss or TRAIN reference margins pass.

All new long jobs should run from a clean immutable git worktree using
DL_TASK_SOURCE and DL_TASK_COMMIT. Submit from the main repo to retain the usual
Slurm log directory. Code deploys through git; artifacts/logs may be rsynced.
