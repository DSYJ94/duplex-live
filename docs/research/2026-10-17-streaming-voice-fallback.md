# 17 October: disclosed streaming-voice fallback

Plain words: a strong text controller speaking through Kyutai TTS is a plausible
separate fallback. It would move answer selection and delegation out of the native
7B voice model. It does not qualify that model or establish equivalent duplex
behavior. This is source research only: no model download, installation, GPU job,
credential inspection or fallback implementation was performed.

## Decision for Isambard

Plain words: do not spend the immediate Isambard window deploying stock Unmute.
Its publisher explicitly requires x86_64 and excludes aarch64, so GH200's ARM host
is outside its supported setup. The published minimum is one CUDA GPU with 16 GB;
the default small language model is Gemma 3 1B, which is not evidence of strong
reasoning. Tool calling is not built in. The README proposes hiding it behind an
LLM-server wrapper. These are upstream statements, not verified Isambard results.
[Unmute README](https://github.com/kyutai-labs/unmute/blob/main/README.md).

Plain words: the concrete next action, after the native comparison, is a bounded
compatibility check on one already allocated GH200: record Python/PyTorch/CUDA
versions and whether the exact DSM checkpoints and a published voice embedding
are already cached. If available, test the upstream PyTorch TTS path with one
actual successful backend receipt, then one interrupted utterance. Missing assets
or incompatible imports are an explicit blocker; this note authorizes no new
resource or download. Do not start a Rust/Unmute ARM port as the emergency path.
This recommendation follows the supported-platform gap above and the direct
PyTorch example below; it is an engineering inference, not a measured baseline.

## Concrete runtime seams and pins

Plain words: use the publisher's actual APIs, with their dependency pins. The DSM
streaming TTS example specifies Python >=3.12 and `moshi==0.2.11`. It loads
`CheckpointInfo.from_hf_repo`, then `TTSModel.from_checkpoint_info`; obtains
`get_voice_path` and `make_condition_attributes`; and uses `script_to_entries`,
`TTSGen.append_entry`, `process`, `process_last` and an `on_frame` callback decoding
with `mimi.decode`. `TTSGen` is example-local code, not a promised stable library
class. Its stdin loop waits for lines, so piping token deltas without line breaks
does not demonstrate immediate incremental input. A controller adapter would
need to feed entries explicitly and manage per-response state.
[Streaming TTS source](https://github.com/kyutai-labs/delayed-streams-modeling/blob/main/scripts/tts_pytorch_streaming.py).

Plain words: the corresponding PyTorch STT example also pins `moshi==0.2.11`.
It creates `CheckpointInfo`, `get_mimi`, `get_moshi(dtype=torch.bfloat16)` and
`LMGen`; steps encoded chunks under `mimi.streaming(1)` and
`lm_gen.streaming(1)`; and supports VAD through `step_with_extra_heads`. Its
`--vad` default selects `kyutai/stt-1b-en_fr-candle`. Do not copy its timestamp
clamping into this repo's independently validated recognition contract.
[STT source](https://github.com/kyutai-labs/delayed-streams-modeling/blob/main/scripts/stt_from_file_pytorch.py).

Plain words: Unmute's Dockerless speech scripts pin Rust `moshi-server@0.6.4`;
the TTS script retrieves embedded Python dependency files from Moshi commit
`9837ca328d58deef5d7a4fe95a0fb49c902ec0ae`. The backend requires Python
`>=3.12,<3.13`. These pins do not prove ARM compatibility. The inspected main-branch
source URLs and HF model identifiers are mutable: an actual trial must save
source revisions, checkpoint/voice hashes and installed versions. No immutable
Unmute/DSM repository head or HF revision was established in this research.
[TTS startup](https://github.com/kyutai-labs/unmute/blob/main/dockerless/start_tts.sh),
[STT startup](https://github.com/kyutai-labs/unmute/blob/main/dockerless/start_stt.sh),
[Python requirements](https://github.com/kyutai-labs/unmute/blob/main/pyproject.toml).

Plain words: the concrete Unmute interfaces are browser WebSocket `/v1/realtime`
and speech-service WebSockets `/api/asr-streaming` and `/api/tts_streaming`.
The latter uses MessagePack `Text`/`Eos` input and `Ready`/`Audio`/timestamped `Text`
output, with `format=PcmMessagePack`. Browser audio uses base64 Opus, 24 kHz mono;
internal PCM frames are 1,920 samples (80 ms). Its LLM adapter sends streaming
Chat Completions to `KYUTAI_LLM_URL + /v1`, selects `KYUTAI_LLM_MODEL`, and
re-chunks output into whole words. An existing Responses-based worker therefore
needs an explicit adapter; changing the URL alone is insufficient.
[Protocol](https://github.com/kyutai-labs/unmute/blob/main/docs/browser_backend_communication.md),
[Constants](https://github.com/kyutai-labs/unmute/blob/main/unmute/kyutai_constants.py),
[TTS client](https://github.com/kyutai-labs/unmute/blob/main/unmute/tts/text_to_speech.py),
[LLM adapter](https://github.com/kyutai-labs/unmute/blob/main/unmute/llm/llm_utils.py).

## Voice and latency limits

Plain words: the accepted Context Spanning Mimi reference is not automatically a
Kyutai TTS voice. The official `tts-1.6b-en_fr` card says the model actually has
1.8B parameters and conditions on precomputed cross-attention embeddings. It
explicitly restricts cloning to those embeddings. Shared use of Mimi does not
make reference audio and speaker embeddings interchangeable. First test a
published embedding and disclose the different voice; preserving the accepted
identity remains unproved. The card's 16-frame/1.28-second text/audio shift is
an architectural delay, not a measured end-to-end response time.
[Model card](https://huggingface.co/kyutai/tts-1.6b-en_fr).

Plain words: Unmute has custom-voice plumbing, but it calls a separate
`VOICE_CLONING_SERVER /api/voice` endpoint and caches the returned embedding.
That client code does not supply an independently runnable encoder or prove that
our accepted reference transfers.
[Voice client](https://github.com/kyutai-labs/unmute/blob/main/unmute/tts/voice_cloning.py).

Plain words: upstream reports TTS latency around 750 ms with all services sharing
one L40S and around 450 ms on its separated production GPUs. Its Dockerless
estimates are STT 2.5 GB, TTS 5.3 GB and LLM 6.1 GB. These are publisher figures,
not GH200 measurements or complete conversational latency. DSM describes the
1B STT's 0.5-second delay and semantic VAD; the English 2.6B alternative has a
2.5-second delay. Prefer the 1B candidate for this latency proof. A first
single-session GH200 test should avoid competing native jobs and use batch one;
no minimum GH200 allocation or achieved latency has been verified here.
[Unmute README](https://github.com/kyutai-labs/unmute/blob/main/README.md),
[DSM overview](https://github.com/kyutai-labs/delayed-streams-modeling/blob/main/README.md).

## Cancellation and what the listener heard

Plain words: Unmute provides useful cancellation machinery, not a completed
delegation contract. `interrupt_bot` clears the FastRTC queue, swaps its own queue,
emits an interruption event, and removes TTS/LLM quests. Each TTS worker retains
its old queue so it cannot contaminate the new response. Assistant history is
updated from TTS-timed text, rather than every generated LLM token. Speech can
interrupt through STT text or semantic VAD, with a special initial VAD guard.
These are inspected source behaviors; no live microphone acceptance was tested.
[Handler](https://github.com/kyutai-labs/unmute/blob/main/unmute/unmute_handler.py).

Plain words: closing TTS does not cancel a backend lookup or retract audio already
played. The TTS client paces queued audio and explicitly warns of transport
buffering/desynchronization; server-yielded samples are not a browser playback
receipt. A fallback needs response IDs and revisions across controller, tool,
TTS and playback, late-result rejection, queue flushing, and a client playback
watermark. Keep generated, emitted and actually played words separate. This is a
required integration design, not a capability verified in stock Unmute.
[TTS client](https://github.com/kyutai-labs/unmute/blob/main/unmute/tts/text_to_speech.py).

Plain words: this repo already owns the task side in `RecordWorld`: revision
reservation, actual cancel receipts and `deliverable` reject superseded results.
`run_worker` explicitly marks `speech_model_routing=False`. The concurrent Lychee
vocoder tracks event IDs, independently received stops and late PCM, but its
output is causal playout simulation, with no tested device or microphone. Reuse
these ownership ideas; do not label their simulation as heard-audio evidence or
call externally controlled TTS native voice-model delegation.
[Worker](../../evals/recovery_worker.py),
[Concurrent vocoder](../../evals/lychee_concurrent_vocoder.py).

## Read-only Isambard inventory

Plain words:the17 October inventory identifies an existing
`models--kyutai--stt-1b-en_fr-candle` cache in the checked HF directory, but no
TTS/Unmute asset in the checked model/vendor names. The old Moshi environment
has no matching distribution-metadata directory; that does not imply its
PYTHONPATH-provided Moshi source is absent. No model was loaded, downloaded or
installed. This narrows the next compatibility check; it is not a ready fallback.
Evidence: `runs/recovery-continuation-20261016/duplex-streaming-assets-inventory-full.txt`.
