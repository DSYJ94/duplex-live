# Recovery base selection — 2026-10-13

Plain words: the next useful step is to run an already speaking duplex model on our tasks, before spending the remaining allocation adapting another model into that capability. **MoshiRAG is the first delegation baseline; MiniCPM-o 4.5 is the strongest alternative to test for general conversation.** Plain Moshi has the clearest official fine-tuning path. PersonaPlex is a useful speech-quality comparator. Qwen remains a diagnostic and possible recovery route, but connecting its speech decoder cannot establish that our custom stream produces sensible answers.

This is a source audit and proposed decision, not a benchmark result. No GPU jobs were launched for this note. Sources were checked on 13 October 2026; upstream `main` links can change, so pin the actual source and model revisions before execution.

## What the target requires

Plain words: success means answering with meaningful new speech, hearing a correction during that speech, and completing background work without abandoning the conversation.

Thinking Machines describes continuous 200 ms input/output micro-turns and an asynchronous background model with shared conversation context. It explicitly includes integrating results while interaction continues. Its design is useful guidance; the blog is not a released training recipe we can reproduce in eight days. Measure meaning, timing and completed work together. [Thinking Machines system description](https://thinkingmachines.ai/blog/interaction-models/)

The local handoff reports successful turn timing but failed free-run word content and delegation mixing. `layera/train_p1.py:237` freezes the model and permits gradients only through new embedding/head rows; old vocabulary rows are masked. This is not a general fine-tune of the language core. The note in `HANDOFF.md` suggesting Talker or more scale as the next move is a hypothesis, not a demonstrated remedy. Preserve that checkpoint as a timing baseline.

## What is actually released

| Candidate | Released capability established by primary sources | Practical decision |
|---|---|---|
| MoshiRAG | Weights, live and offline inference; a retrieval trigger and a streaming reference-conditioning path | First baseline for speech plus asynchronous delegation. Training integration still needs work. |
| MiniCPM-o 4.5 | Weights and native duplex audio/video/text-to-speech inference | First alternative for conversational content. Verify timed audio fine-tuning before allocating a training run. |
| Moshi | Native two-stream speech and official LoRA/full fine-tuning | Clearest trainable fallback; adding result conditioning is additional work. |
| PersonaPlex | Moshi-derived full-duplex weights with voice and role prompts | Speech-quality comparator. Official inference release does not establish turnkey fine-tuning or live result injection. |
| Qwen3-Omni | Thinker/Talker weights, speech generation and audio function-call example | Run native baseline to isolate damage from custom streaming. Native generation alone does not demonstrate continual input while output runs. |

The table's engineering priorities are recommendations, supported by the release details below.

### MoshiRAG: the closest released starting point

Plain words: this model already contains the difficult bridge between an ongoing voice conversation and arriving background text.

The official repository provides a duplex front end, an asynchronous text backend, and a reference encoder. Its trigger sends conversation text to the backend; returned references condition later speech. It cautions that retrieval delays above about three seconds hurt quality. This is evidence for a useful starting point, not proof of long-running task management. [Official repository](https://github.com/kyutai-labs/moshi-rag)

The checkpoint is English and has a single voice. Its card names frozen ARC-Encoder components and a separate streaming ASR dependency. The card lists CC-BY-4.0 weights. [Model card](https://huggingface.co/kyutai/moshika-rag-pytorch-bf16)

The model repository totals approximately **15.8 GB**: `model.safetensors` is 15.4 GB and the Mimi checkpoint approximately 385 MB, plus configuration/tokenizer files. This excludes additional models fetched by dependencies. [Actual file list](https://huggingface.co/kyutai/moshika-rag-pytorch-bf16/tree/main)

Training in the paper updates the speech model while freezing the reference encoder; it uses timed reference arrival and reference dropout, with about 47,770 hours of synthetic speech. That scale is a reason to reuse the released weights. It does not imply our 515 lessons can recreate them. [Paper, training and data tables](https://arxiv.org/html/2604.12928v1)

**Executable seams and caveats:**

- `inference_utils/utils.py` loads the model with `force_streaming_sum=True`. Setting `REFERENCE_ENCODER_URL` omits the reference conditioner from the front-end model and calls the separate service's `/embed` endpoint. [Loader and encoder client](https://raw.githubusercontent.com/kyutai-labs/moshi-rag/main/moshi/moshi/inference_utils/utils.py)
- `InferenceJob._async_update_reference` installs encoded text through `lm_gen.update_streaming_sum_tensors(per_slot)`. Its trace records trigger, retrieval and conditioning steps and saves generated WAV. However, the offline output loop explicitly waits while retrieval is pending, then represents elapsed retrieval time in frames. **Offline output is not evidence that the live system remains responsive during retrieval.** [Offline session implementation](https://raw.githubusercontent.com/kyutai-labs/moshi-rag/main/moshi/moshi/inference_utils/inference_job.py)
- `RAGManager` creates a background task, tracks reference history, and cancels an existing request on a new trigger. Replace or wrap the text backend at this boundary for a bounded task executor; retain explicit task identity and stale-result rejection. Multi-task orchestration is our extension. [RAG manager](https://raw.githubusercontent.com/kyutai-labs/moshi-rag/main/moshi/moshi/inference_utils/rag_manager.py)
- The live `Channel` owns independent audio queues and retrieval state. Test this path with deliberately slow results, new user instructions, cancellation and errors. [Live channel](https://raw.githubusercontent.com/kyutai-labs/moshi-rag/main/moshi/moshi/inference_utils/channel.py)
- `LMModel.forward` has a training entry point, but `forward_text` currently asserts a one-step sequence when streaming conditioning is supplied. **The stock whole-sequence Moshi trainer is not established as compatible with timed MoshiRAG conditioning.** Before adapting, demonstrate a differentiable batch with correctly timed references, finite gradients and checkpoint reload. [Model implementation](https://raw.githubusercontent.com/kyutai-labs/moshi-rag/main/moshi/moshi/models/lm.py)

### MiniCPM-o 4.5: real duplex release, incomplete training evidence

Plain words: this is a credible ready-made conversation model; the existence of general fine-tuning support does not prove the specific duplex training loop we need.

OpenBMB provides the 9B model, native `as_duplex()` examples, and SDPA as an attention option. It acknowledges speech mispronunciation, mixed languages, and limits in duplex capability. [Official release and examples](https://github.com/OpenBMB/MiniCPM-V)

The published duplex implementation accepts `text_list` in `streaming_prefill`, providing a candidate context-injection point. Its documentation describes one-second prefill calls. In the inspected source, mixed audio/text processing saves `pending_logits` from audio, then feeds text without refreshing those logits; text-only mode does refresh them. Static inference: the first generated token may use pre-injection logits. Verify before depending on this path, and do not advertise 80 ms decisions merely because the enclosing project uses that tick. [Published implementation, duplex prefill/generate](https://huggingface.co/openbmb/MiniCPM-o-4_5/raw/main/modeling_minicpmo.py)

The repository's fine-tuning README describes image/conversation examples and older model versions. Its training script explicitly sets `init_audio=False` and `init_tts=False`. These particular scripts are not evidence for end-to-end duplex speech fine-tuning. Other supported frameworks may provide relevant paths, but no verified timed-duplex recipe was established in this audit. [Fine-tuning guide](https://raw.githubusercontent.com/OpenBMB/MiniCPM-V/main/finetune/readme.md), [Training source](https://raw.githubusercontent.com/OpenBMB/MiniCPM-V/main/finetune/finetune.py)

### Moshi and PersonaPlex

Plain words: plain Moshi has a concrete official adaptation recipe; PersonaPlex improves the baseline choice but adds compatibility questions.

Kyutai's trainer supports LoRA and full fine-tuning, with stereo conversations (assistant left, user right) and timestamped transcripts. The published example reports 39.6 GB peak allocation on one H100. This supports a bounded GH200 smoke-test proposal, not a GH200 speed prediction. It also warns that shortening training segments can hurt sustained conversation. [Official Moshi fine-tuning repository](https://github.com/kyutai-labs/moshi-finetune)

NVIDIA releases PersonaPlex inference, offline WAV evaluation, voice prompts and role prompts. It is based on Moshi, with MIT code and NVIDIA Open Model License weights. The inspected release does not supply an official matching training recipe or documented mid-session background-result interface. Do not treat third-party PersonaPlex trainer forks as proof that Kyutai's unmodified trainer loads it correctly. [Official PersonaPlex release](https://github.com/NVIDIA/personaplex)

### Qwen: speech output and content recovery are different experiments

Plain words: first find out whether the original model answers correctly; then establish what our stream format changed.

Qwen's released Instruct model contains Thinker and Talker; the Thinking variant contains Thinker only. The official examples cover generated speech and audio function calls and warn that Transformers MoE inference can be slow. They do not establish our custom continuous listening/speaking loop. [Official Qwen3-Omni repository](https://github.com/QwenLM/Qwen3-Omni)

Run the same fresh spoken questions through: (1) unchanged native Thinker with its standard template; (2) native Thinker plus Talker; (3) the custom streamed checkpoint, comparing generated text before speech rendering. If (1) is coherent and (3) is not, examine formatting, audio representation, history and trainable capacity. A Talker connection can repair an absent speech path; it cannot be assumed to repair incorrect linguistic content. Train a small backbone adaptation only after a small-set content experiment proves it can learn and retain answers in free running use.

## ARM/GH200 execution constraints

Plain words: install each candidate separately and first prove it imports, loads and speaks in the working container.

The local handoff identifies `pytorch-new.sif` / NGC 25.10 with container Torch 2.9 as the working base; the old container fails after the driver update. Existing MiniCPM and PersonaPlex environments must be inspected rather than assumed reusable. Preserve the working Qwen overlay.

MoshiRAG declares Torch `>=2.2,<2.10`, Transformers **4.55.3**, Hugging Face Hub `<1`, xformers `>=0.0.33.post2`, and Linux bitsandbytes. Torch 2.9 is nominally in range, but the current Qwen Transformers 5.17 overlay conflicts. These dependencies do not establish ARM wheel availability or binary compatibility. Use an isolated overlay, keep container Torch, and verify imports plus one CUDA operation before downloading every model. [MoshiRAG dependency manifest](https://raw.githubusercontent.com/kyutai-labs/moshi-rag/main/moshi/pyproject.toml)

For the bounded first run, use Python 3.11+ because the inference code calls `asyncio.TaskGroup`. Once the package is installed, the module is `python -m moshi.run_inference`; the README's nested `moshi.moshi` invocation depends on checkout layout. `--use-gt-reference` reads `gt_reference_text` from a WAV's adjacent JSON, isolating result use from backend answer quality. The code still constructs local ASR. Require a timeout and generated output. [Offline entry point](https://raw.githubusercontent.com/kyutai-labs/moshi-rag/main/moshi/moshi/run_inference.py)

Local ASR defaults to `kyutai/stt-1b-en_fr-candle` and loads through the same Moshi stack. Its own model and codec requirements add to the download/memory footprint. [Local ASR implementation](https://raw.githubusercontent.com/kyutai-labs/moshi-rag/main/moshi/moshi/stt/local_stt.py)

## Falsifiable gates before substantial training

Plain words: spend compute on candidates that already demonstrate the required behavior, then improve their measured weaknesses.

These are proposed acceptance gates, not achieved scores or literature claims:

1. **Base speech:** 20 fresh questions, at least 16 substantively correct audible answers, no answer bank; archive input/output PCM and exact generated text. Human listening is required alongside ASR.
2. **Actual duplex:** 20 correction/interrupt scenarios, at least 16 correct uses of new information; report stop latency and answer correctness separately. In ten-minute sessions, capture input while speaking and show no growing processing backlog.
3. **Background result use:** 20 controlled tasks with previously unavailable facts and 0.5, 2, 5 and 15 second return delays. Require at least 18 correct incorporations, no claims of a result before it arrives, and zero stale/cancelled-result application. Include speech during pending work. Report each delay bucket; a pass only at 0.5 seconds is insufficient.
4. **Trainability:** overfit 16 approved conversations with timed references; reload the saved model and run it without teacher forcing. Then evaluate fresh content and interruption cases. No broad run until content improves without loss of baseline turn-taking.
5. **Selection:** rank the measured candidates by joint speech meaning, interactivity and task success; use latency and training effort to break ties. Do not choose on timing F1 alone.

Treat the claimed 16,000 available GPU-hours and 21 October deadline as inputs requiring live accounting verification. Even eight complete days would require about 84 continuously occupied GPUs to consume 16,000 hours; a four-GPU run for eight days consumes only 768. Allocation, queue capacity and useful parallel experiments are separate constraints. Start with capped single-node baselines, then assign larger runs from measured throughput and passing gates. Spend no budget recreating released pretraining merely to use the allocation.

The recommended first decision is therefore **MoshiRAG baseline plus MiniCPM comparison, with native Qwen diagnosis retained**. The winning adaptation target remains conditional on actual audible results and a verified training step. The principal unknown is whether long-delay background work can be taught without damaging continuous conversation; neither retrieval demos nor our old delegation-token emission experiment settles it.

## Minimal MoshiRAG execution plan — source checked, not yet run

Plain words: a completely local test is possible, but `--use-gt-reference` alone does not disable every backend call. Prove the reference encoder works first.

**Import audit:** bitsandbytes is imported inside quantized `QLinear` operations, so the BF16 path can omit it. Gradio is imported only for the tunnel option in the server. In contrast, ARC imports xformers at module load and directly calls its attention kernel; it has no working SDPA fallback in the inspected implementation. Gradium is eagerly imported even when local ASR is selected, so its package is still needed unless that import is made lazy. [Quantization](https://raw.githubusercontent.com/kyutai-labs/moshi-rag/main/moshi/moshi/utils/quantize.py), [ARC encoder](https://raw.githubusercontent.com/kyutai-labs/moshi-rag/main/moshi/moshi/conditioners/arc_encoder.py), [Gradium module](https://raw.githubusercontent.com/kyutai-labs/moshi-rag/main/moshi/moshi/stt/gradium_stt.py)

The reference-generator constructor warms up its configured LLM, and `ServerState.warmup()` does so again, including in offline runs. For a controlled-reference test, explicitly point `LLM_BASE_URL` at a local test server returning a valid chat-completion object, or modify the adapter to skip constructing/warming the unused backend. A local stub establishes conditioning and concurrency, not actual task execution. [Reference generator](https://raw.githubusercontent.com/kyutai-labs/moshi-rag/main/moshi/moshi/reference/llm_reference_generator.py), [Server initialization](https://raw.githubusercontent.com/kyutai-labs/moshi-rag/main/moshi/moshi/server.py)

The encoder service builds only conditioners from the speech checkpoint, then loads their external encoder weights. It can run on the same GPU if memory permits. A separate GPU isolates interference; CPU is an accepted argument but the xformers kernel path must be proved before claiming CPU support. Bind services to localhost. [Encoder service](https://raw.githubusercontent.com/kyutai-labs/moshi-rag/main/moshi/moshi/server_conditioner.py)

Proposed order inside the working ARM container:

1. Clone/pin upstream outside the project package path. Create a separate `venv --system-site-packages`, with the Qwen `PYTHONPATH` unset. Retain NGC Torch. Inspect a pip dry-run report for every dependency install; reject proposed Torch/CUDA replacement. Install the checked-out Moshi package with `--no-deps` only after satisfying its actual imports. Use Python 3.11 or newer.
2. Verify `torch.__file__`, version, CUDA operation, `transformers==4.55.3`, `moshi.__file__`, `sphn`, `gradium`, `fastapi`, `uvicorn`, and `moshi.conditioners.arc_encoder`. Execute a small BF16 xformers `memory_efficient_attention` call with `BlockDiagonalMask`, rather than trusting an import alone. Do not assume an ARM wheel exists or build it in the live Qwen environment.
3. Download/pin `kyutai/moshika-rag-pytorch-bf16`, its configured ARC encoder, and local ASR dependencies. The default ARC dependency is `kyutai/ARC4_Encoder_Llama`. [Checkpoint loader](https://raw.githubusercontent.com/kyutai-labs/moshi-rag/main/moshi/moshi/models/loaders.py)
4. Start the conditioner, using a verified local model directory in `$MOSHI_MODEL_DIR`:

```sh
python -m moshi.server_conditioner \
  --config "$MOSHI_MODEL_DIR/config.json" \
  --moshi-weight "$MOSHI_MODEL_DIR/model.safetensors" \
  --conditioner reference_with_time --cuda-device 0 \
  --host 127.0.0.1 --port 8001
```

5. Point the backend to a local test server or local reasoning model; do not set a public endpoint for this smoke test. Set `REFERENCE_ENCODER_URL=http://127.0.0.1:8001`, `LLM_BASE_URL=http://127.0.0.1:8002/v1`, `LLM_API_KEY=local-test`, and an appropriate `LLM_MODEL_NAME`. First call `/embed` and validate the returned safetensors.
6. For content only, run `python -m moshi.run_inference --input-dir "$PROBE_INPUT" --output-dir "$PROBE_OUTPUT" --use-gt-reference --batch-size 1 --max-consecutive-silence-frames 40`, under an external time limit. Use a verified mono user WAV plus adjacent JSON reference. Cache downloads first; use exact model paths/revisions rather than silently changing weights.
7. For concurrency, run `python -m moshi.server --host 127.0.0.1 --port 8998 --batch-size 1 --init-active-speaker user --rag-timeout 20`. Omit `--gradium-stt` and `--gradio-tunnel`; use local ASR and an SSH-forwarded client. Send paced audio through `/api/chat`, record receive and playback times, and delay local backend replies by 0.5/2/5/15 seconds. Verify ongoing input consumption, output delivery, and updated-answer content. Avoid using the offline loop as this measurement.

No package lock, ARM binary compatibility, GPU memory envelope, live run or benchmark result has been established by these commands. Resolving those is the bounded baseline experiment; training starts only after it succeeds.

## Training data: what can actually be acquired and trained

Plain words: DuplexChat's advertised 282,000 English hours are pointers to podcast audio, not a ready-to-download speech corpus. Start with a tiny existing Moshi-format dataset to verify training, then reconstruct a measured, approved subset with independent evaluation speakers and conversations.

### DuplexChat availability and conversion

The Hugging Face release contains **metadata only**: the English compressed manifest is approximately 508 MB and indexes 15,304,412 dialogues / 282,634 hours. Each row identifies a source episode URL and dialogue interval. Its MIT license covers released metadata/code; it does not grant rights to the source podcasts. Record the source publisher's applicable permission/license in the acquisition manifest rather than treating the MIT label as an audio license. [Released files](https://huggingface.co/datasets/sarulab-speech/DuplexChat/tree/main), [Dataset card](https://huggingface.co/datasets/sarulab-speech/DuplexChat/blame/main/README.md), [Official reconstruction guide and licensing scope](https://github.com/sarulab-speech/DuplexChat)

Reconstruction downloads each episode, converts to 16 kHz mono, slices the intervals and separates speakers using gated DialogueSidon weights. It emits stereo MP3 plus `meta.json` in WebDataset archives, **without word transcripts or `diarization.json`**. No new diarization pass is necessary. Left/right represent two separated speakers, not predetermined assistant/user roles. The official environment requires Python 3.12 or newer, CUDA and ffmpeg; keep it separate from inference. [Reconstruction source](https://raw.githubusercontent.com/sarulab-speech/DuplexChat/master/scripts/reconstruct_dataset.py), [Dependency manifest](https://raw.githubusercontent.com/sarulab-speech/DuplexChat/master/pyproject.toml)

DialogueSidon loads exported Torch programs, so their compatibility with NGC Torch 2.9 on ARM must be demonstrated. Separation uses random noise, splits long clips into overlapping windows and attempts to maintain channel identity. Pin revisions and seeds, preserve source/output hashes, and inspect leakage and speaker swaps. Do not promise byte-identical reconstruction or assume the output sample rate equals the 16 kHz intermediate; read the returned rate. [Separation implementation](https://raw.githubusercontent.com/sarulab-speech/DuplexChat/master/src/duplexchat_pipe/separate.py)

Proposed bounded acquisition:

1. Download the English metadata only. Stream-filter an allowlist of usable sources into a **small `.jsonl.gz`** with a deterministic seed, provenance fields and a hard duration budget. Assign split groups before selecting clips. Aim initially for 10 usable hours across multiple shows, with approximately 8/1/1 train/validation/test hours; expand to 50 only after measured reconstruction quality and throughput.
2. Verify DialogueSidon access and one export load/kernel in the isolated environment. Run two selected episodes first, then process the already bounded manifest:

```sh
python scripts/reconstruct_dataset.py \
  --manifest bounded_en_10h.jsonl.gz --output reconstructed_10h \
  --num-shards 1 --shard-index 0 --limit-episodes 2
```

3. Remove `--limit-episodes` after the smoke passes. That option is applied **after** grouping the manifest in memory, so it does not safely bound a full-manifest load. Track failed sources and successfully accepted duration: partial failures can still produce exit code zero. Keep shard count fixed when resuming. [Loader, limit and failure behavior](https://raw.githubusercontent.com/sarulab-speech/DuplexChat/master/scripts/reconstruct_dataset.py)
4. Decode accepted clips to aligned stereo WAV, map the selected assistant to the left channel, resample both channels together to the target codec's rate (24 kHz for standard Mimi), and annotate assistant words. Preserve silence, overlap and shared time origin. Role-swapped copies must stay in the original split, with transcripts regenerated for the new assistant channel.

### Exact Moshi input and checkpoint constraints

Plain words: each training example is a stereo conversation with an adjacent word-timing file; arbitrary dialogue JSON is insufficient.

Use an absolute-path JSONL manifest such as:

```json
{"path":"/data/approved/conversation_001.wav","duration":24.52}
```

The adjacent `/data/approved/conversation_001.json` contains assistant words with seconds measured from the start of that WAV:

```json
{"alignments":[["Hello",[0.5,0.9],"SPEAKER_MAIN"],["there.",[0.95,1.3],"SPEAKER_MAIN"]]}
```

The official annotator reads channel zero and produces this sidecar. Within an allocated worker, `python annotate.py train.jsonl --local --lang en --whisper_model medium` runs locally; omitting `--local` invokes its Slurm submission logic. Do not run this proposed GPU command on a login node. [Annotator](https://raw.githubusercontent.com/kyutai-labs/moshi-finetune/main/annotate.py)

Require chronological alignments and timestamps within the clip. The interleaver performs a binary search before sorting, so unsorted input is unsafe. Training uses assistant-left/user-right audio and `keep_main_only=True` text supervision; user speech remains an audio input. Sidecar `text_conditions` support does not establish a valid timed MoshiRAG training format. [Interleaver](https://raw.githubusercontent.com/kyutai-labs/moshi-finetune/main/finetune/data/interleaver.py), [Dataset loader](https://raw.githubusercontent.com/kyutai-labs/moshi-finetune/main/finetune/data/dataset.py), [Training entry point](https://raw.githubusercontent.com/kyutai-labs/moshi-finetune/main/train.py)

The official example targets **`kyutai/moshiko-pytorch-bf16`** and supports LoRA or full adaptation of the language/audio model while Mimi stays frozen. Its sample config disables evaluation; explicitly enable it and supply `eval_data`. Start with 16 approved examples, a small batch, checkpoint reload and free-running speech evaluation. The package pins **Torch 2.6 and sphn 0.1.12**, conflicting with the current Torch 2.9/MoshiRAG stack. Pin trainer and Moshi revisions and validate an isolated compatibility patch rather than installing its dependencies over the working container. [Example config](https://raw.githubusercontent.com/kyutai-labs/moshi-finetune/main/example/moshi_7B.yaml), [Trainer dependencies](https://raw.githubusercontent.com/kyutai-labs/moshi-finetune/main/pyproject.toml)

Neither PersonaPlex nor MoshiRAG is verified as a drop-in checkpoint for this trainer. The loader uses `strict=False`; successful loading alone does not establish compatibility. Review missing/unexpected keys, tensor shapes, conditioning behavior, a backward step and saved-checkpoint inference. MoshiRAG's timed conditioning limitation identified above still needs an explicit training implementation. [Wrapped model and loading](https://raw.githubusercontent.com/kyutai-labs/moshi-finetune/main/finetune/wrapped_model.py)

### Immediate smoke data and independent evaluation

Plain words: DailyTalk can prove the trainer works today; it cannot prove generalization to new voices or realistic interruptions.

Kyutai publishes **DailyTalkContiguous**, approximately 13.8 GB, with stereo conversations, timestamps and a JSONL manifest already suited to Moshi. Select 16 manifest entries and their paired WAV/JSON files before requesting a larger snapshot. The dataset is labeled CC-BY-SA 4.0. Its conversations concatenate DailyTalk turns; the original corpus was performed by only **two actors**. It cannot supply three voice-disjoint folds, and concatenated turns do not establish naturally occurring overlap. Use it for ingestion, optimizer and speech-retention smoke tests only. [Kyutai dataset and files](https://huggingface.co/datasets/kyutai/DailyTalkContiguous/tree/main), [Kyutai dataset card](https://huggingface.co/datasets/kyutai/DailyTalkContiguous), [Original DailyTalk paper, collection procedure](https://arxiv.org/html/2207.01063)

Proposed split/quality contract for the actual experiment:

- Group the existing synthetic lessons by original conversation, scenario family, repeated answer bank, near-duplicate transcript, source utterance hashes and TTS voice ID **before** augmentation or splitting. Different filenames do not establish independence.
- For podcasts, keep whole episodes and related shows together; audit recurring speakers across feeds. Manifest speaker labels are episode-local, so show-disjoint alone cannot be advertised as voice-disjoint. Reserve unseen user voices and scenario families. If the assistant voice is deliberately fixed, state that controlled factor explicitly; evaluate new assistant voices separately only if claiming that ability.
- Reject missing provenance, corrupt/channel-misaligned audio, excessive leakage, unreliable word timing and undetected speaker swaps. Listen to a stratified sample including overlap. Publish accepted duration by split, voice and source plus rejected counts, and retain explicit quality approvals.
- Add separate, freshly authored delegation episodes with delayed returns, cancellation, correction and failure. Podcasts supply interaction behavior; they do not teach a task lifecycle. Hold out task families and result facts, and never expose a result in model context before its recorded arrival.

These are source-verified acquisition and training proposals, not completed downloads, permission determinations, ARM compatibility tests or training results. The next concrete experiment is a 16-example native Moshi training smoke plus a two-episode DuplexChat reconstruction smoke, independently of the live-baseline selection.

## Native Moshi smoke using our existing verified audio

Plain words: we can test whether a native speech model trains and reloads without waiting for new data. This is a plain Moshi engineering smoke; adapting the retrieval model still needs a different conditioning path.

### Pin the base and make the data conversion explicit

The verified HF revision of `kyutai/moshiko-pytorch-bf16` is **`2bfc9ae6e89079a5cc7ed2a68436010d91a3d289`**. It contains `model.safetensors`, `tokenizer-e351c8d8-checkpoint125.safetensors` and `tokenizer_spm_32k_3.model`; there is no `config.json`. It is ungated, with CC-BY 4.0 weights. [Model metadata and revision](https://huggingface.co/api/models/kyutai/moshiko-pytorch-bf16)

Trainer YAML exposes local model paths but no revision argument. Download those three files at the explicit revision, record hashes, and set all paths locally. Generate `base-config.json` from a deep copy of `moshi.models.loaders._lm_kwargs` in the pinned **plain Moshi** checkout. Record both code commit IDs and this config hash; no immutable code commit was resolved by this web audit. With `hf_repo_id: null` and all four local paths, the inspected loader avoids its config download/fallback. [Path arguments](https://raw.githubusercontent.com/kyutai-labs/moshi-finetune/main/finetune/args.py), [Checkpoint loader](https://raw.githubusercontent.com/kyutai-labs/moshi/main/moshi/moshi/models/loaders.py)

Conversion requirements from this repository's actual code:

- Select 16 conversations only after `data.synthetic.quality.checked_rows(..., require_timing=True)` passes against each source batch; retain source IDs and evidence hashes. Run the same check again at training startup.
- Stack `stem_assistant.wav` as channel zero and `stem_user.wav` as channel one. Both are already positioned on the same 24 kHz timeline. **Do not use `mixed.wav`**, which sums assistant and user. Initially select conversations without background speech; adding a third speaker needs an explicit microphone-mixture policy. Check channel lengths, clipping and duration against decoded samples, not just manifest fields.
- `timeline.actual.json` provides actual event offsets and sometimes `word_onsets_ms`; it does not provide full measured word intervals. The timing relabeler can interpolate missing onsets. Therefore do not manufacture word ends or present all stored onsets as measured. Re-annotate the final left stem with the official annotator, compare words against the verified audible text, and review truncated barge-in utterances. Preserve only spoken words, omitting our control-token labels and unsounded scripted tails.

Local evidence: [renderer](../../data/synthetic/render.py), [timeline schema](../../data/synthetic/timeline.py), [timing audit](../../data/synthetic/timing_audit.py), [quality startup gate](../../data/synthetic/quality.py). Use the stereo/sidecar schema above. Keep four additional approved conversations in a separate smoke evaluation manifest; these are engineering diagnostics until their family/voice independence is established.

### Concrete bounded configuration

Plain words: use one GPU, a small adapter and a few optimizer steps; verify every layer of the path before trying to improve quality.

Proposed `smoke.yaml`, with `/data/...` replaced by actual absolute paths. Each of the selected 16 conversations must fit within 100 seconds so four microbatches over four steps cover all 16 once with `shuffle: false`:

```yaml
data:
  train_data: /data/moshi-smoke/train16.jsonl
  eval_data: /data/moshi-smoke/eval4.jsonl
  shuffle: false
moshi_paths:
  hf_repo_id: null
  moshi_path: /data/moshiko-pinned/model.safetensors
  mimi_path: /data/moshiko-pinned/tokenizer-e351c8d8-checkpoint125.safetensors
  tokenizer_path: /data/moshiko-pinned/tokenizer_spm_32k_3.model
  config_path: /data/moshiko-pinned/base-config.json
full_finetuning: false
lora:
  enable: true
  rank: 8
  scaling: 2.0
  ft_embed: false
duration_sec: 100
batch_size: 1
num_microbatches: 4
max_steps: 4
gradient_checkpointing: true
param_dtype: bfloat16
first_codebook_weight_multiplier: 100.0
text_padding_weight: 0.5
optim:
  lr: 2.0e-6
  weight_decay: 0.1
  pct_start: 0.5
seed: 0
log_freq: 1
do_eval: true
eval_freq: 0
do_ckpt: true
ckpt_freq: 0
save_adapters: true
num_ckpt_keep: 1
run_dir: /data/moshi-smoke/run-unique
overwrite_run_dir: false
```

Inside an allocated GH200 worker, invoke `torchrun --standalone --nproc_per_node=1 train.py --config /data/moshi-smoke/smoke.yaml`. Single-GPU loading bypasses FSDP, but distributed initialization/barriers still require `torchrun`. For the smallest plumbing-only pass, use one microbatch per step and four steps; this does not cover all 16 conversations. Avoid arbitrary one/two-step OneCycle schedules without checking their phase boundaries. This rank/schedule is a proposed smoke setting, not a tuned recipe. [Training entry point](https://raw.githubusercontent.com/kyutai-labs/moshi-finetune/main/train.py), [Single-GPU branch](https://raw.githubusercontent.com/kyutai-labs/moshi-finetune/main/finetune/wrapped_model.py)

`eval_freq: 0` deliberately means final evaluation only. The trainer constructs a finite evaluation generator once and reuses it: a later evaluation can be empty. Its evaluator also counts the batch that triggers its 40-batch cap before breaking. Four evaluation conversations under 100 seconds avoid that cap. Before longer training, reconstruct the iterator for each evaluation and count processed batches correctly. [Evaluator](https://raw.githubusercontent.com/kyutai-labs/moshi-finetune/main/finetune/eval.py), [Finite dataset iteration](https://raw.githubusercontent.com/kyutai-labs/moshi-finetune/main/finetune/data/dataset.py)

### Strict load, backward and reload acceptance

Plain words: an output checkpoint file is insufficient; prove that intended weights changed and the saved adapter reproduces them.

1. Capture `load_state_dict`'s incompatible keys in a bounded trainer patch. For base loading into LoRA, permit missing keys **only** from the model's exact `lora_A`/`lora_B` parameter set, and require zero unexpected keys. Standard LoRA load hooks remap base `weight` to `frozen_W.weight`. Require no meta parameters, finite tensors and the expected shapes; log trainable parameter names/counts. [LoRA load hook](https://raw.githubusercontent.com/kyutai-labs/moshi/main/moshi/moshi/modules/lora.py)
2. Before stepping, require nonempty text/audio loss masks and finite losses. After backward require finite gradients and at least one nonzero adapter gradient; after stepping require changed adapter weights and unchanged frozen base weights. Zero gradients in some initial LoRA matrices can be legitimate, so do not demand every tensor change. Capture memory and elapsed time. Preserve the trainer's FP32 optimizer copies when adapting its code. [Mixed precision handling](https://raw.githubusercontent.com/kyutai-labs/moshi-finetune/main/finetune/mixed_precision.py)
3. Expected output is `run-unique/checkpoints/checkpoint_000004/consolidated/{lora.safetensors,config.json}`. The current save method writes weights/config only despite accepting an optimizer; it is not a resumable optimizer checkpoint. Compare saved adapter keys exactly against the trainable parameter set. [Checkpoint writer](https://raw.githubusercontent.com/kyutai-labs/moshi-finetune/main/finetune/checkpointing.py)
4. Start a fresh process. Build `CheckpointInfo` with the three original local files, saved `config.json`, and explicit `lora_weights=.../lora.safetensors`; call `get_moshi`. Ensure the saved config says LoRA enabled, rank 8, scaling 2. Pre-check that every expected adapter key is present, because the loader rejects unexpected adapter keys but does not explicitly reject all missing adapter keys. Compare fixed-input logits against the pre-save model within measured BF16 numerical tolerance, then generate audible speech from user-only audio. Loss/serialization success is not content quality. [Adapter loading](https://raw.githubusercontent.com/kyutai-labs/moshi/main/moshi/moshi/models/loaders.py)

### ARM compatibility and the retrieval-model boundary

Plain words: the old trainer and current speech package disagree even before GPU compatibility is considered.

Current plain Moshi requires `sphn>=0.2,<0.3`; the trainer pins `sphn==0.1.12`. This conflict exists even without MoshiRAG. Preserve NGC Torch 2.9, torchaudio and Triton, install a separate pinned Moshi/trainer checkout without dependency replacement, and test the selected sphn version's `dataset_jsonl`, `.seq`, `.shuffle` and returned `data/unpadded_len/start_time_sec/path` fields on one converted conversation. Then run CUDA SDPA forward/backward and Mimi encode before the full smoke. Plain Moshi attention uses Torch SDPA; it does not introduce ARC's mandatory xformers kernel. [Current Moshi dependencies](https://raw.githubusercontent.com/kyutai-labs/moshi/main/moshi/pyproject.toml), [Attention implementation](https://raw.githubusercontent.com/kyutai-labs/moshi/main/moshi/moshi/modules/transformer.py)

Do not silently substitute the concurrently installed MoshiRAG package under the same `moshi` import. Log module paths. In MoshiRAG, nonempty streaming reference conditioning asserts sequence length one, while ordinary training supplies whole sequences. Its conditioners also intentionally retain FP32, which conflicts with the plain trainer's blanket BF16 conversion/assertions. A retrieval adaptation needs a causal per-frame condition schedule plus an intentional mixed-dtype policy, validated against framewise inference; dropping the condition to make backward pass is not a delegation-training result. [MoshiRAG conditioning and model initialization](https://raw.githubusercontent.com/kyutai-labs/moshi-rag/main/moshi/moshi/models/lm.py)

This plan has not been executed. No additional data, model files, packages or GPU jobs were acquired or launched during this source audit.
