# 18 October: streaming speech dependency check

Plain words: this is a separate, disclosed hybrid fallback. A stronger text
controller would choose answers and tools, with Kyutai TTS speaking its output.
It does not establish native voice-model reasoning, request timing, microphone
acceptance or the owner's accepted voice identity. Native diagnostic work continues.

Plain words: primary-source metadata now pins the TTS model to
`f65439609986c392cb12df63938abcc550c3fb15`, published voice repository to
`323332d33f997de8394f24a193e1a76df720e01a`, and DSM example source to
`4c4f65e147df056adf3346290d64c7b9649b18c9`. The model uses separate speaker
embeddings. The example's published default embedding exists at
`expresso/ex03-ex01_happy_001_channel1_334s.wav.1e68beda@240.safetensors`.
[Model](https://huggingface.co/kyutai/tts-1.6b-en_fr/tree/f65439609986c392cb12df63938abcc550c3fb15),
[voices](https://huggingface.co/kyutai/tts-voices/tree/323332d33f997de8394f24a193e1a76df720e01a),
[example](https://github.com/kyutai-labs/delayed-streams-modeling/blob/4c4f65e147df056adf3346290d64c7b9649b18c9/scripts/tts_pytorch_streaming.py).

Plain words: the example pins Moshi0.2.11, whose pure Python wheel SHA256 is
`f426461f1a3fcffd5fb8d317c0a0a482e63766638c89a8ba9c21981b2c88a4f6`.
Its declared Torch requirement is>=2.2,<2.8; the current Isambard SIF uses2.9.
An import pass would not resolve that version mismatch or prove GPU compatibility.
Do not replace the working training environment to satisfy it. The bounded CPU
probe extracts only that verified wheel into its own output folder, imports
the actual TTS classes using existing dependencies, and records all versions.
No model weights or GPU are involved in this first check.
[Package metadata](https://pypi.org/pypi/moshi/0.2.11/json).

Plain words: inspection of the pinned package shows that TTS uses a state machine
to schedule words while generating acoustics. This is distinct from forcing
teacher words at their recorded times in the current native diagnostic. Any
later model trial must pass explicit local weight/config/tokenizer paths to
`CheckpointInfo.from_hf_repo`, because that package API has no revision argument;
speaker embeddings must also use explicit pinned local paths. Keep model loading
offline after asset retrieval. No runnable TTS system is claimed yet.
[Pinned wheel release](https://pypi.org/project/moshi/0.2.11/).
