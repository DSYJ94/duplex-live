"""Opt-in GPU transcription of immutable native-selected task evidence.

Uses only the existing pinned local Whisper cache. No downloads, files, provider
credentials or speech/routing decisions. Runtime speed still requires GPU proof.
"""
from .task_context import AudioEvidence

MODEL = 'openai/whisper-large-v3-turbo'
REVISION = '41f01f3fe87f28c78e2fbf8b568835947dd65ed9'


class GPUContextASR:
    def __init__(self):
        from .native_app import ensure_guard
        ensure_guard()
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Context ASR requires the guarded single GPU; no Mac CPU fallback')
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL, revision=REVISION, local_files_only=True,
            torch_dtype=torch.float16, attn_implementation='eager').cuda().eval()
        model.requires_grad_(False)
        processor = AutoProcessor.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
        self.pipe = pipeline('automatic-speech-recognition', model=model,
            tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor,
            torch_dtype=torch.float16, device=0)
        self.torch = torch

    def __call__(self, evidence):
        if type(evidence) is not AudioEvidence:
            raise ValueError('Immutable bounded original audio required')
        from .native_app import ensure_guard
        import numpy as np
        ensure_guard()
        audio = np.frombuffer(evidence.pcm16, dtype='<i2').astype(np.float32) / 32768
        with self.torch.inference_mode():
            result = self.pipe(dict(array=audio, sampling_rate=16000),
                generate_kwargs=dict(language='en', task='transcribe', do_sample=False))
        ensure_guard()
        # Empty or malformed recognition fails the contextual transport validator;
        # there is no fabricated transcript, retry, or description-only fallback.
        return dict(text=result['text'], model=MODEL, revision=REVISION,
                    audio_receipt=evidence.receipt())
