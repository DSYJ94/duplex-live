"""Independent CPU speech diagnosis of one preserved replacement recording.

Uses the same frozen whole32 recognition policy. A component pass cannot qualify
a corpus, approve semantic content, permit conversion or authorize training.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path

from evals import native_reasoning_teacher_speech_v2 as audit
from evals.native_reasoning_source_quality import assess
from evals.native_reasoning_teacher_quality import expected_output


def run(a):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    import torch
    import whisper_timestamped as whisper
    wav = a.recording / 'model_out.wav'
    packages = {p: importlib.metadata.version(p) for p in audit.PACKAGES}
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_available()
            or packages != audit.PACKAGES or audit.sha(a.weights) != audit.WEIGHT
            or wav.is_symlink() or audit.sha(wav) != a.wav_sha256):
        raise ValueError('Pinned CPU recognizer and unchanged original recording required')
    a.out.mkdir(parents=True, exist_ok=False)
    pcm, rate = sf.read(wav, dtype='float32'); spans = audit.regions(pcm, rate)
    torch.set_num_threads(8); torch.set_num_interop_threads(1)
    model = whisper.load_model(str(a.weights), device='cpu')
    # The independent recognizer receives samples and waveform-derived regions
    # only. Actual provider words are read AFTER recognition has completed.
    raw = whisper.transcribe(model, resample_poly(pcm,2,3).astype(np.float32),
                             language='en', vad=[tuple(r) for r in spans],
                             best_of=5, beam_size=5, temperature=0., fp16=False, verbose=None)
    audit.write(a.out / 'asr.json', raw)
    raw = json.loads((a.out / 'asr.json').read_text())
    text = expected_output(a.recording); verdict = assess(text, raw, pcm, rate)
    if audit.sha(wav) != a.wav_sha256: raise ValueError('Original waveform changed')
    result = dict(schema='dl-native-reasoning-component-speech-diagnostic-v1',
                  code_sha256=audit.sha(Path(__file__)), policy=audit.POLICY,
                  asr_weight_sha256=audit.WEIGHT, asr_packages=packages,
                  original_wav_sha256=a.wav_sha256, original_clock_regions=spans,
                  asr_sha256=audit.sha(a.out / 'asr.json'),
                  actual_provider_transcript_sha256=audit.sha(a.recording / 'transcripts.json'),
                  component_speech_verdict=verdict, whole32_approved=False,
                  semantic_approved=False, training_approved=False,
                  original_waveform_changed=False, provider_requests=0)
    audit.write(a.out / 'component-speech.json', result)
    print(json.dumps(result)); return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('recording','weights','out'): parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--wav-sha256',required=True)
    run(parser.parse_args())
