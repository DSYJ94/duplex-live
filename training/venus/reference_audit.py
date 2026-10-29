"""Independent full-waveform word audit for an existing synthetic reference only."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import numpy as np,soundfile as sf
from portable.venus.native_app import ensure_guard
from .voice_compare import recognizer,REVISION
from data.venus.repair import wer


def main(a):
    ensure_guard()
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=10)
    if active.strip():raise RuntimeError('Reference recognizer must run before the native model, without overlap')
    meta=json.loads(Path(a.metadata).read_text());audit_path=Path(a.original_audit)
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert digest(a.reference)==meta['derived_sha256'] and digest(audit_path)==meta['independent_original_word_audit_sha256']
    original=json.loads(audit_path.read_text());assert original['source_sha256']==meta['source_sha256'] and original['word_error_rate']==0
    wave,sr=sf.read(a.reference,dtype='float32');assert sr==16000 and wave.ndim==1 and np.isfinite(wave).all()
    pipe=recognizer();recognized=pipe(dict(array=wave,sampling_rate=16000),generate_kwargs=dict(language='en',task='transcribe',do_sample=False))['text']
    error=wer(original['authored'],recognized);peak=float(np.abs(wave).max())
    row=dict(source_sha256=meta['source_sha256'],derived_sha256=meta['derived_sha256'],expected=original['authored'],recognized=recognized,word_error_rate=error,peak=peak,passed=error<=.05 and peak<=.995,recognizer='openai/whisper-large-v3-turbo',revision=REVISION,full_waveform=True,no_transcript_hint=True,no_gain_or_clamp=True,reference_only=True,cohort_training_approved=False,native_voice_qualified=False)
    Path(a.output).write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row),flush=True)
    assert row['passed'],'No custom native reference trial from failed source words or clipping'

if __name__=='__main__':
    p=argparse.ArgumentParser();[p.add_argument('--'+k,required=True) for k in ('reference','metadata','original-audit','output')];main(p.parse_args())
