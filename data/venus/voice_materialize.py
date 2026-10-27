"""Verify complete ElevenLabs raw sources and derive unapproved16k audit copies.

Raw PCM/WAV files and their receipts remain immutable. Resampling changes only
sample rate, never gain, words, folds or approval. A failed clip stays in the set.
"""
import argparse,array,hashlib,json,wave,subprocess
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def build(plan_path,source,output):
    import numpy as np
    plan_path,source,output=Path(plan_path),Path(source),Path(output)
    plan=json.loads(plan_path.read_text());receipt_path=source/'render-receipts.json';receipts=json.loads(receipt_path.read_text())
    assert receipts['complete'] and receipts['plan_sha256']==sha(plan_path)
    cases={r['id']:r for r in plan['cases']};expected={r['id'] for r in plan['cases'] if r['assistant_text']}|{'voice-v1-reference'}
    rows=receipts['utterances'];assert len(rows)==len(expected)==67 and {r['id'] for r in rows}==expected
    output.mkdir(exist_ok=False,parents=True);converted=[];total=0
    for row in rows:
        ident=row['id'];text=plan['reference_text'] if ident=='voice-v1-reference' else cases[ident]['assistant_text'];split='REFERENCE_ONLY' if ident=='voice-v1-reference' else cases[ident]['split']
        assert row['text']==text and row['split']==split and row['provider']=='ElevenLabs' and row['model_id']=='eleven_v3'
        folder=source/ident;original=folder/'source.wav';raw=(folder/'source.pcm').read_bytes()
        assert sha(original)==row['wav_sha256'] and hashlib.sha256(raw).hexdigest()==row['raw_pcm_sha256']
        with wave.open(str(original)) as w:
            assert (w.getframerate(),w.getnchannels(),w.getsampwidth())==(24000,1,2)
            assert w.readframes(w.getnframes())==raw
        samples=np.frombuffer(raw,dtype='<i2');assert samples.size>0
        peak=float(np.abs(samples.astype(np.int32)).max()/32768);assert peak==row['peak'] and (peak<=.995)==row['peak_pass']
        # System sample-rate conversion only; no gain/normalization parameters.
        dest=output/(ident+'.wav')
        subprocess.run(['/usr/bin/afconvert','-f','WAVE','-d','LEI16@16000','-c','1',str(original),str(dest)],check=True,capture_output=True)
        with wave.open(str(dest)) as w:
            assert (w.getframerate(),w.getnchannels(),w.getsampwidth())==(16000,1,2)
            pcm=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2')
        derived_peak=float(np.abs(pcm.astype(np.int32)).max()/32768)
        converted.append(dict(id=ident,text=text,split=split,wav=dest.name,sha256=sha(dest),sample_rate=16000,seconds=len(pcm)/16000,peak=derived_peak,source_wav_sha256=row['wav_sha256'],source_pcm_sha256=row['raw_pcm_sha256'],source_receipt_sha256=sha(folder/'receipt.json'),original_seconds=row['seconds'],original_peak=peak,original_peak_pass=row['peak_pass'],derived_peak_pass=derived_peak<=.995,provider=row['provider'],model_id=row['model_id'],voice_id=row['voice_id'],provenance_class=row['provenance_class'],provenance_authority=row['provenance_authority'],training_approved=False))
        total+=row['seconds']
    report=dict(complete=len(converted)==67,utterances=converted,original_audio_receipts_sha256=sha(receipt_path),plan_sha256=sha(plan_path),all_original_hashes_verified=True,all_original_peaks_pass=all(r['original_peak_pass'] for r in converted),all_derived_peaks_pass=all(r['derived_peak_pass'] for r in converted),original_seconds=total,resampling='Apple afconvert PCM16@16000; no gain/crop/text changes',silence_targets=[r['id'] for r in plan['cases'] if not r['assistant_text']],independent_words='PENDING_GPU_AUDIT',naturalness='PENDING_LISTENING',training_approved=False,subset_acceptance=False)
    (output/'render-receipts.json').write_text(json.dumps(report,indent=2)+'\n')
    return {k:v for k,v in report.items() if k not in ('utterances','silence_targets')}

if __name__=='__main__':
    p=argparse.ArgumentParser();[p.add_argument('--'+x,required=True) for x in ('plan','source','output')];a=p.parse_args();print(json.dumps(build(a.plan,a.source,a.output)))
