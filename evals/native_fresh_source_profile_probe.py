"""Replay all38 immutable ASR transcripts under a PROPOSED spelling profile.

This cannot approve source inputs or native inference. Original failed audits stay
unchanged; owner hearing and a separately approved procedure are still required.
"""
import argparse
import json
import math
from pathlib import Path
from evals.moshi_model_download import sha256
from evals.native_fresh_inputs_v2 import load_inputs, waveform, PLAN_SHA256

PROPOSAL = Path(__file__).with_name('fixtures')/'native-fresh2-source-reapproval-proposal-20261019.json'
PROPOSAL_SHA256 = '1575e4108ed6280d88ebad85efc617242e239139f36e68cf1b065d7df7009712'


def agreement(reference, transcript):
    from data.synthetic.filter import norm_words, wer
    a,b=norm_words(reference),norm_words(transcript)
    def canonical(words):
        return [part for word in words for part in
                (['look','up'] if word=='lookup' else ['colour' if word=='color' else word])]
    return dict(raw_wer=wer(a,b),proposed_spelling_wer=wer(canonical(a),canonical(b)))


def require_same_cases(manifest, diagnostic):
    if (len(manifest['rows'])!=20 or len(diagnostic['results'])!=20
            or [r['scenario_id'] for r in diagnostic['results']]!=[r['scenario_id'] for r in manifest['rows']]):
        raise ValueError('Complete same20 source/ASR inventory required; no subset')


def recompute(data,primary,diagnostic):
    import soundfile as sf
    from data.synthetic import filter as normalizer
    spec=json.loads(PROPOSAL.read_text())
    if sha256(PROPOSAL)!=PROPOSAL_SHA256:raise ValueError('Proposed finite source profile changed')
    m=load_inputs(data);q=json.loads(primary.read_text());d=json.loads(diagnostic.read_text())
    if (sha256(data/'inputs.json')!=spec['inputs_manifest_sha256']
            or sha256(primary)!=spec['original_failed_primary_quality_sha256']
            or sha256(diagnostic)!=spec['all20_other_asr_diagnostic_sha256']
            or q['schema']!='dl-native-fresh-quality-v2' or q['passed']
            or len(q['results'])!=20 or q['threshold']!=.05 or q['peak_bar']!=.995
            or d['schema']!='dl-native-fresh2-full20-asr-diagnostic-v1' or not d['diagnostic_only']
            or d['asr_model']!='turbo' or d['asr_weight_sha256']!=spec['asr_weight_sha256']
            or d['plan_sha256']!=PLAN_SHA256 or d['inputs_manifest_sha256']!=sha256(data/'inputs.json')
            or d['original_failed_quality_sha256']!=sha256(primary)
            or d['inference_approved'] or d['training_approved'] or d['expected_text_prompt_used']
            or d['original_waveforms_changed'] or d['generation_calls'] or d['rerolls']
            or d['passing_subset_accepted'] or not d['original_failed_verdict_preserved']
            or d['threshold']!=.05 or d['peak_bar']!=.995 or d['complete_clips']!=38):
        raise ValueError('Retain both exact complete failed audits; no replacement approval')
    require_same_cases(m,d);rows=[]
    for source,asr in zip(m['rows'],d['results']):
        ref=json.loads(Path(source['reference_path']).read_text());clips=[]
        if asr['wav_sha256']!=source['wav_sha256'] or len(asr['clips'])!=len(ref['utterances']):
            raise ValueError('Missing or changed source clip')
        for i,(u,c) in enumerate(zip(ref['utterances'],asr['clips'])):
            wav=Path(source['path']).with_name(u['source_file']);pcm,rate=sf.read(wav,dtype='float32')
            name=source['scenario_id']+'-%d-asr.json'%i;p=diagnostic.parent/name
            if (c['ordinal']!=i or c['asr_file']!=name or sha256(p)!=c['asr_sha256']
                    or sha256(wav)!=c['raw_wav_sha256']):raise ValueError('Raw ASR/audio changed')
            t=json.loads(p.read_text());words=[w for s in t['segments'] for w in s.get('words',[])]
            valid=bool(words) and all(math.isfinite(w['start']) and math.isfinite(w['end'])
                and 0<=w['start']<=w['end']<=len(pcm)/rate+.02
                for s in t['segments'] for w in [s]+s.get('words',[]))
            wave=waveform(pcm,rate);scores=agreement(u['text'],t['text'])
            if (c['valid_bounds']!=valid or c['waveform']!=wave
                    or c['agreement']['raw_wer']!=scores['raw_wer']):raise ValueError('Raw diagnostic does not match physical source')
            clips.append(dict(ordinal=i,raw_wav_sha256=sha256(wav),raw_asr_sha256=sha256(p),
                raw_wer=scores['raw_wer'],proposed_spelling_wer=scores['proposed_spelling_wer'],
                valid_bounds=valid,waveform=wave,diagnostic_meets_proposed_bars=valid and wave['passed']
                and math.isfinite(scores['proposed_spelling_wer']) and 0<=scores['proposed_spelling_wer']<=.05))
        rows.append(dict(scenario_id=source['scenario_id'],wav_sha256=source['wav_sha256'],clips=clips,
            diagnostic_meets_proposed_bars=all(c['diagnostic_meets_proposed_bars'] for c in clips)))
    assert len(rows)==20 and sum(len(r['clips']) for r in rows)==38
    report=dict(schema='dl-native-fresh2-proposed-profile-diagnostic-v1',diagnostic_only=True,
        proposal_sha256=PROPOSAL_SHA256,source_code_sha256=sha256(Path(__file__)),
        normalizer_sha256=sha256(Path(normalizer.__file__)),inputs_manifest_sha256=sha256(data/'inputs.json'),
        original_failed_primary_quality_sha256=sha256(primary),complete_turbo_diagnostic_sha256=sha256(diagnostic),
        results=rows,complete_cases=20,complete_clips=38,threshold=.05,peak_bar=.995,
        finite_spellings=spec['finite_canonicalization'],raw_wer_preserved=True,room_rune_equivalent=False,
        all20_diagnostic_meets_proposed_bars=all(r['diagnostic_meets_proposed_bars'] for r in rows),
        original_failed_verdict_preserved=True,input_quality_approved=False,inference_approved=False,
        owner_hearing_received=False,owner_scope_approval_received=False,training_approved=False,
        generation_calls=0,rerolls=0,original_audio_changed=False,passing_subset_accepted=False)
    return report


def run(data,primary,diagnostic,out):
    report=recompute(data,primary,diagnostic)
    out.mkdir(parents=True,exist_ok=False);(out/'diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='results'}),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('data','primary','diagnostic','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();run(a.data,a.primary,a.diagnostic,a.out)


if __name__=='__main__':main()
