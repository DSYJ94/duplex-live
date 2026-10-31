"""Pinned32 zero-update comparison: original release or explicit406 host repair."""
import argparse
import asyncio
import gc
import hashlib
import json
import time
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def validate_reference(manifest, root, cases):
    """Bind every input and original reference, including unsuccessful cases."""
    root = Path(root)
    assert manifest['required_cases'] == len(cases) == len(manifest['cases']) == 32
    expected = {x['id']: x for x in manifest['cases']}
    assert len(expected) == 32 and set(expected) == {x['id'] for x in cases}
    assert sha(root/'results.json') == manifest['reference_collection_sha256']
    assert sha(root/'paired-native-report.json') == manifest['reference_speech_report_sha256']
    assert sha(root/'paced-continuation/profile.json') == manifest['reference_profile_sha256']
    report = read(root/'paired-native-report.json')
    assert report['complete'] and len(report['cases']) == 64
    audits = {x['id']: x for x in report['cases'] if x['condition'] == 'paced-continuation'}
    assert set(audits) == set(expected)
    for case in cases:
        row = expected[case['id']]
        assert (case['input_sha256'], case['frames'], case['family']) == (row['input_sha256'], row['input_frames'], row['family'])
        folder = root/'paced-continuation'/case['id']
        assert sha(folder/'trace.jsonl') == row['reference_trace_sha256']
        assert sha(folder/'joined-emitted.wav') == row['reference_wave_sha256'] == audits[case['id']]['joined_wav_sha256']
        for chunk in audits[case['id']]['source_chunks']:
            assert Path(chunk['file']).name == chunk['file']
            assert sha(folder/chunk['file']) == chunk['sha256']
    return read(root/'paced-continuation/profile.json')


async def main(a):
    import numpy as np
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    from transformers import set_seed
    from .behavior import preflight, run_case
    from .behavior_review import cue_frame
    from .spoken_audit import references, REVISION
    from .voice_compare import recognizer
    from data.venus.pacing import words
    from data.venus.repair import wer
    from portable.venus.native_app import NativeEngine, ensure_guard

    manifest = read(a.manifest)
    assert manifest['new_optimizer_updates'] == manifest['new_api_calls'] == 0
    cases = preflight(a.plan, a.timelines, a.approval)
    for case in cases: case['evaluation_set'] = 'fresh24'
    retained = [x for x in preflight(a.old_plan, a.old_timelines, a.old_approval) if x['family'] == 'ordinary']
    assert len(cases) == 24 and len(retained) == 8
    for case in retained: case['evaluation_set'] = 'old8ordinary-retention'
    cases += retained
    profile = validate_reference(manifest, a.reference_collection, cases)
    stop = read(a.lease)['stop_unix']
    assert stop-time.time() > sum(x['frames'] for x in cases)*1.5+600
    quality = read(a.word_quality)
    assert quality['all_passed'] and quality['passed'] == quality['requested'] == 218
    recognized = {x['source_sha256']: x for x in quality['rows']}
    out = Path(a.output); out.mkdir(exist_ok=False)
    voice = profile['voice_reference']
    repair = getattr(a, 'private_action_repair', False)
    checkpoint = getattr(a, 'candidate_checkpoint', None)
    assert bool(checkpoint) == repair, 'Only the declared406 runtime comparison may load an adapter'
    if repair: assert sha(checkpoint) == manifest['reference_checkpoint_sha256']
    engine = NativeEngine(a.model, adapter_checkpoint=checkpoint, voice_reference=voice['path'], voice_reference_sha256=voice['sha256'], tts_attention='released', nonspoken_actions=repair)
    fields = ('seed','torch','cuda','precision','attention','input_sample_rate','output_sample_rate','system_prompt','decode_mode','max_new_speak_tokens_per_chunk','temperature','top_k','top_p','listen_prob_scale','tts_attention','voice_reference','nonspoken_action_channel')
    assert all(engine.profile[k] == profile[k] for k in fields if not (repair and k=='nonspoken_action_channel')), 'Comparison profile changed'
    assert engine.profile['nonspoken_action_channel'] == repair
    assert engine.profile['optimizer_updates'] == (406 if repair else 0)
    if not repair: assert not any(type(m).__name__ == 'LoRALinear' for m in engine.model.modules())
    condition='private-action-repair406' if repair else 'untouched-release'
    (out/'profile.json').write_text(json.dumps(dict(engine.profile, condition=condition, source=a.source, model_revision=manifest['model_revision'], qualified=False), indent=2)+'\n')
    results = []
    for case in cases:
        ensure_guard(); set_seed(42)
        row = dict(condition=condition, evaluation_set=case['evaluation_set'], **await run_case(engine, case, out, stop))
        if case['family'] == 'quiet':
            turn = case['user_turns'][-1]; eligible, end = cue_frame(turn, recognized[turn['sha256']])
            trace = [json.loads(x) for x in (out/case['id']/'trace.jsonl').read_text().splitlines()]
            voiced = lambda t: t['audio_samples'] > 0 and t['peak'] > .001
            row.update(cue_word_end_seconds=end, first_chunk_that_has_heard_cue=eligible, voiced_frames_before_cue_word=sum(voiced(t) and t['sequence'] < eligible for t in trace), voiced_frames_after_cue_word=sum(voiced(t) and t['sequence'] >= eligible for t in trace))
        results.append(row)
        (out/'results.json').write_text(json.dumps(dict(cases=results, required_cases=32, qualified=False), indent=2)+'\n')
        print(json.dumps(dict(event='untouched_case_complete', completed=len(results), required=32)), flush=True)
    del engine; gc.collect(); torch.cuda.empty_cache()
    pipe = recognizer(); audits = []
    for result in results:
        ensure_guard(); assert stop-time.time() > 180
        folder = out/result['id']; trace = [json.loads(x) for x in (folder/'trace.jsonl').read_text().splitlines()]
        chunks = [sf.read(folder/t['chunk_file'], dtype='float32')[0] for t in trace]
        audio = np.concatenate(chunks); assert np.isfinite(audio).all()
        text = pipe(dict(array=resample_poly(audio,2,3).astype(np.float32), sampling_rate=16000), generate_kwargs=dict(language='en', task='transcribe', do_sample=False))['text'] if np.any(audio) else ''
        ref = references(trace); error = wer(' '.join(words(ref['intended_user_facing'])), ' '.join(words(text)))
        peak = float(np.abs(audio).max()) if len(audio) else 0
        wav = folder/'joined-emitted.wav'; sf.write(wav, audio, 24000, subtype='FLOAT')
        audits.append(dict(id=result['id'], condition=condition, references=ref, asr_text=text, user_facing_text_wer=error, word_pass=error<=.05, raw_peak=peak, raw_peak_pass=peak<=.995, joined_wav_sha256=sha(wav), source_chunks=[dict(file=t['chunk_file'],sha256=sha(folder/t['chunk_file'])) for t in trace], qualified=False))
        (out/'speech-audit-progress.json').write_text(json.dumps(dict(completed=len(audits), required=32, cases=audits), indent=2)+'\n')
    assert len(results) == len(audits) == 32
    (out/'foundation-report.json').write_text(json.dumps(dict(complete=True, cases=audits, source=a.source, new_native_generations=32, reference_cases=32, reference_collection=a.reference_collection, reference_collection_sha256=manifest['reference_collection_sha256'], optimizer_updates=0, external_model_calls=0, asr_revision=REVISION, word_limit=.05, peak_limit=.995, qualified=False, adopted=False, semantic_and_naturalness_review_required=True), indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('model','manifest','reference-collection','plan','timelines','approval','old-plan','old-timelines','old-approval','word-quality','output','lease','source'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--private-action-repair',action='store_true')
    p.add_argument('--candidate-checkpoint')
    asyncio.run(main(p.parse_args()))
