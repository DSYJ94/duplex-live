"""Offline complete-cohort speech/runtime triage; never changes a gate or audio.

Native frame timings exclude browser/network/device audio. Evidence of acoustic
EOS is not proof that words or naturalness are correct. No model/API execution.
"""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    if not values:
        return None
    ordered = sorted(values)
    if not all(math.isfinite(v) and v >= 0 for v in ordered):
        raise ValueError('Finite nonnegative measured timings required')
    return dict(count=len(ordered), median=statistics.median(ordered),
                p95=ordered[math.ceil(.95*len(ordered))-1], maximum=max(ordered))


def case_review(result, speech, frames):
    if (not result['complete'] or len(frames) != result['input_frames']
            or [f['sequence'] for f in frames] != list(range(len(frames)))
            or speech['id'] != result['id']):
        raise ValueError('Whole matching original case and trace required')
    events = [dict(frame=f['sequence'], **e) for f in frames
              for e in f['speech_completion']['events']]
    voiced = [f for f in frames if f['audio_samples'] > 0 and f['peak'] > .001]
    steps = [float(f['elapsed']) for f in frames]
    # Same emitted native PCM cutoff as collection; this never replaces word WER.
    no_voiced_word_failure = not speech['word_pass'] and not voiced
    return dict(id=result['id'], family=result['family'], word_pass=speech['word_pass'],
                raw_peak_pass=speech['raw_peak_pass'],
                generated_text=speech['references']['intended_user_facing'],
                recognized_text=speech['asr_text'],
                word_error_rate=speech['user_facing_text_wer'],
                voiced_frames=len(voiced), no_voiced_frames_but_ASR_word_failure=no_voiced_word_failure,
                no_voiced_failure_caveat='May include recognizer hallucination over low-level audio; failure is retained, not waived.',
                native_linguistic_turn_open_at_collection_end=result['native_turn_still_open'],
                pending_acoustic_at_collection_end=frames[-1]['speech_completion']['pending'],
                terminal_acoustic_eos_events=sum(e['kind']=='terminal_acoustic_piece' and e.get('reason')=='acoustic_eos' for e in events),
                acoustic_tail_cancellations=[e for e in events if e['kind']=='pending_acoustic_cancelled'],
                frame_step_seconds=stats(steps),
                voiced_frame_step_seconds=stats([float(f['elapsed']) for f in voiced]),
                frames_over_one_second=sum(v>1 for v in steps),
                maximum_synthetic_frame_backlog_seconds=max(0., max(f['wall_since_start']-(f['sequence']+1) for f in frames)),
                speech_or_reasoning_success_claimed=False)


def build(collection):
    root = Path(collection)
    report = json.loads((root/'native-report.json').read_text())
    result = json.loads((root/'results.json').read_text())
    if not report['complete'] or len(report['cases']) != 32 or len(result['cases']) != 32:
        raise ValueError('Complete32 native and independent speech cohort required')
    by = {r['id']: r for r in result['cases']}
    if len(by) != 32 or set(by) != {r['id'] for r in report['cases']}:
        raise ValueError('All32 distinct matching cases required')
    rows, all_frames, trace_hashes = [], [], {}
    for speech in report['cases']:
        trace = root/speech['id']/'trace.jsonl'
        frames = [json.loads(x) for x in trace.read_text().splitlines()]
        rows.append(case_review(by[speech['id']], speech, frames))
        all_frames.extend(frames); trace_hashes[speech['id']] = sha(trace)
    failures = [r for r in rows if not r['word_pass']]
    active = [f for f in all_frames if not f['is_listen']]
    stages = {}
    for name in ('cost_llm','cost_tts_prep','cost_tts','cost_token2wav'):
        values = [f['timings']['native_reported_wall_seconds'][name] for f in active]
        stages[name] = stats(values)
    return dict(schema='dl-native-product-offline-triage-v1', complete32=True,
                source=report['source'], source_report_sha256=sha(root/'native-report.json'),
                source_results_sha256=sha(root/'results.json'), trace_sha256=trace_hashes,
                summary=dict(cases=32, word_failures=len(failures),
                    raw_peak_failures=sum(not r['raw_peak_pass'] for r in rows),
                    word_failures_without_voiced_frames=sum(r['no_voiced_frames_but_ASR_word_failure'] for r in rows),
                    word_failures_with_closed_linguistic_turn=sum(not r['native_linguistic_turn_open_at_collection_end'] for r in failures),
                    word_failures_with_any_recorded_acoustic_EOS=sum(r['terminal_acoustic_eos_events']>0 for r in failures),
                    acoustic_tail_cancellations=sum(len(r['acoustic_tail_cancellations']) for r in rows),
                    cases_ending_with_pending_acoustic_tail=sum(r['pending_acoustic_at_collection_end'] for r in rows)),
                native_frame_step_seconds=stats([float(f['elapsed']) for f in all_frames]),
                native_speak_frame_step_seconds=stats([float(f['elapsed']) for f in active]),
                native_listen_frame_step_seconds=stats([float(f['elapsed']) for f in all_frames if f['is_listen']]),
                speak_frame_stage_wall_seconds=stages, cases=rows,
                interpretation='Observations identify separate text/control, acoustic and runtime questions; they do not establish a unique cause. Acoustic EOS does not prove correct speech.',
                live_browser_or_network_latency=False, native_model_executed=False,
                raw_audio_changed=False, original_gates_changed=False, qualified=False, adopted=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); output = Path(a.output)
    if output.exists():
        raise ValueError('Preserve prior reviews')
    report = build(a.collection)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('summary','native_frame_step_seconds','native_speak_frame_step_seconds','native_listen_frame_step_seconds')}))
