"""Independent local CPU ASR of all preserved synthetic cue diagnostic audio.

Only the hash-pinned three inputs/nine native outputs are accepted. No owner
audio, new generation, training approval, GPU purchase, or reference-text prompt.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import wave

WEIGHT = 'aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a'
INPUT_MANIFEST = 'fce1acec7d28199cf49fa59a7b55d34a56997b065ba6a6886ce3fd8f243ba7d5'
REVIEW = '2607e4c5c7c03e6240661ef87eecd9202d1093a7861a10b4cd9abe34f13feab1'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def pcm16(path):
    with wave.open(str(path)) as stream:
        if (stream.getnchannels(), stream.getsampwidth(), stream.getframerate()) != (1, 2, 24000):
            raise ValueError('Original mono24k PCM16 diagnostic required')
        return stream.readframes(stream.getnframes())


def inputs(root):
    folder = root / 'runs/local-migration-20261021/keyword-control-inputs-v2'
    path = folder / 'manifest.json'
    if sha(path) != INPUT_MANIFEST:
        raise ValueError('Synthetic diagnostic input inventory changed')
    manifest = json.loads(path.read_text())
    clips = {}
    work = []
    for row in manifest['fragments']:
        path = folder / (row['id'] + '.wav')
        if sha(path) != row['sha256']:
            raise ValueError('Original synthetic fragment changed')
        clips[row['id']] = pcm16(path)
        work.append(dict(id='input-' + row['id'], path=path, sha256=row['sha256'],
                         kind='authored-synthetic-input', reference=row['text']))
    silence = bytes(24000 * 2)
    for row in manifest['cases']:
        path = folder / row['wav']
        key = 'ready' if row['id'] == 'ready-no-cue' else 'geronimo'
        expected = silence + clips[key] + silence * 10
        if row['cue_file_start_s'] is not None:
            expected += clips['cue'] + silence * 10
        if sha(path) != row['sha256'] or pcm16(path) != expected:
            raise ValueError('Full diagnostic input is not the original fragment/silence assembly')
    review_path = root / 'docs/NATIVE-KEYWORD-CONTROL-DIAGNOSTIC-2026-10-21.json'
    if sha(review_path) != REVIEW:
        raise ValueError('Original complete9 native diagnostic review changed')
    review = json.loads(review_path.read_text())
    if len(review['cases']) != 9 or review['qualification_or_adoption'] or review['owner_microphone_or_conversation_recorded']:
        raise ValueError('Complete original synthetic-only diagnostic required')
    for i, row in enumerate(review['cases']):
        path = root / row['local_directory'] / 'sampled-agent.wav'
        if sha(path) != row['audio_sha256']:
            raise ValueError('Original sampled native waveform changed')
        work.append(dict(id=f'output-{i:02d}-{row["id"]}', path=path,
                         sha256=row['audio_sha256'], kind='native-sampled-fixed-window-output',
                         reference=row['sampled_text'], variant=row['variant'],
                         nominal_quiet_s=row['quiet_interval_s']))
    if len(work) != 12:
        raise ValueError('All3 inputs and all9 native outputs required')
    return work


def run(root, weights, out, threads):
    import numpy as np
    import soundfile as sf
    import torch
    import whisper_timestamped as whisper
    from scipy.signal import resample_poly
    from data.synthetic.filter import norm_words, wer
    from evals.moshi_speech_audit import asr_window, restore_clock
    implementation = sha(Path(__file__))
    source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    work = inputs(root)
    if sha(weights) != WEIGHT:
        raise ValueError('Pinned independent Whisper turbo weight mismatch')
    out.mkdir(parents=True, exist_ok=False)
    progress = out / 'progress.json'
    def state(**values):
        values.update(pid=os.getpid(), at_utc=datetime.now(timezone.utc).isoformat())
        temporary = progress.with_suffix('.tmp')
        temporary.write_text(json.dumps(values, indent=2) + '\n')
        temporary.replace(progress)
    state(phase='loading_actual_cpu_asr', complete=0, expected=12)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    model = whisper.load_model(str(weights), device='cpu')
    rows = []
    for item in work:
        state(phase='transcribing', current=item['id'], complete=len(rows), expected=12)
        pcm, rate = sf.read(item['path'], dtype='float32')
        if rate != 24000 or pcm.ndim != 1 or not len(pcm) or not np.isfinite(pcm).all():
            raise ValueError('Invalid original diagnostic waveform')
        audio = resample_poly(pcm, 2, 3).astype(np.float32)
        window, head, tail, silent = asr_window(audio, True, True, True)
        started = time.monotonic()
        result = dict(text='', segments=[]) if silent else whisper.transcribe(
            model, window, language='en', vad=False, beam_size=5, best_of=5,
            temperature=0., fp16=False, verbose=None)
        result = restore_clock(result, head / 16000)
        seconds = time.monotonic() - started
        response = out / (item['id'] + '.asr.json')
        response.write_text(json.dumps(result, indent=2) + '\n')
        words = [w for segment in result['segments'] for w in segment.get('words', [])]
        row = {k: v for k, v in item.items() if k != 'path'}
        row.update(asr_file=response.name, asr_sha256=sha(response), asr_text=result['text'],
                   measured_words=words, raw_word_disagreement_diagnostic=wer(norm_words(item['reference']), norm_words(result['text'])),
                   seconds=seconds, audio_seconds=len(pcm) / rate, peak=float(np.abs(pcm).max()),
                   peak_meets_unchanged_ceiling=float(np.abs(pcm).max()) <= .995,
                   leading_silence_window_s=head / 16000, trailing_silence_window_s=tail / 16000,
                   word_times_restored_to_original_wav=True, internal_silence_removed=False,
                   expected_text_prompt_used=False, original_wav_unchanged=sha(item['path']) == item['sha256'])
        if item['kind'] == 'authored-synthetic-input':
            row['diagnostic_word_check_meets_005'] = row['raw_word_disagreement_diagnostic'] <= .05
        else:
            a, b = item['nominal_quiet_s']
            row['recognized_words_in_nominal_quiet'] = [w for w in words if w['start'] < b and w['end'] > a]
            row['full_answer_wer_scored'] = False
        if not row['original_wav_unchanged']:
            raise ValueError('Original waveform changed during independent ASR')
        rows.append(row)
        (out / 'results.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        print(json.dumps({k: row[k] for k in ('id', 'asr_text', 'seconds', 'peak')}), flush=True)
    if sha(Path(__file__)) != implementation:
        raise ValueError('Diagnostic implementation changed during actual ASR')
    report = dict(schema='dl-native-cue-independent-cpu-asr-v1', at_utc=datetime.now(timezone.utc).isoformat(),
                  code_sha256=implementation, source_commit=source_commit,
                  asr='Whisper turbo', weights_sha256=WEIGHT, device='cpu', precision='float32', threads=threads,
                  python=platform.python_version(), platform=platform.platform(),
                  packages={p: importlib.metadata.version(p) for p in ('torch','numpy','scipy','openai-whisper','whisper-timestamped','soundfile')},
                  input_manifest_sha256=INPUT_MANIFEST, original_review_sha256=REVIEW,
                  complete12=True, inputs=3, native_outputs=9, results=rows,
                  all_input_word_checks_meet005=all(r['diagnostic_word_check_meets_005'] for r in rows[:3]),
                  owner_recordings_used=False, api_calls=0, cloud_compute_purchased=False,
                  waveform_gain_or_bytes_changed=False, training_approved=False, native_behavior_qualified=False,
                  limits=['Development diagnostics only; original nine failures remain preserved.',
                          'Native WAV times are concatenated output-frame time, not network/playout latency.',
                          'Fixed windows may truncate answers; output disagreement is not full-answer WER qualification.',
                          'Independent ASR hearing the words does not prove the native model understood them.',
                          'No fullnative10, paired fresh40, actual teacher collection, numerical approval or new training claimed.'])
    (out / 'independent-asr.json').write_text(json.dumps(report, indent=2) + '\n')
    state(phase='complete', complete=12, expected=12, report_sha256=sha(out / 'independent-asr.json'))
    print(json.dumps(dict(complete12=True, all_input_word_checks_meet005=report['all_input_word_checks_meet005'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.threads <= 8:
        parser.error('Bound local CPU threads to1–8')
    run(args.root.resolve(), args.weights, args.out, args.threads)


if __name__ == '__main__':
    main()
