"""Approved clean native caches -> causal Context Spanning TRAIN sequences.

No random crop, new spoken labels, reference embeddings, or held-out examples.
The upstream assembler alone owns the span boundary/acoustic movement rules.
"""
import json
import math
from pathlib import Path
from evals.moshi_model_download import sha256


def load_training(caches, spm, prefix, max_frames=1100):
    import torch
    from contextspan.model.sequence_convention import assemble_training_sequence, context_span_ids
    from evals.recovery_data import validate_cache, validate_conversion
    from evals.recovery_encode import native_word_timing, marker_frame
    from evals.recovery_result_scope import private_result_frames
    from evals.personaplex_train import MODEL_HASHES
    examples, records, seen, waves = [], [], set(), set()
    for cache in caches:
        receipt = validate_cache(cache)
        data = validate_conversion(Path(receipt['conversion']))
        sources = {r['scenario_id']: r for r in data['accepted']}
        if receipt['frame_rate'] != 12.5 or any(
                receipt['model']['files'][n]['sha256'] != digest
                for n, digest in MODEL_HASHES.items() if n != 'model.safetensors'):
            raise RuntimeError('Native clean cache codec/vocabulary mismatch')
        for row in receipt['examples']:
            if row['split'] != 'train':
                continue
            source = sources[row['scenario_id']]
            if source['split'] != 'train' or row['scenario_id'] in seen or source['wav_sha256'] in waves:
                raise RuntimeError('Duplicate TRAIN scene or changed source membership')
            seen.add(row['scenario_id']); waves.add(source['wav_sha256'])
            side = json.loads(Path(source['path']).with_suffix('.json').read_text())
            sample = torch.load(cache / row['file'], map_location='cpu', weights_only=True)
            codes = sample['codes']
            if codes.shape[:2] != (1, 17) or codes.dtype != torch.int64:
                raise RuntimeError('Expected clean assistant-text/assistant-eight/user-eight codes')
            codes = codes[0].clone()
            if (codes[0] < -1).any() or (codes[0] >= spm.vocab_size()).any() or (
                    codes[1:] < -1).any() or (codes[1:] >= 2048).any():
                raise RuntimeError('Invalid native target range')
            timing = native_word_timing(codes[0].tolist(), side['alignments'], spm.encode, 12.5)
            markers = [marker_frame(e, 12.5) for e in side['timeline']['events'] if e['kind'] == 'marker']
            if (markers != sample['markers'] or markers != row['markers']
                    or len(set(markers)) != len(markers) or (
                    codes[0] == 4).nonzero().flatten().tolist() != sorted(markers)):
                raise RuntimeError('RET labels differ from approved silent control events')
            arrivals = [{k: v for k, v in a.items() if k != 'rows'} for a in sample['arrivals']]
            if arrivals != side['reference_arrivals']:
                raise RuntimeError('Cache result arrival/text differs from approved source')
            phrase = private_result_frames(side, codes[0].tolist(), spm.encode) if row['kind'] in (
                'private', 'clarify') else []
            spans = []
            for arrival in arrivals:
                ready = arrival['ready_at_s']
                if not math.isfinite(ready) or ready < 0 or not arrival['text'].strip():
                    raise RuntimeError('Invalid reference receipt')
                frame = math.ceil(ready * 12.5)
                if not 1 <= frame < codes.shape[1] or any(p < frame for p in phrase):
                    raise RuntimeError('Result cannot be inserted before checked fact targets')
                spans.append({'inject_frame': frame, 'reference': arrival['text']})
            if len({s['inject_frame'] for s in spans}) != len(spans):
                raise RuntimeError('Ambiguous simultaneous spans require explicit ordering')
            assembled, tmask, amask = assemble_training_sequence(
                codes, torch.ones(codes.shape[1], dtype=torch.bool), spans, spm)
            # Verify every real text token survives, with only masked span tokens added.
            if not torch.equal(assembled[0, tmask], codes[0]):
                raise RuntimeError('Span assembly changed real text')
            p = prefix.shape[1]
            assembled = torch.cat([prefix, assembled], 1)
            tmask = torch.cat([torch.zeros(p, dtype=torch.bool), tmask])
            amask = torch.cat([torch.zeros(p, dtype=torch.bool), amask])
            if assembled.shape[1] > max_frames:
                raise RuntimeError('Complete causal training sequence exceeds bounded context; no silent crop')
            mapping = [p + f + sum(len(context_span_ids(s['reference'], spm))
                       for s in spans if s['inject_frame'] <= f) for f in range(codes.shape[1])]
            if any(not tmask[mapping[f]] or assembled[0, mapping[f]] != codes[0, f]
                   for f in range(codes.shape[1])):
                raise RuntimeError('Real-to-virtual frame map mismatch')
            record = {'id': row['scenario_id'], 'kind': row['kind'], 'source_model': source['engine'],
                'source_split': 'train', 'cache': str(cache), 'cache_sha256': sha256(cache / 'cache.json'),
                'codes_sha256': row['sha256'], 'wav_sha256': source['wav_sha256'],
                'sidecar_sha256': source['sidecar_sha256'], 'ingestion_sha256': receipt['ingestion_sha256'],
                'frames': assembled.shape[1], 'prefix_frames': p, 'real_frames': codes.shape[1],
                'spans': spans, 'real_to_virtual': mapping, 'word_timing': timing,
                'checked_phrase_tokens': len(phrase),
                'arrival_provenance': 'approved actual GPT-Live receipt' if source['engine'] == 'gpt-live-1'
                    else 'approved declared synthetic schedule',
                'user_targets': 'clean separate user stem, including silence and overlap',
                'echo_only_user_targets_excluded': 0}
            examples.append({'codes': assembled, 'text_mask': tmask, 'audio_mask': amask,
                'user_target_mask': amask.clone(), 'record': record})
            records.append(record)
    if len(examples) < 16 or {r['kind'] for r in records} != {'private', 'simple', 'clarify', 'cancel'}:
        raise RuntimeError('Insufficient TRAIN retention/control coverage')
    return examples, records
