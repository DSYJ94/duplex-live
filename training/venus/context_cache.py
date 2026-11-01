"""Collect new36 native teachers and retain every old268 frozen-prefix cache.

Teacher forcing constructs labels, never demonstrates student task success.
Every new prefix gets exact segmented parity; old inputs are immutable links.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(a):
    import torch
    from portable.venus.native_app import ensure_guard
    from .native_cache import load_native, capture, replay_proof, training_input_proof
    from .prefix_cache import capture_prefix
    from .action_targets import masks, token_ids
    from .inference import MODEL_REVISION
    ensure_guard()
    root, old, output = map(Path, (a.timelines, a.old_cache, a.output))
    source = json.loads((root/'manifest.json').read_text())
    old_manifest = json.loads((old/'manifest.json').read_text())
    old_source = json.loads((Path(a.old_timelines)/'manifest.json').read_text())
    authority = json.loads(Path(a.admission).read_text())
    if sha(root/'manifest.json') != authority['new_timeline_manifest_sha256'] or sha(old/'manifest.json') != authority['old_prefix_manifest_sha256']:
        raise ValueError('Exact admitted old and new complete corpora required')
    if len(source['scenes']) != 36 or len(old_manifest['scenes']) != 268 or old_manifest['proof_only']:
        raise ValueError('No subset collection')
    if sum(r['fold']=='TRAIN' for r in source['scenes']) != 24 or sum(r['fold']=='TRAIN' for r in old_manifest['scenes']) != 216:
        raise ValueError('Original folds must remain intact')
    if output.exists():
        raise ValueError('Observe actual collection; never restart completed units')
    output.mkdir(parents=True)
    new = output/'new'
    new.mkdir()
    source_rows = old_source['scenes'] + source['scenes']
    source_manifest = dict(scenes=source_rows,native_speech_timing_schema='word-aligned-native-v1',old_manifest_sha256=sha(Path(a.old_timelines)/'manifest.json'),new_manifest_sha256=sha(root/'manifest.json'))
    (output/'timelines-manifest.json').write_text(json.dumps(source_manifest,indent=2)+'\n')
    ids = token_ids(json.loads(Path(a.tokenizer_config).read_text()))
    index = []
    model, duplex = load_native(a.model)
    rows = []
    start = time.monotonic()
    for i, row in enumerate(source['scenes']):
        ensure_guard()
        timeline_path = root/row['id']/'timeline.json'
        wav = root/row['id']/'user.wav'
        if sha(timeline_path) != row['timeline_sha256'] or sha(wav) != row['user_wav_sha256']:
            raise ValueError('New teacher inputs changed')
        timeline = json.loads(timeline_path.read_text())
        ex = capture(duplex,timeline,wav,retain_segments=i==0)
        proof = None
        if i == 0:
            proof = dict(native_segmented=replay_proof(duplex,ex),differentiable_input=training_input_proof(model,ex))
            if not proof['differentiable_input']['passed']:
                raise ValueError('Native input proof failed')
        result, prefix_proof = capture_prefix(model.llm,ex)
        path = new/(row['id']+'.pt')
        torch.save(result,path)
        masks_, spans = masks(result['targets'],result['decision_mask'],ids)
        item = dict(id=row['id'],fold=row['fold'],family=row['family'],file=str(path.relative_to(output)),sha256=sha(path),tokens=len(ex['embeddings']),labels=len(ex['targets']),native_prefix_proof=prefix_proof,native_input_proof=proof)
        rows.append(item)
        index.append(dict(id=row['id'],fold=row['fold'],family=row['family'],prefix_sha256=item['sha256'],target_sha256=hashlib.sha256(json.dumps(ex['targets'].tolist(),separators=(',',':')).encode()).hexdigest(),action_spans=spans,indices={k:v.nonzero().flatten().tolist() for k,v in masks_.items()}))
        print(json.dumps(dict(event='complete_new_native_teacher',completed=i+1,required=36,elapsed=time.monotonic()-start,id=row['id'],prefix_proof=prefix_proof)),flush=True)
    original_index = json.loads(Path(a.old_index).read_text())
    if original_index['prefix_manifest_sha256'] != sha(old/'manifest.json') or len(original_index['rows']) != 268:
        raise ValueError('Original exact all268 action index required')
    (output/'old').mkdir()
    for r in old_manifest['scenes']:
        src=old/r['file']
        link=output/'old'/Path(r['file']).name
        link.symlink_to(src.resolve())
        rows.append(dict(r,file=str(link.relative_to(output))))
    manifest=dict(old_manifest,scenes=rows,source_manifest_sha256=sha(output/'timelines-manifest.json'),context_source_manifest_sha256=sha(root/'manifest.json'),old_prefix_manifest_sha256=sha(old/'manifest.json'),prefix_producer=a.source,proof_only=False)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    combined=dict(complete=True,rows=original_index['rows']+index,prefix_manifest_sha256=sha(output/'manifest.json'),token_ids=ids,source_manifest_sha256=manifest['source_manifest_sha256'],old_index_sha256=sha(a.old_index),old_inputs_unchanged=True)
    (output/'action-index.json').write_text(json.dumps(combined,indent=2)+'\n')
    approval=dict(approved=True,train=240,dev=64,manifest_sha256=manifest['source_manifest_sha256'],new_timeline_manifest_sha256=sha(root/'manifest.json'),old_prefix_manifest_sha256=sha(old/'manifest.json'),complete_new36_native_replay=True,teacher_forced_not_student_behavior=True,model_revision=MODEL_REVISION,source=a.source)
    (output/'approval.json').write_text(json.dumps(approval,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('model','timelines','old-cache','old-timelines','old-index','tokenizer-config','admission','output','source'):
        p.add_argument('--'+key,required=True)
    main(p.parse_args())
