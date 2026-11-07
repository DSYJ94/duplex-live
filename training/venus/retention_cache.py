"""Collect all96 independently reviewed native histories, preserving old304.

Full native embeddings/segments/labels; no frozen-prefix shortcuts or updates.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main(a):
    import torch
    from portable.venus.native_app import ensure_guard
    from data.venus.retention_admission import admit
    from data.venus.reviewer_model import sha
    from .native_cache import load_native,capture,replay_proof,training_input_proof
    from .large_inputs import full_native
    from .action_targets import masks,token_ids
    from .inference import MODEL_REVISION
    ensure_guard()
    if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=a.source:
        raise ValueError('Actual Git source required')
    admission=admit(a.timelines,a.semantic,a.semantic_sha)
    manifest=json.loads((a.timelines/'manifest.json').read_text())
    if (sha(a.model/'config.json')!='98edf44576907c9d175df9d21d7ca54ef4e4766b390daa0b6e8a56ee04545ea7'
            or sha(a.model/'tokenizer.json')!=manifest['tokenizer_sha256']
            or manifest['native_speech_timing_schema']!='word-aligned-native-v1'):
        raise ValueError('Preserved native architecture/tokenizer and measured timing required')
    if a.output.exists():raise ValueError('Preserve previous native collection, never blindly repeat it')
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'semantic-admission.json').write_text(json.dumps(admission,indent=2)+'\n')
    ids=token_ids(json.loads((a.model/'tokenizer_config.json').read_text()))
    model,duplex=load_native(a.model)
    rows=[];index=[];started=time.monotonic()
    reused=[]
    prior=Path(a.resume_cache) if getattr(a,'resume_cache',None) else None
    from .retention_inputs import validate_proof
    prior_execution=None
    if prior:
        old_lease=prior.parent/'execution-8177652/lease.json'
        prior_execution=json.loads(old_lease.read_text())
        if prior_execution['job_id']!='8177652' or prior_execution['source']!='219613a9da1747b36f7c8d2e249b54ca5646d8ba':
            raise ValueError('Exact original partial collection producer required')
    for row in manifest['scenes']:
        ensure_guard()
        folder=a.timelines/row['id'];path=folder/'timeline.json';wav=folder/'user.wav'
        if sha(path)!=row['timeline_sha256'] or sha(wav)!=row['user_wav_sha256']:
            raise ValueError('Previously reviewed teacher changed')
        timeline=json.loads(path.read_text())
        prior_proof=prior/(row['id']+'.proof.json') if prior else None
        if prior_proof and prior_proof.exists():
            saved=json.loads(prior_proof.read_text());old=prior/saved['file']
            if (saved['id']!=row['id'] or saved['fold']!='TRAIN' or saved['family']!=row['family']
                    or saved['timeline_sha256']!=row['timeline_sha256'] or saved['user_wav_sha256']!=row['user_wav_sha256']
                    or Path(saved['file']).name!=saved['file'] or sha(old)!=saved['sha256']):
                raise ValueError('Exact original complete partial native artifact required')
            validate_proof(saved['native_input_proof'])
            os.link(old,a.output/saved['file'])
            (a.output/prior_proof.name).write_bytes(prior_proof.read_bytes())
            example=torch.load(old,map_location='cpu',weights_only=False)
            selected,spans=masks(example['targets'],example['decision_mask'],ids)
            rows.append(saved)
            index.append(dict(id=row['id'],fold='TRAIN',family=row['family'],cache_sha256=saved['sha256'],
                target_sha256=hashlib.sha256(json.dumps(example['targets'].tolist(),separators=(',',':')).encode()).hexdigest(),
                action_spans=spans,indices={k:v.nonzero().flatten().tolist() for k,v in selected.items()}))
            reused.append(dict(id=row['id'],original_artifact=str(old),artifact_sha256=saved['sha256'],proof_sha256=sha(prior_proof)))
            continue
        example=full_native(capture(duplex,timeline,wav,retain_segments=True))
        if any(example[k]!=row[k] for k in ('id','fold','family')) or example['fold']!='TRAIN':
            raise ValueError('Exact new TRAIN identity required')
        proof=dict(native_segmented=replay_proof(duplex,example),
                   differentiable_input=training_input_proof(model,example))
        if not proof['differentiable_input']['passed']:raise ValueError('Native training-input proof failed')
        selected,spans=masks(example['targets'],example['decision_mask'],ids)
        artifact=a.output/(row['id']+'.pt');torch.save(example,artifact)
        item=dict(id=row['id'],fold='TRAIN',family=row['family'],file=artifact.name,
                  sha256=sha(artifact),tokens=len(example['embeddings']),labels=len(example['targets']),
                  native_input_proof=proof,timeline_sha256=row['timeline_sha256'],
                  user_wav_sha256=row['user_wav_sha256'])
        rows.append(item)
        index.append(dict(id=row['id'],fold='TRAIN',family=row['family'],cache_sha256=item['sha256'],
            target_sha256=hashlib.sha256(json.dumps(example['targets'].tolist(),separators=(',',':')).encode()).hexdigest(),
            action_spans=spans,indices={k:v.nonzero().flatten().tolist() for k,v in selected.items()}))
        (a.output/(row['id']+'.proof.json')).write_text(json.dumps(item,indent=2)+'\n')
        print(json.dumps(dict(event='complete_native_teacher',completed=len(rows),required=96,
                              id=row['id'],seconds=time.monotonic()-started)),flush=True)
    if len(rows)!=96:raise ValueError('No subset native approval')
    (a.output/'resumed-artifacts.json').write_text(json.dumps(dict(original_cache=str(prior) if prior else None,
        original_execution=prior_execution,reused_complete_native_artifacts=reused,actual_new_captures=96-len(reused),original_files_unchanged=True),indent=2)+'\n')
    result=dict(schema='dl-retention96-native-cache-v1',source=a.source,scenes=rows,
        model_revision=MODEL_REVISION,source_manifest_sha256=admission['timeline_manifest_sha256'],
        semantic_report_sha256=a.semantic_sha,train=96,dev=0,complete96=True,
        native_speech_timing_schema='word-aligned-native-v1',forward_path='native-segmented-bptt-v1',
        teacher_forced_not_student_behavior=True,optimizer_updates=0,training_approved=False)
    (a.output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    (a.output/'action-index.json').write_text(json.dumps(dict(rows=index,token_ids=ids,
        manifest_sha256=sha(a.output/'manifest.json'),complete96=True),indent=2)+'\n')
    (a.output/'replay-approval.json').write_text(json.dumps(dict(complete96=True,
        all96_native_replay_and_input_checks=True,manifest_sha256=sha(a.output/'manifest.json'),
        action_index_sha256=sha(a.output/'action-index.json'),semantic_admission=admission,
        source=a.source,original304_untouched=True,new_optimizer_updates=0,
        composite_training_admission_pending=True,training_approved=False),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','semantic-sha'):p.add_argument('--'+name,required=True)
    for name in ('model','timelines','semantic','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--resume-cache',type=Path)
    main(p.parse_args())
