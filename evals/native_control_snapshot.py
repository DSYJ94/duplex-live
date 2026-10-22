"""Verify the exact whole40 workset whose original approval chain completed.

The inventory and three completed gate roots are pinned in source, not supplied
approval flags. Rehash EVERY captured/derived file and independently recompute
the current40 word/control, semantic-response and conversion checks. Historical
failed ancestors stay byte-bound without recursively replaying their derivations.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from data.synthetic.native_control_render import sha256

INVENTORY_SHA256='79401e2c1051361d150da644250e8acf87814bda1c48b167af8a34345214fa8f'
QUALITY=Path('runs/native-control3-webrtc-reapproval-20261022-v1/quality/teacher-quality.json')
SEMANTIC=Path('runs/native-control3-webrtc-postteacher-20261022-v1/semantic/semantic.json')
DATA=Path('runs/native-control3-webrtc-postteacher-20261022-v1/converted')
GATES={str(QUALITY):'1f3f4b45e847e7a02b4e781fa88f4802029cd1c9982148560e1b91c40ba76858',
       str(SEMANTIC):'f03bebe4dda16d4a159bdc0e8cf1127f772fb01e543a8af90c392dd0671aca36',
       str(DATA/'ingestion.json'):'93e2eb6e90c00d3414797bc977fb01b5c26f2f5892f30656e1c3c6a538e1295e'}


def verify_files(inventory,root):
    if sha256(inventory)!=INVENTORY_SHA256:
        raise ValueError('Only the source-pinned independently approved complete workset is accepted')
    spec=json.loads(inventory.read_text());rows=spec['files'];seen=set();size=0
    if spec['file_count']!=7797 or spec['uncompressed_bytes']!=8000842614 or len(rows)!=7797:
        raise ValueError('Exact complete7797-file workset required')
    for row in rows:
        name=Path(row['path']);path=root/name
        if (name.is_absolute() or '..' in name.parts or len(name.parts)<3 or name.parts[0]!='runs'
                or not name.parts[1].startswith('native-control3-') or row['path'] in seen
                or path.is_symlink() or not path.resolve().is_relative_to(root.resolve())
                or not path.is_file() or path.stat().st_size!=row['bytes'] or sha256(path)!=row['sha256']):
            raise ValueError('Frozen complete workset file changed, missing or outside snapshot')
        seen.add(row['path']);size+=row['bytes']
    if size!=8000842614 or not set(GATES).issubset(seen):
        raise ValueError('Complete gate closure missing')
    for path,digest in GATES.items():
        if sha256(root/path)!=digest:raise ValueError('Completed original gate root changed')
    return spec


def verify(data,inventory):
    import numpy as np
    import soundfile as sf
    from data.synthetic.native_control_repair import build
    from evals import native_control_teacher_scoped_quality as decision
    from evals import native_control_webrtc_quality as quality
    from evals import native_control_semantic as semantic,native_control_convert as conversion
    if data.resolve()!=DATA.resolve():raise ValueError('Snapshot is bound to one exact complete conversion')
    verify_files(inventory,Path.cwd())
    receipt=json.loads((data/'ingestion.json').read_text());q=json.loads(QUALITY.read_text());sem=json.loads(SEMANTIC.read_text())
    raw,source_audit,batch=(Path(receipt[k]) for k in ('raw','source_audit','batch'))
    plan=build()['cases'];ids=[s['id'] for s in plan];checks=json.loads(source_audit.read_text())['results']
    if (q['code_pins']!=quality.pins() or sem['code_sha256']!=sha256(Path('evals/native_control_semantic.py'))
            or receipt['code_sha256']!=sha256(Path('evals/native_control_convert.py'))
            or q['passed'] is not True or sem['passed'] is not True or receipt['complete40'] is not True
            or len(ids)!=40 or [r['id'] for r in q['results']]!=ids
            or [r['id'] for r in sem['results']]!=ids or [r['scenario_id'] for r in receipt['accepted']]!=ids
            or Counter(r['family'] for r in receipt['accepted'])!=Counter({s['family']:8 for s in plan})
            or any(s['split']!='train' for s in plan)
            or receipt['training_approved'] is not False or receipt['optimizer_updates']!=0):
        raise ValueError('Whole40 source, verifier code, family or gate identity changed')
    for scene,row,check,converted,sem_row in zip(plan,q['results'],checks,receipt['accepted'],sem['results']):
        folder=batch/scene['id'];asr_path=QUALITY.parent/row['asr_file']
        if row['asr_file']!=scene['id']+'.asr.json' or sha256(asr_path)!=row['asr_sha256']:
            raise ValueError('Original independent recognizer artifact changed')
        actual=decision.decision(raw,row['source_row'],scene,folder,check['clips'],json.loads(asr_path.read_text()))
        if any(row[k]!=v for k,v in actual.items()) or not actual['checks']['passed']:
            raise ValueError('Complete current teacher word/phase/quiet decision changed or failed')
        sent=[dict(role='system',content=semantic.SYSTEM),dict(role='user',content=json.dumps(dict(
            authored_train_scene=scene,actual_measured_assistant_speech=row['speech'],
            actual_measured_control_checks=row['checks'],teacher_reference_conditioning='authored-control3-TRAIN-only',
            original_audio_changed=False,all_source_and_teacher_artifacts_revalidated=True),sort_keys=True))]
        response=SEMANTIC.parent/sem_row['response_file'];value=json.loads(response.read_text())
        calls=[x for x in value['output'] if x['type']=='function_call']
        if (sem_row['input_sha256']!=semantic.input_sha(sent) or sha256(response)!=sem_row['response_sha256']
                or sem_row['passed'] is not True or value['status']!='completed' or len(calls)!=1
                or calls[0]['name']!=semantic.TOOL['name']
                or json.loads(calls[0]['arguments'])!=dict(verdict='pass',reason=sem_row['reason'])):
            raise ValueError('Independent complete semantic judgment changed')
        timeline,side,offset,user,assistant,clock=conversion.derived(raw,source_audit,batch,row['source_row'],scene,row)
        target=data/scene['id']
        if (converted['split']!='train' or converted['frames']!=clock['frames'] or converted['max_grid_offset_ms']!=offset
                or json.loads((target/'conversation.json').read_text())!=side
                or json.loads((target/'timeline.actual.json').read_text())!=timeline.to_dict()):
            raise ValueError('Measured conversion changed')
        for name,wanted in (('stem_user.wav',user),('stem_assistant.wav',assistant),('conversation.wav',np.stack([assistant,user],axis=1))):
            pcm,rate=sf.read(target/name,dtype='float32')
            if rate!=24000 or not np.array_equal(pcm,wanted):raise ValueError('Full original PCM or stereo ownership changed')
    scene=next(s for s in plan if s['id']==quality.SID)
    received=json.loads((QUALITY.parent/'server-input.turbo.json').read_text())
    if q['server_input_checks']!=quality.reflected_words(batch/quality.SID,scene,received) or not q['server_input_checks']['passed']:
        raise ValueError('Independent server-input words changed or failed')
    for fold in ('train','validation','test'):
        rows=[json.loads(x) for x in (data/(fold+'.jsonl')).read_text().splitlines()]
        if rows!=[r for r in receipt['accepted'] if r['split']==fold]:raise ValueError('Complete TRAIN-only folds changed')
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=DATA)
    p.add_argument('--inventory',type=Path,required=True);a=p.parse_args();r=verify(a.data,a.inventory)
    print(json.dumps(dict(complete40=r['complete40'],all7797_files_rehashed=True,optimizer_updates=0,training_approved=False)))


if __name__=='__main__':main()
