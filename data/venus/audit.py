"""Source, split and causal checks. No passing subset can issue approval."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(plan_path,root,audio_root):
    plan=json.loads(Path(plan_path).read_text());root=Path(root);audio_root=Path(audio_root)
    manifest=json.loads((root/'manifest.json').read_text());ar=json.loads((audio_root/'render-receipts.json').read_text())
    assert sha(plan_path)==manifest['source_sha256']
    assert ar['complete'] and manifest['audio_receipts_sha256']==sha(audio_root/'render-receipts.json')
    assert {r['text'] for r in ar['utterances']}=={t for s in plan['scenes'] for t in s['user_turns']}
    assert len({s['id'] for s in plan['scenes']})==224
    assert Counter(s['fold'] for s in plan['scenes'])==Counter(TRAIN=152,DEV=36,HOLDOUT=36)
    assert len(manifest['scenes'])==224
    records={r['id']:r for r in manifest['scenes']};groups=defaultdict(list);folds=defaultdict(set)
    for row in ar['utterances']:
        assert row['seconds']>0 and row['peak']<=.995 and sha(audio_root/(row['id']+'.wav'))==row['sha256']
    for scene in plan['scenes']:
        row=records[scene['id']];folder=root/scene['id'];timeline=json.loads((folder/'timeline.json').read_text())
        assert sha(folder/'timeline.json')==row['timeline_sha256'] and sha(folder/'user.wav')==row['user_wav_sha256']
        assert not timeline['training_approved'] and not timeline['serving_oracle_dispatch']
        assert [x['text'] for x in timeline['user_turns']]==scene['user_turns']
        p=scene['teacher_program'];family=scene['family'];fold=scene['fold']
        if family=='quiet':folds['cue:'+p['cue']].add(fold)
        if family=='pause':folds['steps:'+p['first_text']].add(fold)
        if scene['synthetic_private_world']:
            key=next(iter(scene['synthetic_private_world']));folds['lookup:'+key].add(fold)
            receipt=json.loads((folder/'real-worker.json').read_text())
            assert receipt['scene_id']==scene['id'] and receipt['teacher_forced_request'] and not receipt['native_student_dispatch_claimed']
            assert receipt['world']==scene['synthetic_private_world']
            events=receipt['events'];completion=[e for e in events if e['type']=='worker_completed']
            if family=='cancel':
                assert not completion and any(e['type']=='cancellation_confirmed' and e['targets'] for e in events)
                value=next(iter(scene['synthetic_private_world'].values()))
                assert value not in json.dumps(timeline['actions'])
            else:
                assert len(completion)==1 and completion[0]['result'].endswith(next(iter(scene['synthetic_private_world'].values())))
                result_actions=[x for x in timeline['actions'] if x['text']==p['result_answer']]
                assert len(result_actions)==1 and result_actions[0]['frame']>=timeline['backend_feedback'][0]['frame']
            available=timeline['backend_feedback'][0]['frame']
            prefix=[x for x in timeline['actions'] if x['frame']<available]
            groups[(fold,family,key)].append((timeline['user_wav_sha256'],json.dumps(prefix,sort_keys=True),p['worker_delay_seconds']))
        if family=='ordinary':assert all(x['kind']!='delegate' for x in timeline['actions'])
        if family=='quiet':assert min(x['frame'] for x in timeline['actions'])>=timeline['user_turns'][-1]['start_seconds']+timeline['user_turns'][-1]['seconds']
    assert all(len(x)==1 for x in folds.values()),'Semantic split groups overlap'
    assert all(len(rows)==4 and len(set(rows))==1 for rows in groups.values()),'Private worlds must have matched input and pre-result actions/delay'
    return dict(passed=True,scenes=224,unique_audio=len(ar['utterances']),counts=plan['counts'],
        matched_counterfactual_groups=len(groups),manifest_sha256=sha(root/'manifest.json'),
        plan_sha256=sha(plan_path),audio_receipts_sha256=sha(audio_root/'render-receipts.json'),
        independent_word_approval=False,independent_semantic_approval=False,training_approved=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--timelines',required=True);p.add_argument('--audio-root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    r=audit(a.plan,a.timelines,a.audio_root);Path(a.output).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
