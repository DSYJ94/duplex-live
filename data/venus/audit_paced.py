"""Complete, fixed292-scene audit of the new paced repair corpus."""
import argparse,hashlib,json,math,wave
from pathlib import Path
from collections import Counter,defaultdict
from .pacing import words

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def audit(plan_path,root,audio_root,assistant_quality,tokenizer):
    plan_path,root,audio_root=Path(plan_path),Path(root),Path(audio_root)
    plan=json.loads(plan_path.read_text());m=json.loads((root/'manifest.json').read_text());ar=json.loads((audio_root/'render-receipts.json').read_text())
    aq=json.loads(Path(assistant_quality).read_text());assert aq['all_passed'] and aq['requested']==aq['passed']==83
    assert plan['schema']=='dl-venus-focused-control-repair-v2' and plan['counts']==dict(TRAIN=216,DEV=52,HOLDOUT=24)
    assert len(plan['scenes'])==len(m['scenes'])==292 and len({s['id'] for s in plan['scenes']})==292
    assert m['source_sha256']==sha(plan_path) and m['audio_receipts_sha256']==sha(audio_root/'render-receipts.json') and m['assistant_word_quality_sha256']==sha(assistant_quality)
    assert ar['complete'] and len(ar['utterances'])==218 and {r['text'] for r in ar['utterances']}=={t for s in plan['scenes'] for t in s['user_turns']}
    for r in ar['utterances']:assert r['seconds']>0 and r['peak']<=.995 and sha(audio_root/r['wav'])==r['sha256']
    records={r['id']:r for r in m['scenes']};assert set(records)=={s['id'] for s in plan['scenes']}
    folds=defaultdict(set);groups=defaultdict(list);assistant_hashes={r['source_sha256'] for r in aq['rows']};total_units=0
    for s in plan['scenes']:
        row=records[s['id']];folder=root/s['id'];t=json.loads((folder/'timeline.json').read_text());p=s['teacher_program']
        assert sha(folder/'timeline.json')==row['timeline_sha256'] and sha(folder/'user.wav')==row['user_wav_sha256']
        assert t['native_speech_timing_schema']=='word-aligned-native-v1' and t['all_assistant_words_preserved'] and not t['training_approved'] and not t['serving_oracle_dispatch']
        assert [u['text'] for u in t['user_turns']]==s['user_turns']
        with wave.open(str(folder/'user.wav')) as w:assert (w.getframerate(),w.getnchannels(),w.getsampwidth(),w.getnframes())==(16000,1,2,t['frames']*16000)
        occupied=set()
        for action in t['actions']:
            assert 0<=action['frame']<t['frames']
            if action['kind']=='speak':
                units=action['native_speech_units'];assert action['timing_source_sha256'] in assistant_hashes
                assert [u['offset'] for u in units]==list(range(len(units))) and [u['final'] for u in units]==[False]*(len(units)-1)+[True]
                assert words(' '.join(u['text'] for u in units))==words(action['text'])
                for u in units:
                    assert len(tokenizer.encode(u['text'],add_special_tokens=False))<=16
                    frame=action['frame']+u['offset'];assert frame not in occupied and frame<t['frames'];occupied.add(frame)
                total_units+=len(units)
            else:
                assert action['frame'] not in occupied;occupied.add(action['frame'])
        spoken=[x for x in t['actions'] if x['kind']=='speak']
        if s['family']=='ordinary':assert len(spoken)==1 and words(spoken[0]['text'])==words(p['answer']) and all(x['kind']!='delegate' for x in t['actions'])
        if s['family']=='quiet':
            folds['cue:'+p['cue']].add(s['fold']);release=t['user_turns'][-1]
            assert min(x['frame'] for x in t['actions'])>=release['start_seconds']+release['seconds']
            assert len(spoken)==1 and words(spoken[0]['text'])==words(p['release_answer'])
        if s['family']=='pause':
            folds['steps:'+p['first_text']].add(s['fold']);assert len(spoken)==2
            assert words(' '.join(x['text'] for x in spoken))==words(p['first_text']+' '+p['remaining_text'])
            stop=next(x for x in t['actions'] if x['kind']=='end_turn');assert spoken[0]['frame']+len(spoken[0]['native_speech_units'])<=stop['frame']
            release=t['user_turns'][-1];assert spoken[1]['frame']>=release['start_seconds']+release['seconds']
        if s['synthetic_private_world']:
            key=next(iter(s['synthetic_private_world']));folds['lookup:'+key].add(s['fold'])
            receipt=json.loads((folder/'real-worker.json').read_text());assert receipt['scene_id']==s['id'] and receipt['world']==s['synthetic_private_world'] and receipt['teacher_forced_request'] and not receipt['native_student_dispatch_claimed']
            available=t['backend_feedback'][0]['frame'];assert available>=t['actions'][0]['frame']
            if s['family']=='cancel':
                assert any(e['type']=='worker_started' for e in receipt['events']) and any(e['type']=='worker_cancelled' for e in receipt['events'])
                assert any(e['type']=='cancellation_confirmed' and e['targets'] for e in receipt['events']) and not any(e['type']=='worker_completed' for e in receipt['events'])
                assert next(iter(s['synthetic_private_world'].values())) not in json.dumps(t['actions'])
                assert all(x['frame']>=available for x in spoken)
            else:
                done=[e for e in receipt['events'] if e['type']=='worker_completed'];assert len(done)==1 and done[0]['result'].endswith(s['synthetic_private_world'][key])
                result=[x for x in spoken if x['text']==p['result_answer']];assert len(result)==1 and result[0]['frame']>=available
                if s['family']=='pending':assert any(x['text']==p['followup_answer'] and x['frame']<available for x in spoken)
            before=json.dumps([x for x in t['actions'] if x['frame']<available],sort_keys=True)
            groups[(s['fold'],s['family'],key)].append((row['user_wav_sha256'],before,p['worker_delay_seconds'],s['synthetic_private_world'][key]))
    assert all(len(v)==1 for v in folds.values()),'Cue/key/step semantic folds must remain disjoint'
    for variants in groups.values():assert len(variants)>=2 and len({x[:3] for x in variants})==1 and len({x[3] for x in variants})==len(variants),'Every private-world variant must have identical input/prefix/delay and a distinct hidden value'
    return dict(passed=True,scenes=292,unique_audio=218,counts=plan['counts'],matched_counterfactual_groups=len(groups),native_speech_units=total_units,manifest_sha256=sha(root/'manifest.json'),plan_sha256=sha(plan_path),audio_receipts_sha256=sha(audio_root/'render-receipts.json'),assistant_word_quality_sha256=sha(assistant_quality),independent_semantic_approval=False,training_approved=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();[p.add_argument('--'+x,required=True) for x in ('plan','timelines','audio-root','assistant-quality','tokenizer','output')];a=p.parse_args()
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(a.tokenizer,trust_remote_code=True,local_files_only=True)
    result=audit(a.plan,a.timelines,a.audio_root,a.assistant_quality,tokenizer);Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
