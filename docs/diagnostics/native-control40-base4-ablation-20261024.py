"""Complete matched ordinary4 with adapters disabled; diagnosis, never approval.

Runs unchanged C9 probe in four fresh processes with all original source PCM,
voice, persona, native protocol, seed, sampler and runtime. The existing paired20
provides the native520/control40 comparators. Only adapter presence changes.
No source/target edits, optimization, provider calls or sealed fresh evaluation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

SHA='c9efaaac24d983342f7eded46617e6450bc102ce'


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(4194304),b''):
            h.update(chunk)
    return h.hexdigest()


def run(a):
    assert subprocess.check_output(['git','-C',str(a.source),'rev-parse','HEAD'],text=True).strip()==SHA
    assert not subprocess.check_output(['git','-C',str(a.source),'status','--porcelain'],text=True)
    assert not os.environ.get('OPENAI_API_KEY')
    import sys
    sys.path.insert(0,str(a.source))
    from portable.context.native_development import command,ROOT,PLAN,INFERENCE_FILES,atomic,CASE_FILES
    plan=json.loads((a.source/PLAN).read_text())
    pair=json.loads((a.paired/'execution.json').read_text())
    assert pair['complete_paired20'] and pair['source_commit']==SHA
    assert len(plan['cases'])==10 and all(c['kind']=='simple' and c['backend']=='observe-only' for c in plan['cases'][:4])
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    a.out.mkdir(exist_ok=False)
    pins={n:digest(a.source/n)for n in INFERENCE_FILES}
    assert pins==pair['execution_implementation_sha256']
    deadline=datetime.fromisoformat(a.deadline_utc.replace('Z','+00:00')).timestamp()
    state=dict(schema='dl-native-control40-matched-published-base-ordinary4-diagnostic-v1',
        pid=os.getpid(),birth=int(Path('/proc/self/stat').read_text().split()[21]),
        started_utc=datetime.now(timezone.utc).isoformat(),source_commit=SHA,
        inference_implementation_sha256=pins,paired_execution_sha256=digest(a.paired/'execution.json'),
        cases=[],phase='running_complete4',optimizer_updates=0,provider_requests=0,
        adapter_disabled_only_intervention=True,fresh_process_per_case=True,
        complete4=False,behavior_qualified=False,training_approved=False,model_adopted=False)
    def save():
        state['at_utc']=datetime.now(timezone.utc).isoformat();atomic(a.out/'execution.json',state)
    save()
    try:
        for i,case in enumerate(plan['cases'][:4]):
            folder=a.out/('case-%02d'%i)
            args=[str(a.python),'-u','-m','evals.context_spanning_probe',
                *command(case,ROOT/'runs/context-training-proof-7510331',folder,a.python,'container-default','serial-one-gpu'),
                '--adapter-disabled']
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1')
            env.pop('OPENAI_API_KEY',None)
            with (a.out/('case-%02d.log'%i)).open('x') as log:
                child=subprocess.Popen(args,cwd=a.source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state['child']=dict(pid=child.pid,birth=int(Path('/proc',str(child.pid),'stat').read_text().split()[21]),case=i)
                save()
                try:
                    code=child.wait(timeout=max(1,deadline-time.time()))
                finally:
                    if child.poll() is None:
                        import signal
                        os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=20)
            if code:
                raise RuntimeError('Original native probe failed: '+str(code))
            r=json.loads((folder/'report.json').read_text())
            assert r['loading']['adapter']['disabled'] is True
            assert r['scheduled_dispatch'] is False and r['word_or_acoustic_tokens_forced'] is False and r['response_start_forced'] is False
            assert r['real_user_samples_processed']==r['real_user_samples_expected']
            hashes={n:digest(folder/n)for n in CASE_FILES}
            assert digest(folder/'user_in.wav')==digest(a.paired/'baseline520'/('case-%02d'%i)/'user_in.wav')==digest(a.paired/'candidate-control40'/('case-%02d'%i)/'user_in.wav')
            state['cases'].append(dict(index=i,command=args,actual_exit=code,artifact_sha256=hashes,text=r['text']))
            save()
        state['phase']='independent_speech4';save()
        args=[str(a.python),'-u','-m','evals.moshi_speech_audit','--runs',
            *[str(a.out/('case-%02d'%i))for i in range(4)],'--device','cuda',
            '--trim-leading-silence','--trim-trailing-silence','--keep-silent-failures','--out',str(a.out/'independent-speech4')]
        with (a.out/'independent-speech4.log').open('x') as log:
            child=subprocess.Popen(args,cwd=a.source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state['child']=dict(pid=child.pid,birth=int(Path('/proc',str(child.pid),'stat').read_text().split()[21]),stage='speech4');save()
            try:
                code=child.wait(timeout=max(1,deadline-time.time()))
            finally:
                if child.poll() is None:
                    import signal
                    os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=20)
        if code:
            raise RuntimeError('Independent speech4 failed: '+str(code))
        rows=json.loads((a.out/'independent-speech4/report.json').read_text())['results']
        assert len(rows)==4
        for i,row in enumerate(rows):
            assert Path(row['run'])==a.out/('case-%02d'%i)
            assert row['wav_sha256']==state['cases'][i]['artifact_sha256']['model.wav']
        assert pins=={n:digest(a.source/n)for n in INFERENCE_FILES}
        assert state['paired_execution_sha256']==digest(a.paired/'execution.json')
        state.update(phase='complete4_diagnostic_review_required',complete4=True,
            speech_report_sha256=digest(a.out/'independent-speech4/report.json'))
    except BaseException as error:
        state.update(phase='failed_preserved_no_retry',error_type=type(error).__name__,reason=str(error));raise
    finally:
        save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','paired','python','out'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--deadline-utc',required=True)
    run(p.parse_args())
