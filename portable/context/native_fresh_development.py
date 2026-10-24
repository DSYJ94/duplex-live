"""Collect complete portable fresh40 and independent speech40 after native review.

One GPU, serial model actors, separate ASR processes, and an actual finite budget
guard. Fresh audio is never recorded as TRAIN history. Completeness authorizes no
adoption: semantic, silence, interruption, receipt and timing review is separate.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from evals.moshi_model_download import sha256
from portable.context.native_development import ROOT,BASE,atomic,artifact_hashes,utc
from portable.context.native_control_preflight import git
from portable.context.native_speech_followup import SPEECH_FILES
from portable.context.native_gradient_followup import process_birth

ROLES=('baseline520','candidate-control40')
REASONING_ROLES=('baseline520','candidate-reasoning32')


def plan_roles(schema):
    if schema=='dl-native-control40-fresh-paired40-preflight-v1':return ROLES
    if schema=='dl-native-reasoning32-fresh-paired40-preflight-v1':return REASONING_ROLES
    raise ValueError('Explicit supported fresh cohort required')

CASE_FILES=('report.json','loading.json','model.wav','user_in.wav','packet-stream.wav',
            'events.jsonl','worker-events.json')
PARITY_FOLDERS=('parity-cold','parity-intervening','parity-reused')
OWNER_LOCK=Path('/workspace/migration/native-control40-gpu-owner.lock')


def clean(source,commit):
    if git(source,'rev-parse','HEAD')!=commit or git(source,'status','--porcelain'):
        raise ValueError('Exact clean committed portable fresh execution source required')


def budget_guard(path,pod_id,deadline,now=None):
    now=time.time() if now is None else now
    guard=json.loads(Path(path).read_text())
    age=now-datetime.fromisoformat(guard['at_utc']).timestamp()
    if (guard['phase']!='armed' or guard['pod_id']!=pod_id or not 0<=age<=90
            or datetime.fromisoformat(guard['deadline_utc']).timestamp()<deadline+120):
        raise ValueError('Fresh same-Pod live cost guard must outlast complete evaluation')
    return guard


def preflight_document(args):
    from evals.native_control40_fresh_session import cases
    path=args.out/'preflight.json'
    if sha256(path)!=args.preflight_sha256:raise ValueError('Changed portable fresh preflight')
    plan=json.loads(path.read_text())
    roles=plan_roles(plan['schema'])
    expected=(cases(Path(plan['candidate_proof']),Path(plan['baseline_proof']),roles[1])
              if roles==REASONING_ROLES else cases(Path(plan['candidate_proof']),Path(plan['baseline_proof'])))
    if (plan['complete40'] is not True
            or plan['cases']!=expected
            or Path(plan['candidate_proof']).resolve()!=args.proof.resolve()
            or plan['gpu_mode']!='serial-one-gpu' or plan['maximum_gpus']!=1
            or plan['behavior_review_sha256']!=sha256(args.review)
            or plan['source_quality_sha256']!=sha256(args.quality)
            or plan['source_profile']!='owner-approved-turbo-spelling-v1'
            or plan['evaluation_only'] is not True or plan['training_approved'] is not False
            or plan['checkpoint_adopted'] is not False):
        raise ValueError('Exact complete reviewed same-resource fresh40 preflight required')
    return plan


def command(args,row,folder):
    result=['--assets',str(ROOT/'runs/context-spanning-assets-7454205/manifest.json'),
        '--base',str(BASE),'--data',str(args.data),'--out',str(folder),
        '--native-fresh-quality',str(args.quality),'--adapter-proof',row['proof'],
        '--native-fresh-candidate-proof',str(args.proof),'--native-fresh-behavior-review',str(args.review),
        '--native-fresh-source-profile','owner-approved-turbo-spelling-v1',
        '--scenario',row['scenario'],'--expected-kind',row['kind'],'--backend-mode',row['backend'],
        '--lookup-delivery-delay-s',str(row['minimum_lookup_delay_s']),
        '--native-asr-placement','shared-gpu-evaluation',
        '--voice-reference',str(ROOT/'runs/voice-quality-7414595/codec/reference.wav'),
        '--voice-provenance',str(ROOT/'runs/voice-quality-7414595/codec/report.json')]
    if row['backend']=='native-asr':result+=['--asr-python',str(args.asr_python)]
    return result


def raw_case(folder,row,identity):
    raw=json.loads((folder/'report.json').read_text())
    loading=raw['loading'];adapter=loading['adapter']
    if (raw['scenario']!=row['scenario'] or raw['source_split']!='fresh-evaluation'
            or raw['fresh_holdout'] is not True or raw['evaluation_only'] is not True
            or raw['training_approved'] is not False or raw.get('student_history')
            or raw['quality_qualified'] is not False or raw['full20_task_qualification_passed'] is not False
            or raw['actual_lm_and_pcm_executed_together'] is not True
            or (folder/'student-history.pt').exists() or raw['scheduled_dispatch']
            or raw['word_or_acoustic_tokens_forced'] or raw['response_start_forced']
            or raw['real_user_samples_processed']!=raw['real_user_samples_expected']
            or raw['real_user_samples_expected']<=0 or raw['audio_sha256']!=sha256(folder/'model.wav')
            or adapter['checkpoint_sha256']!=identity['checkpoint_sha256']
            or Path(adapter['proof']).resolve()!=Path(row['proof']).resolve()
            or adapter['zero'] or adapter['disabled'] or not adapter['installed_before_native_graphs']
            or loading.get('input_warmup') or loading.get('native_sampling_diagnostic')
            or loading.get('ret_logit_calibration')
            or loading.get('native_runtime_comparison_only') or adapter.get('runtime_precision_diagnostic')
            or loading['native_asr_resource_placement']['profile']!='shared-gpu-evaluation'
            or loading['native_asr_resource_placement']['distinct_physical_gpus'] is not False
            or loading['native_asr_resource_placement']['separate_native_and_asr_processes'] is not True
            or loading['lookup_delivery_minimum_delay_s']!=row['minimum_lookup_delay_s']):
        raise ValueError('Fresh actual unforced source, checkpoint, audio, resource and delay identity required')
    if row['backend']=='native-asr' and (raw['asr_environment']['reference_transcript_supplied'] is not False
            or raw['asr_environment']['assigned_visible_gpu']!='0'
            or raw['distinct_lm_asr_gpu_processes'] is not False
            or raw['separate_native_and_asr_processes'] is not True):
        raise ValueError('Fresh actual arrived-audio recognition must retain its isolated shared-GPU process')
    return artifact_hashes(folder,CASE_FILES)


def compare_fresh(left,right):
    def tokens(folder):
        return [json.loads(line)['text_token'] for line in (folder/'events.jsonl').read_text().splitlines()]
    a=json.loads((left/'report.json').read_text());b=json.loads((right/'report.json').read_text())
    checks=dict(user_input_wav_sha256_exact=sha256(left/'user_in.wav')==sha256(right/'user_in.wav'),
        packet_stream_wav_sha256_exact=sha256(left/'packet-stream.wav')==sha256(right/'packet-stream.wav'),
        sampled_text_tokens_exact=tokens(left)==tokens(right),sampled_text_exact=a['text']==b['text'])
    if not all(checks.values()):raise RuntimeError('Fresh cold/reused output parity failed: '+repr(checks))
    return dict(checks,passed=True,intervening_different_conversation=True,
        actual_cuda_lm_and_pcm=True,wall_clock_playout_not_compared=True,
        native_delayed_input_tensors_recorded_or_compared=False,
        development_native_delayed_input_parity_required_by_prior_review=True,
        training_history_recorded=False,behavior_or_training_approved=False)


def controller_identity(args):
    if (args.controller_pid!=os.getppid() or process_birth(args.controller_pid)!=args.controller_birth):
        raise ValueError('Fresh actor must be a child of the exact live owning controller')
    state=json.loads((args.out/'execution.json').read_text())
    if (state['schema']!='dl-native-portable-fresh40-controller-v1' or state['pid']!=args.controller_pid
            or state['source_commit']!=args.source_commit or state['pod_id']!=args.pod_id
            or state['preflight_sha256']!=args.preflight_sha256 or state['phase']!='paired_fresh40'
            or state['active_actor_role']!=args.role or state['deadline_utc']!=args.deadline_utc
            or datetime.fromisoformat(state['cost_guard_deadline_utc']).timestamp()<args.deadline+120):
        raise ValueError('Fresh actor differs from the active same-Pod serialized controller dispatch')
    return state


def actor(args):
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    if os.environ.get('OPENAI_API_KEY'):raise RuntimeError('Provider credential must remain on Mac')
    clean(args.source,args.source_commit)
    controller_identity(args)
    plan=preflight_document(args)
    rows=[r for r in plan['cases'] if r['role']==args.role]
    if [r['question_index'] for r in rows]!=list(range(20)):
        raise ValueError('Complete ordered20 required in each actor')
    identity=plan['baseline_identity' if args.role==ROLES[0] else 'candidate_identity']
    out=args.out/args.role;out.mkdir(exist_ok=False)
    state=dict(schema='dl-native-portable-fresh20-actor-v1',pid=os.getpid(),role=args.role,
        proof=rows[0]['proof'],source_commit=args.source_commit,phase='parity',cases=[],
        gpu_mode='serial-one-gpu',assigned_main_gpu=0,assigned_asr_gpu=0,
        preflight_sha256=args.preflight_sha256,complete20=False,optimizer_updates=0,
        provider_credential_present=False,training_approved=False,behavior_approved=False)
    atomic(out/'execution.json',state)
    from evals.native_eval_session import NativeEvalSession
    from evals.context_spanning_fresh_probe import main as probe
    session=NativeEvalSession(scope='fresh-full20')
    def run(row,folder):
        if time.time()>=args.deadline:raise TimeoutError('Fresh actor deadline reached')
        clean(args.source,args.source_commit);preflight_document(args)
        probe(command(args,row,folder),session=session)
        return raw_case(folder,row,identity)
    try:
        parity={}
        for name,index in zip(PARITY_FOLDERS,(0,1,0)):
            parity[name]=run(rows[index],out/name)
        state['parity']=compare_fresh(out/'parity-cold',out/'parity-reused')
        state['parity_artifact_sha256']=parity
        state['phase']='collecting_fresh20';atomic(out/'execution.json',state)
        for row in rows:
            files=run(row,out/('case-%02d'%row['question_index']))
            state['cases'].append(dict(index=row['index'],question_index=row['question_index'],
                scenario=row['scenario'],exit_code=0,artifact_sha256=files,completed_utc=utc()))
            atomic(out/'execution.json',state)
        state.update(phase='complete',complete20=True,finished_utc=utc());atomic(out/'execution.json',state)
    except BaseException as error:
        state.update(phase='failed',error_type=type(error).__name__,error=str(error),at_utc=utc())
        atomic(out/'execution.json',state);raise


def complete_actor(root,role,plan,commit,preflight_sha256):
    folder=root/role;state=json.loads((folder/'execution.json').read_text())
    rows=[r for r in plan['cases'] if r['role']==role]
    if (state['schema']!='dl-native-portable-fresh20-actor-v1' or state['role']!=role
            or state['source_commit']!=commit or state['preflight_sha256']!=preflight_sha256
            or state['phase']!='complete' or state['complete20'] is not True
            or state['optimizer_updates']!=0 or state['provider_credential_present'] is not False
            or state['gpu_mode']!='serial-one-gpu' or state['assigned_main_gpu']!=0 or state['assigned_asr_gpu']!=0
            or [r['index'] for r in state['cases']]!=[r['index'] for r in rows]
            or len(rows)!=20 or any(r['exit_code']!=0 for r in state['cases'])):
        raise ValueError('Both complete native fresh20 actors must finish successfully')
    identity=plan['baseline_identity' if role==ROLES[0] else 'candidate_identity']
    for name,index in zip(PARITY_FOLDERS,(0,1,0)):
        if raw_case(folder/name,rows[index],identity)!=state['parity_artifact_sha256'][name]:
            raise ValueError('Changed fresh cold/intervening/reused artifact')
    if compare_fresh(folder/'parity-cold',folder/'parity-reused')!=state['parity']:
        raise ValueError('Actual fresh PCM/token/input parity must reproduce')
    for row,record in zip(rows,state['cases']):
        if raw_case(folder/('case-%02d'%row['question_index']),row,identity)!=record['artifact_sha256']:
            raise ValueError('Changed native fresh audio/report/source artifact')
    return sha256(folder/'execution.json')


def speech_report(path,folders):
    report=json.loads(path.read_text());rows=report['results']
    if report['schema']!='dl-native-speech-audit-v1' or report['device']!='cuda' or len(rows)!=20:
        raise ValueError('Complete independent CUDA speech20 required for each model')
    for row,folder in zip(rows,folders):
        if (Path(row['run']).resolve()!=folder.resolve() or row['source_kind']!='native'
                or row['wav_sha256']!=sha256(folder/'model.wav')
                or row['model_text']!=json.loads((folder/'report.json').read_text())['text']):
            raise ValueError('Independent speech audit changed or omitted actual fresh audio')
    return sha256(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','proof','review','data','quality','asr-python','cost-guard-receipt','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('source-commit','pod-id','deadline-utc'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--role',choices=(*ROLES,REASONING_ROLES[1]));parser.add_argument('--preflight-sha256')
    parser.add_argument('--controller-pid',type=int);parser.add_argument('--controller-birth',type=int)
    args=parser.parse_args()
    args.deadline=datetime.fromisoformat(args.deadline_utc.replace('Z','+00:00')).timestamp()
    clean(args.source,args.source_commit)
    if args.role:
        if not args.preflight_sha256 or not args.controller_pid or not args.controller_birth:
            parser.error('Actor needs exact parent preflight hash and actual controller PID/birth')
        actor(args);return
    if args.out.exists() or args.deadline<=time.time()+120:
        raise ValueError('Fresh versioned output and bounded future deadline required')
    if os.environ.get('OPENAI_API_KEY'):raise RuntimeError('Provider credential must remain on Mac')
    if not os.environ.get('DUPLEX_OPENAI_LOOPBACK_PROXY') or not os.environ.get('DUPLEX_OPENAI_PROXY_RUN'):
        raise RuntimeError('Actual scoped Mac-only backend transport required')
    guard=budget_guard(args.cost_guard_receipt,args.pod_id,args.deadline)
    gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader'],text=True).strip().splitlines()
    if len(gpus)!=1:raise ValueError('Fresh same-resource cohort requires exactly one physical GPU')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    OWNER_LOCK.parent.mkdir(parents=True,exist_ok=True)
    lock=OWNER_LOCK.open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
    if any(x.strip().isdigit() for x in occupied.splitlines()):raise RuntimeError('GPU occupied; never overlap fresh evaluation with training')
    args.out.mkdir(exist_ok=False)
    from evals.native_control40_fresh_session import preflight
    plan=preflight(args.proof,args.review,args.data,args.quality,args.out/'preflight.json')
    args.preflight_sha256=sha256(args.out/'preflight.json')
    roles=plan_roles(plan['schema'])
    files=('portable/context/native_fresh_development.py','evals/native_control40_fresh_session.py',
           'evals/native_control40_fresh_review.py','evals/context_spanning_fresh_probe.py',
           'evals/native_fresh_probe.py','evals/native_fresh_source_reapproval.py',
           'evals/native_fresh_inputs_v2.py','evals/native_fresh_review.py',*SPEECH_FILES)
    state=dict(schema='dl-native-portable-fresh40-controller-v1',pid=os.getpid(),pod_id=args.pod_id,
        source_commit=args.source_commit,started_utc=utc(),deadline_utc=args.deadline_utc,gpus=gpus,
        phase='paired_fresh40',children=[],preflight_sha256=args.preflight_sha256,
        behavior_review_sha256=plan['behavior_review_sha256'],source_quality_sha256=plan['source_quality_sha256'],
        implementation_sha256={name:sha256(args.source/name) for name in files},
        initial_cost_guard_sha256=sha256(args.cost_guard_receipt),cost_guard_deadline_utc=guard['deadline_utc'],
        gpu_mode='serial-one-gpu',complete_paired40=False,complete_independent_speech40=False,
        optimizer_updates=0,training_approved=False,behavior_approved=False,checkpoint_adopted=False,
        full20_task_qualification_passed=False,provider_credential_transferred=False)
    args.controller_pid=os.getpid();args.controller_birth=process_birth(args.controller_pid)
    if args.controller_birth is None:raise RuntimeError('Actual Linux controller identity required')
    atomic(args.out/'execution.json',state)
    child=None
    def stop():
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    def interrupted(signum,_):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0';env['PYTHONDONTWRITEBYTECODE']='1'
    def execute(name,cmd):
        nonlocal child
        clean(args.source,args.source_commit);preflight_document(args)
        if time.time()>=args.deadline:raise TimeoutError('Complete fresh40/speech40 deadline reached')
        with (args.out/(name+'.log')).open('x') as log:
            child=subprocess.Popen(cmd,cwd=args.source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state['children'].append(dict(name=name,pid=child.pid,command=cmd));atomic(args.out/'execution.json',state)
            code=child.wait(timeout=max(1,args.deadline-time.time()))
            state['children'][-1].update(exit_code=code,ended_utc=utc());atomic(args.out/'execution.json',state)
            if code:raise RuntimeError(name+' failed with exit '+str(code))
    try:
        actors={}
        for role in roles:
            cmd=[sys.executable,'-u','-m','portable.context.native_fresh_development']
            for key in ('source','proof','review','data','quality','asr_python','cost_guard_receipt','out',
                        'source_commit','pod_id','deadline_utc','preflight_sha256','controller_pid','controller_birth'):
                cmd+=['--'+key.replace('_','-'),str(getattr(args,key))]
            cmd+=['--role',role]
            state['active_actor_role']=role;atomic(args.out/'execution.json',state)
            execute(role,cmd)
            actors[role]=complete_actor(args.out,role,plan,args.source_commit,args.preflight_sha256)
        for i in range(20):
            if sha256(args.out/roles[0]/('case-%02d'%i)/'user_in.wav')!=sha256(args.out/roles[1]/('case-%02d'%i)/'user_in.wav'):
                raise ValueError('Every baseline/candidate fresh question must receive identical user PCM')
        state.update(phase='independent_speech40',complete_paired40=True,actor_execution_sha256=actors)
        atomic(args.out/'execution.json',state);reports={}
        for role in roles:
            folders=[args.out/role/('case-%02d'%i) for i in range(20)]
            out=args.out/'independent-speech40'/role;out.parent.mkdir(exist_ok=True)
            execute('speech-'+role,[str(args.asr_python),'-u','-m','evals.moshi_speech_audit',
                '--runs',*map(str,folders),'--device','cuda','--trim-leading-silence',
                '--trim-trailing-silence','--keep-silent-failures','--out',str(out)])
            reports[role]=speech_report(out/'report.json',folders)
        clean(args.source,args.source_commit);preflight_document(args)
        for role in roles:
            if complete_actor(args.out,role,plan,args.source_commit,args.preflight_sha256)!=actors[role]:
                raise ValueError('Actual paired fresh output changed during independent speech audit')
        state.update(phase='complete_fresh40_and_speech40_only',complete_independent_speech40=True,
            independent_speech_report_sha256=reports,finished_utc=utc(),
            semantic_silence_pause_causal_receipt_latency_and_audio_review_still_required=True)
        atomic(args.out/'execution.json',state)
    except BaseException as error:
        stop();state.update(phase='failed',error_type=type(error).__name__,error=str(error),at_utc=utc())
        atomic(args.out/'execution.json',state);raise


if __name__=='__main__':main()
