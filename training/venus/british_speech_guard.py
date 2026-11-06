"""Fail-closed admission for the proposed British/native speech comparison.

No allocation, optimizer, model import or default voice change occurs here.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path

KEY='venus-british-speech-v83'
AUTH=Path('docs/NATIVE-BRITISH-SPEECH-RECOVERY-AUTHORIZATION-2026-11-06.json')
PROPOSAL=Path('docs/NATIVE-BRITISH-SPEECH-RECOVERY-PROPOSAL-2026-11-06.json')
PLAN=Path('runs/venus-large-capacity-20261104/british-native-v83/execution.json')
BACKUP=Path('runs/venus-large-capacity-20261104/results-recovery-v83/copyback-verified.json')
VM_ID='7521667153519853425'
DISK_ID='2297341858915206463'
DEADLINE=1794052800.0
CHECKPOINTS=dict(main='a89a39b7dd1e2a08f06ee2776d92adcd41e285245d6a91fd1d36d85297a42e81',control='df75747e576c0c72bfede122a4f9e39e8f1fa26964538523a91e59f58bbcc991')
RUNTIME_FILES=('portable/venus/native_app.py','portable/venus/guard.py','portable/venus/speech_completion.py','portable/venus/speech_channel.py','portable/venus/speech_composition.py','training/venus/british_speech_guard.py','training/venus/british_behavior.py','training/venus/large_behavior.py')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def authority():
    a=json.loads(AUTH.read_text());p=json.loads(PROPOSAL.read_text())
    if a.get('status')!='APPROVED' or not a.get('human_reply_verbatim') or a.get('proposal_path')!=str(PROPOSAL) or a.get('proposal_sha256')!=sha(PROPOSAL):
        raise ValueError('Exact consolidated owner approval required')
    expected=dict(phase_ceiling_USD=175,project_ceiling_USD=350,maximum_new_GPU_intervals=1,maximum_new_GPU_seconds_including_reserve=3600,same_GPU_id=VM_ID,same_disk_id=DISK_ID,new_optimizer_updates=0)
    if any(a.get(k)!=v for k,v in expected.items()):raise ValueError('Exact British speech scope required')
    if (p['GPU']['maximum_new_running_intervals']!=1 or p['GPU']['maximum_additional_running_seconds_including_shutdown_reserve']!=3600
            or p['same_GPU_id']!=VM_ID or p['same_250GB_disk_id']!=DISK_ID or p['new_optimizer_updates']!=0
            or p['phase_ceiling_USD']!=175 or p['project_ceiling_USD']!=350):
        raise ValueError('Consolidated proposal changed')
    return a


def resource(lease):
    authority();p=json.loads(PLAN.read_text())
    if lease.get('resource_key')!=KEY or p.get('resource_key')!=KEY:raise ValueError('Separate changed-runtime lease required')
    if lease.get('authorization_sha256')!=sha(AUTH) or p.get('authorization_sha256')!=sha(AUTH) or lease.get('execution_plan_sha256')!=sha(PLAN):
        raise ValueError('Exact authorization/plan hash binding required')
    expected=dict(vm_id=VM_ID,disk_id=DISK_ID,zone='europe-west4-a',disk_size_gib=250,prior_GPU_seconds_upper=45500,actual_GPU_interval=13,maximum_native_GPU_seconds=3500,shutdown_reserve_seconds=100,storage_deadline_unix=DEADLINE,new_optimizer_updates=0)
    if any(lease.get(k)!=v or p.get(k)!=v for k,v in expected.items()):raise ValueError('Resource/time/count state cannot reset')
    if p.get('required_native_cases')!=64 or p.get('required_voice_audition_cases')!=9 or p.get('qualified') is not False:
        raise ValueError('Full native64 and separate audition9 required; no prequalification')
    if p.get('checkpoint_hashes')!=CHECKPOINTS or p.get('training_math_source')!='b39ce3ebe167b059b67fa3d131978e5da7c41ac1':
        raise ValueError('Exact saved trained checkpoints required')
    if p.get('selected_profile')!='rp-ben' or p.get('reference_sha256')!='552adfe446367b04b29cdc0c715822fee95c45e485046ae50ff0a06b7afc61d0':
        raise ValueError('Preselected RP reference required')
    if p.get('runtime_file_sha256')!={name:sha(name) for name in RUNTIME_FILES}:raise ValueError('Changed runtime bytes differ from bound source')
    if p.get('backup_receipt_sha256')!=sha(BACKUP):raise ValueError('Complete original results recovery required first')
    b=json.loads(BACKUP.read_text())
    if not (b.get('all_hashes_verified') and b.get('all_link_targets_verified') and b.get('files')==1856 and b.get('bytes')==17330768218 and b.get('links')==1 and b.get('inventory_sha256')=='e36199a1ad93b745281586c2fc839860f24cf982d7849b01060c6b0845c7d6c2'):
        raise ValueError('Full1856-file backup proof required')
    return dict(name='duplex-venus-large-gpu-20261104-eua-v67',ident=VM_ID,disk=DISK_ID,zone='europe-west4-a',disk_name='duplex-venus-large-data-20261104-eua-v67',machine='a2-ultragpu-1g',authority=AUTH,provider_read_limits=(25,20))


def validate(vm,instances,pods,lease,now):
    r=resource(lease);start=datetime.fromisoformat(vm['lastStartTimestamp']).timestamp();end=start+3500
    if vm['id']!=VM_ID or vm['name']!=r['name'] or vm['status']!='RUNNING' or vm['creationTimestamp']!=lease['creation_timestamp'] or not vm['machineType'].endswith('/a2-ultragpu-1g') or not vm['zone'].endswith('/europe-west4-a'):
        raise ValueError('Exact same live A10080 required')
    boot=[d for d in vm['disks'] if d.get('boot')]
    if len(boot)!=1 or boot[0]['autoDelete'] or int(boot[0]['diskSizeGb'])!=250 or not boot[0]['source'].endswith('/disks/'+r['disk_name']):raise ValueError('Same retained boot disk required')
    if vm.get('serviceAccounts') or set(vm['tags']['items'])!={'duplex-venus-train-bounded'}:raise ValueError('Credential/ingress scope changed')
    accel=vm.get('guestAccelerators',[])
    if len(accel)!=1 or accel[0]['acceleratorCount']!=1 or not accel[0]['acceleratorType'].endswith('/nvidia-a100-80gb'):raise ValueError('Exactly one A10080 required')
    s=vm['scheduling']
    if s['provisioningModel']!='STANDARD' or int(s['maxRunDuration']['seconds'])!=3500 or s['instanceTerminationAction']!='STOP' or s['automaticRestart']:raise ValueError('Native bounded STOP required')
    if lease['stop_unix']!=end or not start<=now<end or end+100>DEADLINE:raise ValueError('Actual start/deadline mismatch')
    if start<=datetime.fromisoformat('2026-11-06T00:10:49.641+00:00').timestamp():raise ValueError('Consumed interval cannot be relabelled')
    price=lease['verified_hourly_price_usd']
    if not 0<price<=5.9 or 45500+3500+100>73600:raise ValueError('Original time/price ceiling exceeded')
    phase=49100*5.9/3600+54+.1
    if phase>175 or phase+165.7900361111111>350:raise ValueError('Cumulative cost cannot reset')
    active=[v for v in instances if v['status'] not in ('TERMINATED','STOPPED') and (v.get('guestAccelerators') or any('/'+m in v.get('machineType','') for m in ('a2-','a3-','a4-','g2-','g4-')))]
    if len(active)!=1 or active[0]['id']!=VM_ID or any(v['status'] not in ('EXITED','STOPPED','TERMINATED') for v in pods):raise ValueError('GPU overlap')
    return end
