"""One approved298 continuation: native restore, full epochs, full native test."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

PARENT = 'e7340edd15537523db78120a97902399faa0af9caa46e875ec92e56889d6edef'
INDEX = Path('runs/venus-training-20261029/action-target-audit-v1/index-verified-v2.json')


def execute(c, env, end, lease_path):
    version=c.R.name.rsplit('-',1)[-1];unit_prefix='duplex-venus-action'+version[1:]
    out, box = c.OUT+'/action-candidate-'+version, c.OUT+'/opal-mailbox-'+version
    parent = c.OUT+'/adaptive-candidate-v34/step-0054.pt'
    index = c.OUT+'/action-target-index-'+version+'.json'
    c.sync(INDEX,index)
    assert c.ssh('sha256sum '+index).split()[0] == 'e1ded570cb385b49bcd2e9560f9e4a8cf6c6286e66754f264f7282a6fd7ef541'
    common = ' --model '+c.MODEL+' --cache '+c.OUT+'/paced-prefix-cache-v24 --lease '+lease_path+' --source '+c.source+' --opal-source '+c.ROOT+'/opal-upstream-v34'
    def stage(label, module, reserve, maximum=1800):
        assert not c.ssh('nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits').strip(), 'A prior GPU process is still live'
        unit=unit_prefix+'-'+label
        c.start_unit(unit,env+' -u -m '+module,min(maximum,max(1,int(end-time.time()-reserve))))
        c.wait_unit(unit,end-reserve)
    restore_out=c.OUT+'/action-parent-restore-'+version
    stage('parent-restore','training.venus.native_restore --kind parent --checkpoint '+parent+' --sha256 '+PARENT+' --output '+restore_out+common,5500,600)
    parent_proof=json.loads(c.ssh('cat '+restore_out+'/restore.json'))
    assert parent_proof['total_training_steps']==298 and parent_proof['new_optimizer_updates']==0
    c.record('independent_native_parent298_restored',proof=parent_proof)
    args=common+' --approval '+c.OUT+'/paced-data-approval-v22.json --output '+out+' --cache-producer 8fea2b1b1720c6744390006cef435b6a80dddfd4 --parent-checkpoint '+parent+' --parent-sha256 '+PARENT+' --opal-backend mailbox:'+box+' --opal-agent-budget .01 --maximum-epochs 2 --coverage-manifest '+c.OUT+'/paced-timelines-v22/manifest.json --action-target-index '+index+' --parent-native-restore '+restore_out+'/restore.json'
    stage('numerical-start','training.venus.train --stage start'+args,4500,1500)
    recipe=json.loads(c.ssh('cat '+out+'/recipe.json'))
    assert recipe['parent_training_steps']==298 and recipe['recipe']['updates'] in (54,108)
    assert recipe['recipe']['action_aux_weight']==.25
    c.record('actual_action_recipe_admitted',recipe=recipe['recipe'],admission=recipe['admission'])
    code="import torch,json;s=torch.load('"+out+"/step-0001.pt',map_location='cpu',weights_only=False);print(json.dumps(s['optimizer']['backend']))"
    run_id=json.loads(c.ssh(c.ROOT+'/venus-runtime-v1/venv/bin/python -c '+shlex.quote(code)))['run_id']
    unit=unit_prefix+'-restore-train'
    c.start_unit(unit,env+' -u -m training.venus.train --stage restore-continue'+args,max(1,int(end-time.time()-2500)))
    broker=None
    try:
        while True:
            current=c.props(unit)
            if current['SubState']=='exited' or current['ActiveState'] in ('failed','inactive'):break
            if c.guard.poll() is not None or time.time()>end-2500:raise TimeoutError('Observe same training unit; do not duplicate')
            if c.ssh('test -f '+box+'/'+run_id+'/latest.json && echo ready || true').strip()=='ready':
                command=[sys.executable,'-u','-m','training.venus.opal_supervisor','--source','/private/tmp/duplex-opal-upstream-20261028','--box',box,'--repo',str(Path.cwd()),'--decisions',str(c.R/'opal-decisions'),'--workdir',str(c.R/'opal-broker'),'--run-id',run_id,'--ssh-argv',str(c.R/'ssh.json'),'--remote-repo',c.remote]
                broker=subprocess.Popen(command,stdout=(c.R/'opal-broker.log').open('x'),stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
                c.record('current_agent_opal_broker_started',pid=broker.pid,run_id=run_id,external_api_calls=0)
                break
            time.sleep(3)
        c.wait_unit(unit,end-2500)
        terminal=json.loads(c.ssh('cat '+out+'/training-terminal.json'))
        assert terminal['completed'] and terminal['actual_updates'] in (54,108) and terminal['maximum_authorized_updates']==108
        assert terminal['total_optimizer_updates']==298+terminal['actual_updates']<=406
        c.record('actual_training_complete',result=terminal)
    finally:
        if broker and broker.poll() is None:
            try:broker.wait(timeout=15)
            except subprocess.TimeoutExpired:broker.terminate()
    # Preserve the complete new candidate immediately, before further model work.
    backup=c.R/'critical-candidate';backup.mkdir()
    rows=json.loads(c.ssh('/usr/bin/python3 -c '+shlex.quote("import pathlib,hashlib,json;r=pathlib.Path('"+out+"');print(json.dumps([dict(path=str(p.relative_to(r)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in sorted(r.rglob('*')) if p.is_file() and not p.is_symlink()]))"),60))
    c.run(['rsync','-ar','--no-owner','--no-group','--exclude=session-token','--exclude=.private','--exclude=credentials','--exclude=owner-conversations','--exclude=keys','--exclude=API keys.md','--exclude=core','--exclude=core.*','-e',shlex.join(c.ssh_argv[:-1]),c.ssh_argv[-1]+':'+out+'/',str(backup)+'/'],180)
    for row in rows:
        path=backup/row['path']
        assert path.stat().st_size==row['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
    (c.R/'critical-candidate-verified.json').write_text(json.dumps(dict(files=rows,all_hashes_verified=True,source=c.source),indent=2)+'\n')
    c.record('critical_candidate_hash_backup_pass',files=len(rows),bytes=sum(r['bytes'] for r in rows))
    final=out+'/step-%04d.pt'%terminal['actual_updates'];final_sha=c.ssh('sha256sum '+final).split()[0]
    final_out=c.OUT+'/action-final-restore-'+version
    stage('final-restore','training.venus.native_restore --kind final --checkpoint '+final+' --sha256 '+final_sha+' --output '+final_out+common,2200,600)
    c.ssh('cp '+final_out+'/restore.json '+out+'/final-native-restore.json')
    c.record('independent_final_native_restore_pass',proof=json.loads(c.ssh('cat '+out+'/final-native-restore.json')))
    paired='training.venus.paced_behavior --model '+c.MODEL+' --candidate '+out+' --parent-checkpoint '+parent+' --parent-sha256 '+PARENT+' --plan '+c.remote+'/evals/fixtures/venus-control-repair-source-v2-20261027.json --timelines '+c.OUT+'/paced-timelines-v22 --approval '+c.OUT+'/paced-data-approval-v22.json --word-quality '+c.OUT+'/paced-word-quality-v22.json --old-plan '+c.remote+'/evals/fixtures/venus-control-corpus-v4-20261026.json --old-timelines '+c.OUT+'/timelines-v7-repaired --old-approval '+c.OUT+'/data-approval-v7.json --output '+c.OUT+'/action-native-paired-'+version+' --source '+c.source+' --lease '+lease_path+' --tts-attention released --voice-reference '+c.OUT+'/george-reference-v33/reference-16k.wav --voice-reference-sha256 40159a59a35053c358d5192b3919a86453c8c6bf1b2019b289119ceca444f674 --reuse-parent-collection '+c.OUT+'/adaptive-native-paired-v37'
    stage('paired-native',paired,500,2100)
    report=json.loads(c.ssh('cat '+c.OUT+'/action-native-paired-'+version+'/paired-native-report.json'))
    assert report['complete'] and len(report['cases'])==64 and report['original_completed_cases_reused']==32
    c.record('full64_paired_native_and_asr_complete',reused_baselines=32,new_native_cases=32,adopted=False)
