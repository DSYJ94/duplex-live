"""One admitted54-update phase from244, current-agent OPAL, full paired native."""
import json,subprocess,sys,time
from pathlib import Path


def execute(c,env,end,lease_path,stage):
    assert stage['maximum_candidate_updates']==54 and stage['runtime_backend'] in ('released','sdpa')
    out=c.OUT+'/adaptive-candidate-v34';box=c.OUT+'/opal-mailbox-v34'
    args=' --model '+c.MODEL+' --cache '+c.OUT+'/paced-prefix-cache-v24 --approval '+c.OUT+'/paced-data-approval-v22.json --output '+out+' --source '+c.source+' --cache-producer 8fea2b1b1720c6744390006cef435b6a80dddfd4 --lease '+lease_path+' --parent-checkpoint '+c.OUT+'/paced-candidate-v25/step-0054.pt --parent-sha256 c634e6cf38831210be0165ce7f6dd09c5454226f62ff622477fd4d051abde270 --opal-source '+c.ROOT+'/opal-upstream-v34 --opal-backend mailbox:'+box+' --opal-agent-budget .01 --maximum-epochs 1 --coverage-manifest '+c.OUT+'/paced-timelines-v22/manifest.json'
    def execute(label,module,reserve):
        unit='duplex-venus-adaptive34-'+label;c.start_unit(unit,env+' -u -m '+module,max(1,int(end-time.time()-reserve)));c.wait_unit(unit,end-reserve)
    execute('numerical-start','training.venus.train --stage start'+args,6000)
    recipe=json.loads(c.ssh('cat '+out+'/recipe.json'));assert recipe['recipe']['updates']==54 and recipe['parent_training_steps']==244
    c.record('actual_adaptive_recipe_admitted',recipe=recipe['recipe'],admission=recipe['admission'])
    import shlex
    code="import torch,json; s=torch.load('"+out+"/step-0001.pt',map_location='cpu',weights_only=False); print(json.dumps(s['optimizer']['backend']))"
    backend=json.loads(c.ssh(c.ROOT+'/venus-runtime-v1/venv/bin/python -c '+shlex.quote(code)));run_id=backend['run_id']
    unit='duplex-venus-adaptive34-restore-train';c.start_unit(unit,env+' -u -m training.venus.train --stage restore-continue'+args,max(1,int(end-time.time()-3800)))
    broker=None
    try:
        until=end-3800
        while True:
            current=c.props(unit)
            if current['SubState']=='exited' or current['ActiveState'] in ('failed','inactive'):break
            if c.guard.poll() is not None or time.time()>until:raise TimeoutError('Observe actual training stage; no duplicate')
            if c.ssh('test -f '+box+'/'+run_id+'/latest.json && echo ready || true').strip()=='ready':
                command=[sys.executable,'-u','-m','training.venus.opal_supervisor','--source','/private/tmp/duplex-opal-upstream-20261028','--box',box,'--repo',str(Path.cwd()),'--decisions',str(c.R/'opal-decisions'),'--workdir',str(c.R/'opal-broker'),'--run-id',run_id,'--ssh-argv',str(c.R/'ssh.json'),'--remote-repo',c.remote]
                broker=subprocess.Popen(command,stdout=(c.R/'opal-broker.log').open('x'),stderr=subprocess.STDOUT,env={**__import__('os').environ,'PYTHONDONTWRITEBYTECODE':'1'})
                c.record('current_agent_opal_broker_started',pid=broker.pid,run_id=run_id,external_api_calls=0);break
            time.sleep(3)
        c.wait_unit(unit,end-3800)
        terminal=json.loads(c.ssh('cat '+out+'/training-terminal.json'));assert terminal['completed'] and terminal['actual_updates']==54 and terminal['total_optimizer_updates']==298
        c.record('actual_training_complete',result=terminal)
    finally:
        if broker and broker.poll() is None:
            try:broker.wait(timeout=15)
            except subprocess.TimeoutExpired:broker.terminate()
    paired='training.venus.paced_behavior --model '+c.MODEL+' --candidate '+out+' --parent-checkpoint '+c.OUT+'/paced-candidate-v25/step-0054.pt --parent-sha256 c634e6cf38831210be0165ce7f6dd09c5454226f62ff622477fd4d051abde270 --plan '+c.remote+'/evals/fixtures/venus-control-repair-source-v2-20261027.json --timelines '+c.OUT+'/paced-timelines-v22 --approval '+c.OUT+'/paced-data-approval-v22.json --word-quality '+c.OUT+'/paced-word-quality-v22.json --old-plan '+c.remote+'/evals/fixtures/venus-control-corpus-v4-20261026.json --old-timelines '+c.OUT+'/timelines-v7-repaired --old-approval '+c.OUT+'/data-approval-v7.json --output '+c.OUT+'/adaptive-native-paired-v34 --source '+c.source+' --lease '+lease_path+' --tts-attention '+stage['runtime_backend']+' --voice-reference '+c.OUT+'/george-reference-v33/reference-16k.wav --voice-reference-sha256 40159a59a35053c358d5192b3919a86453c8c6bf1b2019b289119ceca444f674'
    execute('paired-native',paired,600);c.record('complete64_paired_native_collected',adopted=False)
