"""ZERO-update completion of interrupted full native evaluation; no case repeats."""
import json,time,shutil
from .adaptive_guard import execution_plan

def execute(c,env,end,lease_path):
    plan=json.loads(execution_plan(json.loads((c.R/'lease.json').read_text())).read_text());assert plan['new_optimizer_updates_authorized_by_this_execution']==0
    assert c.ssh('sha256sum '+c.OUT+'/adaptive-candidate-v34/step-0054.pt').split()[0]==plan['checkpoint_sha256']
    # Critical checkpoint backup already verified on Mac. The derived audit
    # clones originals without overwriting them; full hash copyback follows it.
    c.ssh('mkdir -p '+c.OUT+'/journals; sudo journalctl -u duplex-venus-adaptive34-paired-native --no-pager -o cat > '+c.OUT+'/journals/duplex-venus-adaptive34-paired-native.log')
    out=c.OUT+'/adaptive-candidate-v34'
    paired='training.venus.paced_behavior --model '+c.MODEL+' --candidate '+out+' --parent-checkpoint '+c.OUT+'/paced-candidate-v25/step-0054.pt --parent-sha256 c634e6cf38831210be0165ce7f6dd09c5454226f62ff622477fd4d051abde270 --plan '+c.remote+'/evals/fixtures/venus-control-repair-source-v2-20261027.json --timelines '+c.OUT+'/paced-timelines-v22 --approval '+c.OUT+'/paced-data-approval-v22.json --word-quality '+c.OUT+'/paced-word-quality-v22.json --old-plan '+c.remote+'/evals/fixtures/venus-control-corpus-v4-20261026.json --old-timelines '+c.OUT+'/timelines-v7-repaired --old-approval '+c.OUT+'/data-approval-v7.json --output '+c.OUT+'/adaptive-native-paired-v37 --source '+c.source+' --lease '+lease_path+' --tts-attention '+'released'+' --voice-reference '+c.OUT+'/george-reference-v33/reference-16k.wav --voice-reference-sha256 40159a59a35053c358d5192b3919a86453c8c6bf1b2019b289119ceca444f674'
    paired+=' --resume-collection '+c.OUT+'/adaptive-native-paired-v34 --resume-from-source dd767d65ac663e701142f2c882aeb5e1c2a2c6e5'
    unit='duplex-venus-adaptive37-paired-native'
    c.start_unit(unit,env+' -u -m '+paired,max(1,int(end-time.time()-600)));c.wait_unit(unit,end-600)
    report=json.loads(c.ssh('cat '+c.OUT+'/adaptive-native-paired-v37/paired-native-report.json'))
    assert report['complete'] and len(report['cases'])==64 and report['new_optimizer_updates']==0
    c.record('complete64_resumed_native_and_asr',reused=report['original_completed_cases_reused'],adopted=False,optimizer_updates=0)
