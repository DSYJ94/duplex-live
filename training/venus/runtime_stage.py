"""Complete original32 comparison only; no training stage is reachable."""
import json
import time


def execute(c, env, end, lease_path):
    assert not c.ssh('nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits').strip()
    args = dict(model=c.MODEL, manifest=c.remote+'/docs/NATIVE-VENUS-FOUNDATION32-INPUTS-2026-10-30.json',
        reference_collection=c.OUT+'/action-native-paired-v42',
        plan=c.remote+'/evals/fixtures/venus-control-repair-source-v2-20261027.json',
        timelines=c.OUT+'/paced-timelines-v22', approval=c.OUT+'/paced-data-approval-v22.json',
        old_plan=c.remote+'/evals/fixtures/venus-control-corpus-v4-20261026.json',
        old_timelines=c.OUT+'/timelines-v7-repaired', old_approval=c.OUT+'/data-approval-v7.json',
        word_quality=c.OUT+'/paced-word-quality-v22.json', output=c.OUT+'/runtime-native-v44',
        lease=lease_path, source=c.source)
    command = env+' -u -m training.venus.foundation_compare'+''.join(' --'+k.replace('_','-')+' '+v for k,v in args.items())
    assert end-time.time()>1800, 'Complete cohort and backup reserve required'
    command += ' --private-action-repair --candidate-checkpoint '+c.OUT+'/action-candidate-v42/step-0108.pt'
    unit='duplex-venus-runtime44-native'
    c.start_unit(unit,command,int(end-time.time()-400))
    c.wait_unit(unit,end-400)
    report=json.loads(c.ssh('cat '+args['output']+'/foundation-report.json'))
    assert report['complete'] and len(report['cases'])==32 and report['optimizer_updates']==0
    c.record('full32_untouched_native_and_asr_complete',new_native_cases=32,reference_cases=32,adopted=False)
