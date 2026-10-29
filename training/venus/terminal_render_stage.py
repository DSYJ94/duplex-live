"""One1200s native rendering diagnostic, zerooptimizerupdates, no adoption."""
import json,time

def execute(c,env,end,lease_path):
    version=c.R.name.rsplit('-',1)[-1];out=c.OUT+'/terminal-speech-probe-'+version
    args='training.venus.terminal_speech_probe --model '+c.MODEL+' --checkpoint '+c.OUT+'/paced-candidate-v25/step-0054.pt --selection '+c.remote+'/evals/fixtures/venus-terminal-render-probe-20261029.json --plan '+c.remote+'/evals/fixtures/venus-control-repair-source-v2-20261027.json --timelines '+c.OUT+'/paced-timelines-v22 --approval '+c.OUT+'/paced-data-approval-v22.json --voice-reference '+c.OUT+'/george-reference-v33/reference-16k.wav --voice-reference-sha256 40159a59a35053c358d5192b3919a86453c8c6bf1b2019b289119ceca444f674 --lease '+lease_path+' --source '+c.source+' --output '+out
    unit='duplex-venus-terminal-render-'+version
    c.start_unit(unit,env+' -u -m '+args,max(1,int(end-time.time()-300)));c.wait_unit(unit,end-300)
    result=json.loads(c.ssh('cat '+out+'/probe.json'));assert result['complete'] and len(result['rows'])==4 and result['optimizer_updates']==0
    c.record('complete_native_terminal_render_diagnostic',pairs=result['pairs'],optimizer_updates=0,adopted=False)
