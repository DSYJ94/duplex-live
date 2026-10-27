"""Bind interrupted exact-prefix outputs to the actual completed native cache."""
import argparse,hashlib,json,subprocess
from pathlib import Path
PRODUCER='d6e3b9ffb20b1de253a3d94ec1d84a4bdcfa31cc'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main(a):
    root,prefix=Path(a.cache),Path(a.prefix);manifest=json.loads((root/'manifest.json').read_text());plan=json.loads(Path(a.execution_plan).read_text())
    assert manifest['native_speech_timing_schema']=='word-aligned-native-v1' and not manifest['proof_only'] and len(manifest['scenes'])==268
    assert manifest['source_manifest_sha256']==plan['approved_timeline_manifest_sha256']
    assert plan['original_native_job_terminal_success'] and plan['original_native_producer']==PRODUCER
    assert subprocess.check_output(['git','-C',a.original_worktree,'rev-parse','HEAD'],text=True).strip()==PRODUCER
    source=Path(a.original_worktree)/'training/venus/prefix_cache.py'
    committed=subprocess.check_output(['git','-C',a.original_worktree,'show',PRODUCER+':training/venus/prefix_cache.py'])
    assert source.read_bytes()==committed
    proofs={}
    for line in Path(a.journal).read_text().splitlines():
        try:r=json.loads(line)
        except json.JSONDecodeError:continue
        if r.get('event')=='exact_frozen_prefix' and r.get('total')==268:
            q=r['proof'];assert q['passed'] and q['logit_error']==q['hidden_error']==0;proofs[r['id']]=q
    files=[]
    for row in manifest['scenes']:
        p=prefix/row['file']
        if row['id'] in proofs and p.is_file():files.append(dict(id=row['id'],file=row['file'],sha256=sha(p),original_proof=proofs[row['id']]))
    assert 0<len(files)<268,'Interrupted full job must never be fabricated as complete'
    result=dict(producer_source=PRODUCER,original_job_terminal_success=False,original_native_job_terminal_success=True,original_failure='Independent guard stopped after live RunpodHTTP503; no newoptimizerupdates',native_cache_manifest_sha256=sha(root/'manifest.json'),original_prefix_journal_sha256=sha(a.journal),all_prior_proofs_exact_zero=True,files=files,independently_recomputed_this_process=False)
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(completed_exact_prefix_files=len(files),full_required=268,witness_sha256=sha(a.output))))

if __name__=='__main__':
    p=argparse.ArgumentParser();[p.add_argument('--'+x,required=True) for x in ('cache','prefix','journal','original-worktree','execution-plan','output')];main(p.parse_args())
