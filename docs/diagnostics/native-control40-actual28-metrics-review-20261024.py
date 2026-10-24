import json,math,hashlib,datetime,subprocess
from pathlib import Path
p=Path('runs/local-migration-20261023/native-control40-continuous-copyback-v17/20261024T065903.224466Z/native-control40-precise-pilot-v11-environment-retry/training')
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
assert sha(p/'report.json')=='500781444712ebb6368d534406ee25f663df4b511c6f920fae9e985db9aa5a18'
assert sha(p/'data.json')=='de92099057ded15a1af0e387c96b457cdf4c1731c26594dcdca211d4c442fc1e'
r=json.loads((p/'report.json').read_text());d=json.loads((p/'data.json').read_text())
commit=r['repo_commit'];name='evals/context_actual_train.py'
assert Path(name).read_bytes()==subprocess.check_output(['git','show',commit+':'+name])
assert len(r['initial_stage_metrics'])==len(r['final_metrics'])==len(d['components'])==28
rows=[]
for i,(before,after,data,member)in enumerate(zip(r['initial_stage_metrics'],r['final_metrics'],d['components'],r['assistant_audio_auxiliary']['complete_batch_membership']['actual'])):
 assert before['scenario']==after['scenario']==data['scenario']
 assert member['id']==str(i)+':'+data['component']
 n=data.get('text_loss_targets',data['text_targets']);assert type(n)is int and n>0
 assert n==data['text_targets']
 b=round(before['top1']*n);a=round(after['top1']*n)
 assert math.isclose(b,before['top1']*n,abs_tol=1e-4) and math.isclose(a,after['top1']*n,abs_tol=1e-4)
 rows.append(dict(index=i,scenario=data['scenario'],kind=member['kind'],supervised_text_targets=n,parent_correct_tokens=b,after8_correct_tokens=a,correct_tokens_delta=a-b,parent_nll=before['nll'],after8_nll=after['nll'],nll_delta=after['nll']-before['nll'],parent_silence_top1=before['silence_top1'],after8_silence_top1=after['silence_top1']))
total=sum(x['supervised_text_targets']for x in rows);b=sum(x['parent_correct_tokens']for x in rows);a=sum(x['after8_correct_tokens']for x in rows)
assert total==782 and b==604 and a==601
out=dict(schema='dl-native-control40-actual28-teacher-forced-metrics-review-v1',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_commit=commit,source_metrics_implementation_sha256=hashlib.sha256(subprocess.check_output(['git','show',commit+':'+name])).hexdigest(),report_sha256=sha(p/'report.json'),data_sha256=sha(p/'data.json'),complete28_reviewed=True,all_supervised_text_targets=total,parent_correct_tokens=b,after8_correct_tokens=a,parent_micro_token_accuracy=b/total,after8_micro_token_accuracy=a/total,parent_macro_component_accuracy=sum(x['top1']for x in r['initial_stage_metrics'])/28,after8_macro_component_accuracy=sum(x['top1']for x in r['final_metrics'])/28,accuracy_changed_components=[x['index']for x in rows if x['correct_tokens_delta']],all28_components=rows,interpretation='Three fewer correct teacher-forced tokens across all782: two in ordinary component2, one in three-token cancellation component27. The macro component mean gives the three-token component the same weight as each longer component. Neither aggregation is free-running voice/task accuracy or the full weighted training loss. No change to training/gates.',model_loaded=False,GPU_work_executed=False,behavior_qualified=False,model_adopted=False)
p=Path('runs/local-migration-20261023/native-control40-v11-actual28-metrics-review-v1.json');assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k!='all28_components'}))
