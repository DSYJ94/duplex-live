import os,json,hashlib,datetime,math
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
parent=Path('/projects/u6si/duplex-live/runs/context-training-proof-7510331')
original=parent/'training-state.pt';report=json.loads((parent/'report.json').read_text());oldsha=hashlib.sha256(original.read_bytes()).hexdigest();assert oldsha==report['training_state_sha256']
new=Path('/tmp/duplex-control40-backup-snapshots/20261024T050444.013404Z/native-control40-precise-pilot-v11-environment-retry/training/training-state.pt');newsha=hashlib.sha256(new.read_bytes()).hexdigest();assert newsha=='52ef1ca99c780bded4e66d4fdfb57cc814ba03682a688f1579f14d5033778e9e'
a=torch.load(original,weights_only=True,map_location='cpu');b=torch.load(new,weights_only=True,map_location='cpu');assert a['completed_steps']==520 and b['completed_steps']==1
assert a['parameter_names']==b['parameter_names'];assert len(b['parameter_names'])==26
pg=b['optimizer']['param_groups'];assert len(pg)==1;pg=pg[0];beta1,beta2=pg['betas'];assert beta1==.9 and beta2==.999 and pg['lr']==2e-7 and pg['weight_decay']==0
oldids=[i for g in a['optimizer']['param_groups']for i in g['params']];ids=pg['params'];assert len(ids)==len(oldids)==26
sums=dict(gradient_squared=0.,delta_squared=0.,gradient_dot_actual_delta=0.,parent_moment_squared=0.,gradient_dot_parent_moment=0.,rounding_dot_error_bound=0.);max_adam_delta_error=0.;per=[]
for name,oi,ni in zip(b['parameter_names'],oldids,ids):
 p0=a['adapters'][name];p1=b['adapters'][name];m0=a['optimizer']['state'][oi]['exp_avg'];state=b['optimizer']['state'][ni];m1=state['exp_avg'];v1=state['exp_avg_sq'];assert int(state['step'])==521 and int(a['optimizer']['state'][oi]['step'])==520
 assert p0.dtype==p1.dtype==m0.dtype==m1.dtype==v1.dtype==torch.float32
 g=(m1.double()-beta1*m0.double())/(1-beta1);delta=p1.double()-p0.double();m=m0.double()
 predicted=-pg['lr']*(m1.double()/(1-beta1**521))/(torch.sqrt(v1.double()/(1-beta2**521))+pg['eps'])
 max_adam_delta_error=max(max_adam_delta_error,float((predicted-delta).abs().max()))
 # Conservative diagnostic tolerance for reconstructing a FP32 moment update.
 u0=(torch.nextafter(m0,torch.full_like(m0,float('inf')))-m0).abs().double();u1=(torch.nextafter(m1,torch.full_like(m1,float('inf')))-m1).abs().double()
 err=20*(u1+beta1*u0)/(1-beta1)
 terms=dict(gradient_squared=float(g.square().sum()),delta_squared=float(delta.square().sum()),gradient_dot_actual_delta=float((g*delta).sum()),parent_moment_squared=float(m.square().sum()),gradient_dot_parent_moment=float((g*m).sum()),rounding_dot_error_bound=float((err*delta.abs()).sum()))
 for k,v in terms.items():sums[k]+=v
 per.append(dict(parameter=name,gradient_dot_actual_delta=terms['gradient_dot_actual_delta'],delta_norm=math.sqrt(terms['delta_squared'])))
gn=math.sqrt(sums['gradient_squared']);dn=math.sqrt(sums['delta_squared']);mn=math.sqrt(sums['parent_moment_squared']);dot=sums['gradient_dot_actual_delta'];bound=sums['rounding_dot_error_bound']
assert math.isfinite(gn)and abs(gn-1)<.005 and dn>0
r=dict(schema='dl-native-first-adam-direction-diagnostic-v1',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent_state_sha256=oldsha,first_state_sha256=newsha,parameter_count=sum(t.numel()for t in b['adapters'].values()),reconstructed_clipped_gradient_norm=gn,expected_clipped_gradient_norm_approximately=1.,actual_parameter_delta_norm=dn,gradient_dot_actual_delta=dot,gradient_update_cosine=dot/(gn*dn),gradient_parent_moment_cosine=sums['gradient_dot_parent_moment']/(gn*mn),conservative_FP32_reconstruction_dot_tolerance=bound,dot_sign_outside_diagnostic_tolerance=abs(dot)>bound,linearized_first_step_direction='ascent'if dot>bound else 'descent'if dot < -bound else 'unresolved',maximum_absolute_actual_vs_double_Adam_delta_error=max_adam_delta_error,per_parameter=per,model_loaded=False,GPU_initialized=torch.cuda.is_initialized(),optimizer_executed=False,training_settings_changed=False,scope='Approximate clipped gradient reconstructed from saved FP32 Adam moments; checks local first-order direction of the first selected joint objective. This does not measure post-update loss on the same partition, establish a cause of behavioral failure, or approve any optimizer change.',training_extension_or_adoption_approved=False)
assert not r['GPU_initialized'];print(json.dumps(r))
