"""Build decisions from complete checkpoint-backed metrics; never modify training."""
import argparse,collections,hashlib,json,math,time
from pathlib import Path
from .adaptive_policy import Policy,SCHEMA,digest,sha,decide,program


def read(path):
    blob=Path(path).read_bytes();return json.loads(blob),hashlib.sha256(blob).hexdigest()


def positive(value):return type(value)in(int,float)and math.isfinite(value)and value>=0


def development(path,ids,step):
    report,hash_value=read(path);rows=report['rows']
    if report['step']!=step or len(rows)!=len(ids)or {r['id']for r in rows}!=ids:
        raise ValueError('Complete exact DEV membership and boundary required')
    if any(not positive(r[k])for r in rows for k in ('text','audio')):raise ValueError('Finite DEV metrics required')
    mean={k:sum(r[k]for r in rows)/len(rows)for k in ('text','audio')}
    if mean!=report['mean']:raise ValueError('Reported DEV mean differs from complete rows')
    return report,hash_value


def backup(path):
    receipt,hash_value=read(path);primary=Path(receipt['source']).name;files=receipt['files']
    expected={primary,primary+'.sha256.json'}|{primary+'.rank-%04d.pt'%i for i in range(4)}
    if receipt.get('complete')is not True or type(receipt.get('step'))is not int or len(files)!=6 or {f['file']for f in files}!=expected:
        raise ValueError('Complete common and all four rank backup receipts required')
    parent=Path(path).parent
    for item in files:
        name=item['file'];file=parent/name
        if Path(name).name!=name or file.is_symlink()or not file.is_file()or file.stat().st_size!=item['bytes']or not sha(item['sha256']):
            raise ValueError('Backup file identity/size/hash descriptor differs')
    meta,meta_hash=read(parent/(primary+'.sha256.json'))
    by_name={r['file']:r for r in files}
    if meta_hash!=by_name[primary+'.sha256.json']['sha256']or meta['file']!=primary or meta['step']!=receipt['step']or meta['sha256']!=by_name[primary]['sha256']:
        raise ValueError('Checkpoint metadata and backup disagree')
    # Large state bytes are verified by the native publisher and again on
    # fresh restoration. This reader verifies/binds their immutable receipt.
    return receipt['step'],meta['sha256'],hash_value


def training(paths,start,end,train_ids,old_ids,world=4):
    indexed={};artifacts=[]
    for path in paths:
        prefix=hashlib.sha256();size=0
        with Path(path).open('rb')as stream:
            for blob in stream:
                if not blob.endswith(b'\n'):break  # concurrent writer's unfinished tail
                row=json.loads(blob);step=row['step'];rank=row['rank']
                if step>end:break
                prefix.update(blob);size+=len(blob)
                if step<start:continue
                key=(step,rank)
                if type(step)is not int or type(rank)is not int or not 0<=rank<world or row['world']!=world or row['actual_optimizer_updates']!=1 or key in indexed:
                    raise ValueError('Ambiguous/replayed or invalid TRAIN log lineage')
                if not positive(row['global_loss'])or not positive(row['opal']['global_preclip_norm'])or row['opal']['program']['update']!=step:
                    raise ValueError('Invalid finite optimizer observations')
                indexed[key]=row
        artifacts.append(dict(path=str(path),prefix_bytes=size,sha256=prefix.hexdigest()))
    if set(indexed)!={(s,r)for s in range(start,end+1)for r in range(world)}:
        raise ValueError('Every completed epoch step on every rank is required')
    seen=collections.Counter();retained=set();new_metrics=dict(text=0.,audio=0.)
    for step in range(start,end+1):
        rows=[indexed[step,rank]for rank in range(world)]
        if len({r['retained']['exposure_sha256']for r in rows})!=1 or len({r['global_loss']for r in rows})!=1:
            raise ValueError('Retained exposures or global loss differ across ranks')
        for r in rows:
            for item in r['expanded']:
                if type(item['valid'])is not bool or item['id']not in train_ids:raise ValueError('Only declared TRAIN ids allowed')
                mass=item['global_example_scale']
                if not positive(mass)or bool(mass>0)!=item['valid']:raise ValueError('Invalid padding/example mass')
                if item['valid']:
                    if not positive(item['text_ce'])or not positive(item['acoustic_ce']):raise ValueError('Invalid complete native pair loss')
                    seen[item['id']]+=1;new_metrics['text']+=item['text_ce'];new_metrics['audio']+=item['acoustic_ce']
        for item in rows[0]['retained']['records']:
            key=(item['group'],item['id'])
            if key not in old_ids:raise ValueError('Unknown retained rehearsal component')
            retained.add(key)
    if set(seen)!=train_ids or any(n!=1 for n in seen.values())or retained!=old_ids:
        raise ValueError('Full unique TRAIN exposure and original rehearsal coverage required')
    return dict(new_train=len(seen),retained=len(retained)),{k:v/len(seen)for k,v in new_metrics.items()},artifacts


def build_request(*,native,retained,dev,previous_dev,backup_receipt,logs,multiplier,policy=Policy(),now=None,behaviour=None):
    data,data_hash=read(native);old,old_hash=read(retained)
    if not data.get('complete1072')or data['TRAIN']!=policy.train_count or data['DEV']!=policy.dev_count:raise ValueError('Qualified full native data required')
    train_ids={r['id']for r in data['rows']if r['fold']=='TRAIN'};dev_ids={r['id']for r in data['rows']if r['fold']=='DEV'}
    if len(data['rows'])!=len(train_ids|dev_ids)or len(train_ids)!=policy.train_count or len(dev_ids)!=policy.dev_count or train_ids&dev_ids:raise ValueError('Exact disjoint TRAIN/DEV folds required')
    old_ids={(group,row['id'])for group,rows in old['input_tensor_fingerprints'].items()for row in rows}
    if not old.get('complete323')or len(old_ids)!=policy.retained_count:raise ValueError('All original retained identities required')
    step,checkpoint_hash,backup_hash=backup(backup_receipt);boundary=step//policy.epoch_updates*policy.epoch_updates
    if boundary<policy.epoch_updates:raise ValueError('No complete evaluated epoch yet')
    current,dev_hash=development(dev,dev_ids,boundary);previous,previous_hash=development(previous_dev,dev_ids,boundary-policy.epoch_updates)
    coverage,summary,artifacts=training(logs,boundary-policy.epoch_updates+1,boundary,train_ids,old_ids)
    coverage['dev']=len(dev_ids)
    retention=None
    if 'retention_reference'in current and 'retention_reference'in previous:
        retention=dict(previous=previous['retention_reference'],current=current['retention_reference'])
    behaviour_packet=None
    if behaviour is not None:
        report,report_hash=read(behaviour)
        if report.get('fold')!='DEV' or report.get('complete')is not True or type(report.get('passed'))is not bool:
            raise ValueError('Qualified separate DEV behaviour report required')
        behaviour_packet=dict(qualified_DEV=True,passed=report['passed'],report_sha256=report_hash)
    return dict(schema=SCHEMA,policy_sha256=policy.identity,boundary=boundary,resume_step=step,
        data_sha256=data_hash,checkpoint_sha256=checkpoint_hash,backup_sha256=backup_hash,
        dev_sha256=dev_hash,previous_dev_sha256=previous_hash,created_unix=time.time()if now is None else now,
        current_multiplier=multiplier,coverage=coverage,backup_complete=True,previous_dev=previous['mean'],dev=current['mean'],
        retention=retention,behaviour=behaviour_packet,train_summary=summary,
        artifacts=dict(native=data_hash,retained=old_hash,training_prefixes=artifacts))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('native','retained','dev','previous-dev','backup-receipt','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--logs',type=Path,nargs='+',required=True);p.add_argument('--multiplier',type=float,default=1.);p.add_argument('--behaviour',type=Path)
    a=p.parse_args();request=build_request(native=a.native,retained=a.retained,dev=a.dev,previous_dev=a.previous_dev,backup_receipt=a.backup_receipt,logs=a.logs,multiplier=a.multiplier,behaviour=a.behaviour)
    decision=decide(request);payload=dict(status='DECISION_READY_NOT_APPLIED',request=request,decision=decision,
        program_sha256=hashlib.sha256(program(decision['multiplier']).encode()).hexdigest())
    with a.out.open('x')as stream:json.dump(payload,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=payload['status'],boundary=request['boundary'],action=decision['action'],multiplier=decision['multiplier'])))


if __name__=='__main__':main()
