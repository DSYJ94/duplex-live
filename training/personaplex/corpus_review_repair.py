"""Archive and losslessly reuse stopped reviewer generations before continuation."""
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
from .expanded_guard import ACCOUNT,CORPUS_REPAIR_AUTH,sha
from .native_feature_host import archive
from .corpus_semantic_review import reused_review
from .local_reasoner_reference import PIN_SHA,REVISION

BASE=Path('/projects/u6ada/duplex-live')
PARENT=BASE/'native-public-corpus-review-20261117-v1'
PRODUCER='707414c0ca7c5ac772a237a9aeaf450461bda579'


def accounting():
    raw=subprocess.check_output(['sacct','-j','8555523','-n','-P','-o',
        'JobIDRaw,State,ElapsedRaw,AllocTRES,ExitCode,Account'],universal_newlines=True,timeout=30)
    rows=[line.split('|')for line in raw.splitlines()if line.strip()]
    main=[row for row in rows if row[0]=='8555523']
    if (len(main)!=1 or not main[0][1].startswith('CANCELLED') or main[0][5]!=ACCOUNT
            or any(row[1].split()[0]not in ('CANCELLED','FAILED','COMPLETED')for row in rows)
            or not rows or max(int(row[2])for row in rows)>600):
        raise ValueError('Exact fully stopped bounded reviewer parent required')
    return dict(raw=raw,parent_main_seconds=int(main[0][2]),
                parent_process_seconds_upper=max(int(row[2])for row in rows),parent_seconds_reserved=600,
                accounting_padding_GPU_seconds=4*(600-int(main[0][2])))


def prepare(root,backup,source,inputs,readiness):
    readiness=Path(readiness);ready=json.loads(readiness.read_text())
    if (ready.get('passed') is not True or ready.get('source')!=source
            or ready.get('model_pin_sha256')!=PIN_SHA or ready.get('actual_left_padding') is not True
            or ready.get('complete_token_sequences_equal') is not True or ready.get('weights_loaded') is not False):
        raise ValueError('Actual pinned CPU batch-tokenizer readiness required')
    parent=accounting();lease=json.loads((PARENT/'job-8555523/lease.json').read_text())
    if lease.get('source')!=PRODUCER or lease.get('GPU_count')!=4:
        raise ValueError('Exact original frozen reviewer required')
    rows=[]
    for entry in inputs:
        if sha(entry['path'])!=entry['sha256']:raise ValueError('Exact whole public source candidates required')
        rows.extend(json.loads(line)for line in Path(entry['path']).open())
    indexed={row['id']:row for row in rows}
    if len(rows)!=17276 or len(indexed)!=17276:raise ValueError('Whole distinct public corpus required')
    root=Path(root);backup=Path(backup);root.mkdir(exist_ok=False)
    receipts={};records=[];statuses=Counter();malformed=[]
    for rank in range(4):
        receipt=archive(PARENT/('rank-%02d'%rank),backup/'failed-8555523'/('rank-%02d'%rank))
        receipts[str(rank)]=receipt
        for path in sorted(Path(receipt['destination']).glob('*.json')):
            if path.name.startswith('archive-')or path.name=='terminal.json':continue
            try:value=json.loads(path.read_text())
            except ValueError:
                malformed.append(dict(path=str(path),sha256=sha(path)));continue
            if value.get('id')not in indexed:raise ValueError('Unexpected original review source identity')
            entry=dict(id=value['id'],path=str(path),sha256=sha(path),producer_source=PRODUCER)
            derived=reused_review(indexed[value['id']],entry)
            records.append(entry);statuses[derived['review_status']]+=1
    if len({row['id']for row in records})!=len(records):raise ValueError('Never reuse duplicate source generations')
    controls=root/'parent-control';controls.mkdir()
    for path in (PARENT/'plan.json',PARENT/'slurm-8555523.log',PARENT/'job-8555523/lease.json',PARENT/'job-8555523/provider-admission.txt'):
        shutil.copyfile(path,controls/path.name)
        if sha(path)!=sha(controls/path.name):raise ValueError('Original failed control bytes changed')
    (controls/'terminal-accounting.json').write_text(json.dumps(parent,indent=2)+'\n')
    control_receipt=archive(controls,backup/'failed-8555523/control')
    reuse=root/'reuse.json';reuse.write_text(json.dumps(dict(records=records,malformed_originals_preserved=malformed),indent=2)+'\n')
    proof=dict(parent=parent,original_source=PRODUCER,backups=receipts,control_backup=control_receipt,
               reusable_records=len(records),reparsed_status_counts=dict(statuses),
               original_malformed_files_preserved=malformed,original_generations_repeated=0,
               factual_evidence_or_notes_checks_relaxed=False)
    path=root/'repair-proof.json';path.write_text(json.dumps(proof,indent=2)+'\n')
    plan=dict(source=source,inputs=inputs,required_candidates=17276,world_size=4,batch_size=4,
              reviewer_pin_sha256=PIN_SHA,reviewer_revision=REVISION,optimizer_updates=0,external_API_calls=0,
              training_admitted=False,reuse_manifest=str(reuse),reuse_manifest_sha256=sha(reuse),
              batch_tokenizer_readiness=str(readiness),batch_tokenizer_readiness_sha256=sha(readiness),
              repair_proof=str(path),repair_proof_sha256=sha(path),repair_scope_sha256=sha(CORPUS_REPAIR_AUTH))
    (root/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(proof),flush=True)


def validate(plan):
    parent=accounting()
    if plan.get('repair_scope_sha256')!=sha(CORPUS_REPAIR_AUTH)or plan.get('batch_size')!=4:
        raise ValueError('Exact bounded batched-format repair required')
    path=Path(plan['repair_proof'])
    if sha(path)!=plan['repair_proof_sha256']:raise ValueError('Actual repair proof changed')
    proof=json.loads(path.read_text())
    if proof.get('original_source')!=PRODUCER or len(proof.get('backups',{}))!=4:
        raise ValueError('All original partial reviewer archives required')
    for receipt in list(proof['backups'].values())+[proof['control_backup']]:
        if (receipt.get('passed') is not True or receipt.get('every_file_sha256_verified') is not True
                or sha(Path(receipt['destination'])/'archive-inventory.json')!=receipt['inventory_sha256']):
            raise ValueError('Independent exact failed-output preservation required')
    if sha(plan['reuse_manifest'])!=plan['reuse_manifest_sha256']:raise ValueError('Actual reuse manifest changed')
    if sha(plan['batch_tokenizer_readiness'])!=plan['batch_tokenizer_readiness_sha256']:
        raise ValueError('Actual CPU batch-tokenizer readiness changed')
    return parent['accounting_padding_GPU_seconds']
