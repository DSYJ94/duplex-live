"""Conservative source/input-family partitions; semantic paraphrases still pending."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
from .broad_corpus_overlap import tokens,key


def propose(rows,*,development_modulus=20):
    if type(development_modulus) is not int or development_modulus<2:
        raise ValueError('Explicit family-level development fraction required')
    rows=sorted(rows,key=lambda row:row['id'])
    if not rows or len({row['id'] for row in rows})!=len(rows):
        raise ValueError('Complete distinct source candidate identities required')
    parent={row['id']:row['id'] for row in rows};index={};edges=Counter()
    def root(identity):
        while parent[identity]!=identity:
            parent[identity]=parent[parent[identity]];identity=parent[identity]
        return identity
    for row in rows:
        if (row.get('admitted_for_training') is not False or row.get('split') is not None
                or row.get('official_source_split') not in ('train','unsplit')):
            raise ValueError('Only original eligible unadmitted TRAIN-source candidates required')
        user=[turn['text'] for turn in row['turns'] if turn['role']=='user']
        if not user or any(not isinstance(text,str) or not text.strip() for text in user):
            raise ValueError('Complete authored user inputs required')
        prompt='\n\n'.join(user)
        bindings=[('source_group',row['source_dataset']+'\0'+row['source_group_id'])]
        for mode in ('literal','numeric_template'):
            words=tokens(prompt,numeric_template=mode=='numeric_template')
            if words:bindings.append((mode,key(words)))
        for mode,binding in bindings:
            marker=(mode,binding)
            if marker in index:
                left,right=root(row['id']),root(index[marker])
                if left!=right:
                    parent[max(left,right)]=min(left,right);edges[mode]+=1
            else:index[marker]=row['id']
    groups=defaultdict(list)
    for row in rows:groups[root(row['id'])].append(row)
    assignments=[];family_counts=Counter();source_counts=defaultdict(Counter)
    cross_source=0
    for members in groups.values():
        ids=sorted(row['id'] for row in members)
        family=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
        fold='DEV' if int(family,16)%development_modulus==0 else 'TRAIN'
        family_counts[fold]+=1
        cross_source+=int(len({row['source_dataset'] for row in members})>1)
        for row in members:
            assignments.append(dict(id=row['id'],family_id=family,proposed_fold=fold,
                source_dataset=row['source_dataset'],source_group_id=row['source_group_id'],
                source_content_sha256=row['content_sha256'],leakage_qualified=False,training_admitted=False))
            source_counts[row['source_dataset']][fold]+=1
    return dict(assignments=sorted(assignments,key=lambda row:row['id']),
                report=dict(documents=len(rows),families=len(groups),family_counts=dict(family_counts),
                    source_counts={key:dict(value) for key,value in source_counts.items()},
                    union_edges=dict(edges),cross_source_families=cross_source,
                    methods=['whole_source_parent','exact_user_inputs','number_normalized_user_inputs'],
                    semantic_paraphrase_clustering_complete=False,historical_semantic_decontamination_complete=False,
                    folds_are_proposals_only=True,training_admitted=False))


def run(inputs,output):
    rows=[];inventory=[]
    for source in inputs:
        path=Path(source);inventory.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        rows.extend(json.loads(line) for line in path.open())
    result=propose(rows);output=Path(output);output.mkdir(exist_ok=False)
    target=output/'assignments.jsonl'
    with target.open('x') as stream:
        for row in result['assignments']:stream.write(json.dumps(row)+'\n')
    report=dict(result['report'],sources=inventory,assignments_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),GPU_count=0,optimizer_updates=0)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--inputs',nargs='+',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.inputs,a.output)
