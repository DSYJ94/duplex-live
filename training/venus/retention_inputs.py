"""Compose the preserved304 with all96 new native histories; no old gate edits."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random

from .large_inputs import Corpus as OriginalCorpus,full_native,sha
from data.venus.retention_speech import PLAN

LIMITS=dict(minimum_cosine=.9999,maximum_relative_l2=.01,
            maximum_target_ce_delta=.03,minimum_argmax_agreement=.95)


def validate_proof(proof):
    native=proof['native_segmented'];training=proof['differentiable_input']
    if (native['passed'] is not True or native['maximum_logit_error']!=0
            or not 0<native['compared']<=32 or training['passed'] is not True
            or training['compared']!=native['compared'] or training['limits']!=LIMITS):
        raise ValueError('Exact native replay and original declared numerical limits required')
    keys=('minimum_cosine','relative_l2','maximum_target_ce_delta','argmax_agreement')
    if not all(math.isfinite(training[k]) for k in keys):raise ValueError('Finite actual measurements required')
    if (training['minimum_cosine']<LIMITS['minimum_cosine']
            or not 0<=training['relative_l2']<=LIMITS['maximum_relative_l2']
            or not 0<=training['maximum_target_ce_delta']<=LIMITS['maximum_target_ce_delta']
            or training['argmax_agreement']<LIMITS['minimum_argmax_agreement']):
        raise ValueError('Recorded pass flag cannot hide failed native measurements')


def epoch_batches(old_rows,new_rows,arm,seed):
    """One matched336-exposure epoch: every old TRAIN scene plus96 exposures."""
    if arm not in ('expanded-data','rehearsal'):raise ValueError('Explicit paired data arm required')
    old=sorted((r for r in old_rows if r['fold']=='TRAIN'),key=lambda r:r['id'])
    if (len(old)!=240 or len({r['id'] for r in old})!=240
            or sum(r['fold']=='DEV' for r in old_rows)!=64
            or len(new_rows)!=96 or len({r['id'] for r in new_rows})!=96
            or any(r['fold']!='TRAIN' for r in new_rows)
            or {r['id'] for r in old_rows}&{r['id'] for r in new_rows}):
        raise ValueError('Whole old240/64 and new96 TRAIN required')
    wanted=Counter(r['family'] for r in new_rows)
    if wanted!=dict(ordinary=48,delegate=12,pending=12,cancel=24):
        raise ValueError('Exact admitted96 family mix required')
    if arm=='expanded-data':extra=sorted(new_rows,key=lambda r:r['id'])
    else:
        extra=[]
        for family,count in sorted(wanted.items()):
            choices=[r for r in old if r['family']==family]
            if not choices:raise ValueError('No original family for matched rehearsal')
            random.Random(str(seed)+':'+family).shuffle(choices)
            extra.extend(choices[i%len(choices)] for i in range(count))
    ordered=old+extra;random.Random(seed).shuffle(ordered)
    return [ordered[i:i+4] for i in range(0,336,4)]


class Corpus:
    def __init__(self,old_root,new_root,*,action_index,old_admission,
                 new_manifest_sha,semantic_root,timelines):
        from data.venus.retention_admission import admit
        # Original checks, exact hashes, all304 and ContextObjective stay intact.
        self.old=OriginalCorpus(old_root,action_index=action_index,admission=old_admission)
        self.root=Path(new_root)
        if sha(self.root/'manifest.json')!=new_manifest_sha:raise ValueError('Pinned actual full96 cache required')
        manifest=json.loads((self.root/'manifest.json').read_text())
        replay=json.loads((self.root/'replay-approval.json').read_text())
        index=json.loads((self.root/'action-index.json').read_text())
        semantic=admit(timelines,semantic_root,manifest['semantic_report_sha256'])
        if (manifest['schema']!='dl-retention96-native-cache-v1' or manifest['complete96'] is not True
                or manifest['train']!=96 or manifest['dev']!=0 or manifest['optimizer_updates']!=0
                or manifest['training_approved'] is not False
                or manifest['source_manifest_sha256']!=semantic['timeline_manifest_sha256']
                or replay['complete96'] is not True or replay['all96_native_replay_and_input_checks'] is not True
                or replay['manifest_sha256']!=new_manifest_sha or replay['semantic_admission']!=semantic
                or replay['action_index_sha256']!=sha(self.root/'action-index.json')
                or index['manifest_sha256']!=new_manifest_sha or index['complete96'] is not True
                or index['token_ids']!=self.old.objective.ids):
            raise ValueError('Full independent semantic/replay/action lineage required')
        expected={r['id']:r for r in json.loads(PLAN.read_text())['scenes']}
        self.new_rows=manifest['scenes'];self.new_index={r['id']:r for r in index['rows']}
        if (len(self.new_rows)!=96 or len(self.new_index)!=96 or len(index['rows'])!=96
                or {r['id'] for r in self.new_rows}!=set(expected) or set(self.new_index)!=set(expected)):
            raise ValueError('Every new96 identity required, no duplicate or subset')
        self.old_ids={r['id'] for r in self.old.rows}
        if self.old_ids&set(expected):raise ValueError('Old and new identities must remain disjoint')
        for row in self.new_rows:
            if row['fold']!='TRAIN' or row['family']!=expected[row['id']]['family']:
                raise ValueError('No new DEV or changed teacher family')
            if self.new_index[row['id']]['cache_sha256']!=row['sha256']:
                raise ValueError('Native action index binds exact cache bytes')
            validate_proof(row['native_input_proof'])
        self.rows=self.old.rows+self.new_rows
        self.manifest=dict(schema='dl-preserved304-plus-native96-v1',old_manifest_sha256=sha(Path(old_root)/'manifest.json'),
                           new_manifest_sha256=new_manifest_sha,TRAIN=336,DEV=64)

    def validate(self,example):
        if example['id'] in self.old_ids:
            self.old.objective.validate(example);return
        from .action_targets import masks
        row=self.new_index[example['id']]
        target=hashlib.sha256(json.dumps(example['targets'].tolist(),separators=(',',':')).encode()).hexdigest()
        if target!=row['target_sha256'] or any(example[k]!=row[k] for k in ('fold','family')):
            raise ValueError('Preserve all original new native targets/folds')
        selected,spans=masks(example['targets'],example['decision_mask'],self.old.objective.ids)
        if spans!=row['action_spans'] or any(v.nonzero().flatten().tolist()!=row['indices'][k] for k,v in selected.items()):
            raise ValueError('Exact native action masks required')

    def read(self,row):
        if row['id'] in self.old_ids:return self.old.read(row)
        import torch
        if Path(row['file']).is_absolute() or '..' in Path(row['file']).parts:
            raise ValueError('Contained native artifact required')
        path=self.root/row['file']
        if sha(path)!=row['sha256']:raise ValueError('Actual new cache hash differs')
        example=torch.load(path,map_location='cpu',weights_only=False)
        if example['id']!=row['id']:raise ValueError('Native artifact identity differs')
        self.validate(example);return full_native(example)

    def loss(self,model,example,device):
        from .lora import scene_loss
        self.validate(example)
        return scene_loss(model,example,device,kl_weight=.1,hidden_weight=.01,
                          action_token_ids=self.old.objective.ids,action_aux_weight=.25)
