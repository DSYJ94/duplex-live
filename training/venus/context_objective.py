"""Admit complete240/64 context phase without relaxing the old108 authority."""
import hashlib
import json
from pathlib import Path

ADMISSION=Path('docs/NATIVE-VENUS-CONTEXT-TRAINING-ADMISSION-2026-11-01.json')
ADMISSION_V2=Path('docs/NATIVE-VENUS-CONTEXT-TRAINING-ADMISSION-V2-2026-11-02.json')
PLACEMENT_AUTH=Path('docs/NATIVE-VENUS-CONTEXT-ZONAL-CAPACITY-AUTHORIZATION-2026-11-02.json')
ADMISSION_V3=Path('docs/NATIVE-VENUS-CONTEXT-TRAINING-ADMISSION-V3-2026-11-02.json')
CONTINUATION_AUTH=Path('docs/NATIVE-VENUS-CONTEXT-TRAINING-CONTINUATION-AUTHORIZATION-2026-11-02.json')
OLD_INDEX_SHA='e1ded570cb385b49bcd2e9560f9e4a8cf6c6286e66754f264f7282a6fd7ef541'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def authority(path=ADMISSION):
    path=Path(path)
    if path.resolve() not in (ADMISSION.resolve(),ADMISSION_V2.resolve(),ADMISSION_V3.resolve()):raise ValueError('Exact context phase admission required')
    a=json.loads(path.read_text())
    if path.resolve()==ADMISSION_V2.resolve():
        p=json.loads(PLACEMENT_AUTH.read_text())
        if p['status']!='APPROVED' or p['human_approval']!='Approve placement fallback within $70' or sha(p['proposal_path'])!=p['proposal_sha256']:
            raise ValueError('Exact approved placement required')
        if a['placement_authorization_sha256']!=sha(PLACEMENT_AUTH) or a['prior_admission_sha256']!=sha(ADMISSION):raise ValueError('Preserve exact prior phase and new placement authority')
        review=json.loads(Path('docs/NATIVE-VENUS-CONTEXT-PRELAUNCH-BOUNDARY-FIX-2026-11-01.json').read_text())
        if a['new_timeline_manifest_sha256']!=review['new_manifest_sha256']:raise ValueError('Corrected v3 native boundaries required')
    continuation=path.resolve()==ADMISSION_V3.resolve()
    if continuation:
        p=json.loads(CONTINUATION_AUTH.read_text())
        proposal=json.loads(Path(p['proposal_path']).read_text())
        if p['status']!='APPROVED' or p['human_approval']!='Approved - proceed' or sha(p['proposal_path'])!=p['proposal_sha256']:
            raise ValueError('Exact approved continuation required')
        if (a['continuation_authorization_sha256']!=sha(CONTINUATION_AUTH) or a['prior_admission_sha256']!=sha(ADMISSION_V2)
                or a['partial14_witness_sha256']!=sha('docs/NATIVE-VENUS-CONTEXT-PARTIAL14-REUSE-WITNESS-2026-11-02.json')
                or a['partial14_witness_sha256']!=proposal['partial14_witness_sha256']
                or a['focused_prior_gpu_seconds_upper']!=30000 or a['maximum_provider_running_seconds']!=6600):
            raise ValueError('Preserve exact partial14 and actual carried time')
        old=authority(ADMISSION_V2)
        for k in ('recipe','new_timeline_manifest_sha256','new_timeline_path','parent_checkpoint_sha256','old_prefix_manifest_sha256'):
            if a[k]!=old[k]:raise ValueError('Continuation must retain the original full recipe/data/state')
    if a['human_request']!='proceed with GCP for training' or a['maximum_new_updates']!=60 or a['parent_total_updates']!=406 or a['maximum_complete_epochs']!=1:
        raise ValueError('Only the new complete60 phase is admitted')
    if a['maximum_phase_usd']!=(75 if continuation else 70) or a['maximum_project_usd']!=175 or a['maximum_gpu_seconds']!=(6700 if continuation else 4100):
        raise ValueError('Existing time and financial caps are unchanged')
    return a


class ContextObjective:
    def __init__(self,path,cache_manifest,admission):
        a=authority(admission)
        index=json.loads(Path(path).read_text())
        manifest=json.loads(Path(cache_manifest).read_text())
        self.rows={r['id']:r for r in index['rows']}
        if not index['complete'] or index['prefix_manifest_sha256']!=sha(cache_manifest) or len(self.rows)!=304 or len(manifest['scenes'])!=304 or index['old_index_sha256']!=OLD_INDEX_SHA:
            raise ValueError('Full304 original plus new labels required')
        if manifest['old_prefix_manifest_sha256']!=a['old_prefix_manifest_sha256'] or manifest['context_source_manifest_sha256']!=a['new_timeline_manifest_sha256']:
            raise ValueError('Exact old/new corpus lineage required')
        if {f:sum(r['fold']==f for r in manifest['scenes']) for f in ('TRAIN','DEV')}!=dict(TRAIN=240,DEV=64):
            raise ValueError('All original and new folds required')
        for r in manifest['scenes']:
            original=self.rows[r['id']]
            if r['sha256']!=original['prefix_sha256'] or any(r[k]!=original[k] for k in ('fold','family')):
                raise ValueError('Cache target/fold changed')
        self.sha256=sha(path)
        self.ids=index['token_ids']
        self.weight=.25

    def validate(self,example):
        from .action_targets import masks
        row=self.rows[example['id']]
        target=hashlib.sha256(json.dumps(example['targets'].tolist(),separators=(',',':')).encode()).hexdigest()
        if target!=row['target_sha256'] or any(example[k]!=row[k] for k in ('fold','family')):
            raise ValueError('Original full targets required')
        selected,spans=masks(example['targets'],example['decision_mask'],self.ids)
        if spans!=row['action_spans'] or any(v.nonzero().flatten().tolist()!=row['indices'][k] for k,v in selected.items()):
            raise ValueError('Exact derived native masks required')
