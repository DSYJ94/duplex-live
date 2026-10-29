"""Admit the exact human-approved action objective on unchanged cached labels."""
import hashlib
import json
from pathlib import Path

from .action_targets import masks

AUTH = Path('docs/NATIVE-VENUS-ACTION-FIDELITY-AUTHORIZATION-2026-10-29.json')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ActionObjective:
    def __init__(self, path, cache_manifest, authority=AUTH):
        if Path(authority).resolve() != AUTH.resolve():
            raise ValueError('Exact action-fidelity authority required')
        a = json.loads(AUTH.read_text())
        if (a['status'] != 'APPROVED' or a['human_approval'] != 'Approve 108 updates within existing time/$70 cap'
                or a['maximum_candidate_new_updates'] != 108 or a['maximum_candidate_total_updates'] != 406
                or a['external_model_api_and_synthesis_calls'] != 0
                or digest(a['proposal_path']) != a['proposal_sha256']):
            raise ValueError('Explicit unchanged action-fidelity update allowance required')
        index = json.loads(Path(path).read_text())
        self.sha256 = digest(path)
        if self.sha256 != a['proposal']['dataset']['target_index_sha256']:
            raise ValueError('Exact complete audited target index required')
        if not index['complete'] or index['prefix_manifest_sha256'] != digest(cache_manifest):
            raise ValueError('Audited prefix manifest differs')
        manifest = json.loads(Path(cache_manifest).read_text())
        self.rows = {r['id']: r for r in index['rows']}
        if len(self.rows) != 268 or len(manifest['scenes']) != 268:
            raise ValueError('All216TRAIN/52DEV labels required')
        for row in manifest['scenes']:
            admitted = self.rows[row['id']]
            if row['sha256'] != admitted['prefix_sha256'] or any(row[k] != admitted[k] for k in ('fold','family')):
                raise ValueError('Original cache/fold identity differs')
        self.ids = index['token_ids']
        self.weight = a['proposal']['objective']['coefficient']
        if self.weight != .25:
            raise ValueError('Only the reviewed auxiliary coefficient is admitted')

    def validate(self, example):
        row = self.rows[example['id']]
        target_hash = hashlib.sha256(json.dumps(example['targets'].tolist(), separators=(',', ':')).encode()).hexdigest()
        if (target_hash != row['target_sha256'] or example['fold'] not in ('TRAIN','DEV')
                or any(example[k] != row[k] for k in ('fold','family'))):
            raise ValueError('Unchanged original targets/folds required')
        selected, spans = masks(example['targets'], example['decision_mask'], self.ids)
        if spans != row['action_spans'] or any(mask.nonzero().flatten().tolist() != row['indices'][name]
                                             for name, mask in selected.items()):
            raise ValueError('Derived target masks differ from complete audit')

