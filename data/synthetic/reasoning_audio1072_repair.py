"""Versioned re-render of diagnosed peak failures only; original text stays exact."""
import argparse,json,sys
from pathlib import Path
from unittest.mock import patch
from . import reasoning_audio1072 as renderer


def main():
    p=argparse.ArgumentParser()
    for name in ('plan','inventory','key-file','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();inventory=json.loads(a.inventory.read_text());plan=json.loads(a.plan.read_text())
    if not inventory['complete'] or len(inventory['rows'])!=2144:raise ValueError('Complete original inventory required')
    failed={r['id'] for r in inventory['rows'] if not r['raw_peak_pass']}
    if not failed:raise ValueError('No diagnosed raw peak failures')
    # Reuse renderer's credit, receipt and ambiguity protections. Its complete
    # cohort remains2144: unchanged artifacts are linked in a NEW version.
    original=a.inventory.parent.parent/'native-reasoning-audio1072-20261113-v1'
    if not original.is_dir():raise ValueError('Original immutable collection missing')
    a.out.mkdir(parents=True,exist_ok=True)
    for row in plan['exact_speech_payloads']:
        if row['id'] in failed:continue
        for suffix in ('.json','.wav'):
            target=a.out/(row['id']+suffix);source=(original/target.name).resolve()
            if not source.is_file():raise ValueError('Missing original '+str(source))
            if not target.exists():target.symlink_to(source)
            elif not target.is_symlink() or target.resolve()!=source:raise ValueError('Unexpected replacement')
    lineage=a.out/'repair-lineage.json'
    expected=dict(original=str(original.resolve()),repaired_ids=sorted(failed),reason='raw_peak_above_0.995',words_changed=False,gain_changed=False,whole_cohort_required=2144)
    if lineage.exists():
        if json.loads(lineage.read_text())!=expected:raise ValueError('Repair identity changed')
    else:renderer.write(lineage,expected)
    with patch.object(sys,'argv',['renderer','--plan',str(a.plan),'--key-file',str(a.key_file),'--out',str(a.out)]):renderer.main()


if __name__=='__main__':main()
