"""Select one hash-bound, causally eligible future capture request."""
import argparse
import json
from pathlib import Path
import re
from evals.moshi_model_download import sha256


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--eligibility',type=Path,required=True)
    ap.add_argument('--sha256',required=True)
    ap.add_argument('--map',type=Path,required=True)
    ap.add_argument('--index',type=int,required=True)
    args=ap.parse_args()
    if sha256(args.eligibility)!=args.sha256:raise RuntimeError('Changed eligibility report')
    report=json.loads(args.eligibility.read_text());mapping=json.loads(args.map.read_text())
    if (report['schema']!='dl-context-capture-eligibility-v1' or report['training_approved'] is not False
            or report['labels_created'] is not False or report['heldout_used'] is not False
            or report['optimizer_steps']!=0 or report['capture_map_sha256']!=sha256(args.map)
            or mapping['schema']!='dl-context-actual-capture-map-v1' or mapping['training_approved'] is not False):
        raise RuntimeError('Future selection needs unmodified TRAIN eligibility and capture map')
    eligible=[r for r in report['cases'] if r['eligible_for_strict_early_capture']]
    if not 0<=args.index<len(eligible):raise RuntimeError('Invalid future-capture index')
    row=eligible[args.index];original=mapping['cases'][row['index']]
    if (any(row[k]!=original[k] for k in ('job','teacher','scenario'))
            or row['kind']!=original['expected_kind'] or row['kind'] not in ('private','clarify','simple')
            or row['raw_history_validated'] is not True or row['training_approved'] is not False
            or not re.fullmatch('[0-9]+',row['job'])
            or not re.fullmatch('[a-zA-Z0-9_-]+',row['teacher'])
            or not re.fullmatch('[a-f0-9]{40}',mapping['collector_commit'])
            or not re.fullmatch(r'plan-[0-9]+\.json',row['plan_file'])
            or sha256(args.eligibility.parent/row['plan_file'])!=row['plan_sha256']):
        raise RuntimeError('Future request differs from its validated source/teacher plan')
    plan=json.loads((args.eligibility.parent/row['plan_file']).read_text())
    if row['kind']=='simple' and (plan.get('target_kind')!='ordinary_answer'
            or plan.get('ordinary_answer_without_backend_context') is not True):
        raise RuntimeError('Ordinary future requires the separate no-backend-context plan')
    print(row['job']);print(mapping['collector_commit']);print(row['teacher'])


if __name__=='__main__':main()
