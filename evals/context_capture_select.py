"""Select one hash-bound TRAIN capture request without shell interpretation."""
import argparse
import json
from pathlib import Path
import re
from evals.moshi_model_download import sha256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--sha256',required=True)
    ap.add_argument('--index',type=int,required=True)
    args = ap.parse_args()
    if not re.fullmatch('[a-f0-9]{64}',args.sha256) or sha256(args.plan) != args.sha256:
        raise RuntimeError('Changed capture plan')
    plan = json.loads(args.plan.read_text())
    if (plan['schema'] != 'dl-context-approved-teacher-capture-plan-v1'
            or plan['heldout_used'] is not False or plan['labels_created'] is not False
            or plan['training_approved'] is not False or not 0 <= args.index < len(plan['planned'])):
        raise RuntimeError('Capture plan is not hash-bound unapproved TRAIN conditioning')
    row = plan['planned'][args.index]
    if (row['expected_kind'] not in ('private','simple','clarify','cancel')
            or row['teacher_kind'] != row['expected_kind']
            or not re.fullmatch(r'recovery-train-\d{3}',row['scenario'])
            or not re.fullmatch('[A-Za-z]+ [A-Za-z]+',row['private_value'])
            or row['teacher_source_model'] != 'gpt-live-1'
            or row['identical_user_pcm_and_words'] is not True
            or row['conditioning_only'] is not True or row['training_approved'] is not False):
        raise RuntimeError('Invalid approved-teacher capture request')
    print(row['scenario']); print(row['expected_kind']); print(row['private_value'])


if __name__ == '__main__': main()
