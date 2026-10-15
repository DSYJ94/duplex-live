"""Enforce assistant-only future text targets; this is not waveform AEC.

Validate target ownership without deleting user input or assistant history.
The current semantic experiment has no user-prediction or audio-prediction loss.
"""
import math


def check(input_ids, labels, expected, history_end, system_length, received_at_s,
        declared_first_s, declared_count):
    if len(input_ids)!=len(labels) or not 0<system_length<=history_end<=len(labels):
        raise RuntimeError('Invalid native loss-scope alignment')
    slots=[i for i,value in enumerate(labels) if value!=-100]
    if (not slots or slots[0]<history_end or slots[0]<1
            or slots!=list(range(slots[0],slots[0]+len(slots)))
            or len(slots)!=declared_count):
        raise RuntimeError('Loss includes history or noncontiguous/nondeclared assistant targets')
    targets=[labels[i] for i in slots]
    if targets!=list(expected) or targets!=[input_ids[i] for i in slots]:
        raise RuntimeError('Loss targets differ from independently sourced assistant text and EOT')
    first_s=(slots[0]-system_length)*.04
    if (not all(math.isfinite(v) for v in (received_at_s,declared_first_s))
            or received_at_s<0 or first_s+1e-9<received_at_s
            or abs(first_s-declared_first_s)>1e-8):
        raise RuntimeError('Assistant target precedes actual receipt or has a different native clock')
    return {'assistant_text_targets':len(slots),'user_prediction_targets':0,
        'audio_prediction_targets':0,'history_targets':0,
        'first_target_time_s':first_s,'actual_receipt_time_s':received_at_s,
        'input_tokens_modified':False,'attention_mask_modified':False,
        'aec_applied':False,'scope':'assistant semantic text/EOT only; no user-loss objective to mask'}


def audit(item, row, tokenizer):
    expected=tokenizer.encode(row['gold_text'],add_special_tokens=False)
    end=tokenizer.encode('<|EOT|>',add_special_tokens=False)
    if len(end)!=1:raise RuntimeError('Native EOT token identity changed')
    return check(item['input_ids'].tolist(),item['labels'].tolist(),expected+end,
        row['native_past_prefill_tokens'],item['prefix_len'],row['actual_receipt']['received_at_s'],
        row['first_supervised_token_time_s'],row['loss_positions'])


def main():
    import argparse
    import json
    from pathlib import Path
    import torch
    from transformers import AutoTokenizer
    from evals.moshi_model_download import sha256
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('data','assets','out'):ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args()
    data=json.loads((args.data/'semantic-data.json').read_text())
    if data['schema']!='dl-lychee-native-semantic-data-v1' or not data['training_approved']:
        raise RuntimeError('Approved native semantic data required')
    assets=json.loads(args.assets.read_text());root=Path(assets['models']['duplex']['path'])
    for name,record in assets['models']['duplex']['files'].items():
        if not name.endswith('.safetensors') and sha256(root/name)!=record['sha256']:
            raise RuntimeError('Pinned native tokenizer/config changed')
    tokenizer=AutoTokenizer.from_pretrained(root,local_files_only=True,trust_remote_code=True)
    results=[]
    for row in data['examples']:
        if row['source_split']!='train' or sha256(args.data/row['file'])!=row['sha256']:
            raise RuntimeError('Native target source fold/hash changed')
        item=torch.load(args.data/row['file'],map_location='cpu',weights_only=True)
        results.append({'source_scenario':row['source_scenario'],**audit(item,row,tokenizer)})
    if not results:raise RuntimeError('No native targets to check')
    result={'schema':'dl-lychee-loss-scope-audit-v1','results':results,
        'semantic_data_sha256':sha256(args.data/'semantic-data.json'),
        'base_assets_sha256':sha256(args.assets),'user_loss_mask_changes_objective':False,
        'aec_tested':False,'training_performed':False,'task_success_verified':False}
    args.out.mkdir(exist_ok=False)
    (args.out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
