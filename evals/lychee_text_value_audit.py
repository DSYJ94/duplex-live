"""Literal private-value timing diagnostic. Never a speech/task acceptance gate."""
import json
import re
from collections import defaultdict
from evals.moshi_model_download import sha256

COLOURS='red blue green yellow purple orange silver gold white black pink brown'.split()
SHAPES='circle square triangle star diamond oval heart cross'.split()
PATTERN=re.compile(r'\b('+ '|'.join(COLOURS)+r')[\W_]+('+ '|'.join(SHAPES)+r')\b',re.I)


def inspect_snapshots(snapshots, expected, delivered_at):
    groups=defaultdict(list)
    for row in snapshots:groups[row['event_id']].append(row)
    mentions=[]
    for identity,rows in groups.items():
        text=rows[-1]['text']
        for match in PATTERN.finditer(text):
            # A private word can start before the receipt and finish after it.
            # Attribute the first decoded character, not completion of the pair.
            starts=[r['at_s'] for r in rows if len(r['text'])>match.start()
                and text.startswith(r['text'])]
            if not starts:raise RuntimeError('Cannot map actual private text to its sampled-token clock')
            phrase=(match[1]+' '+match[2]).lower()
            mentions.append({'event_id':identity,'phrase':phrase,'first_word_token_at_s':min(starts),
                'before_actual_context_delivery':min(starts)<delivered_at,
                'equals_actual_receipt':phrase==expected.lower()})
    return {'expected':expected,'actual_context_delivered_at_s':delivered_at,'literal_mentions':mentions,
        'actual_value_mentioned_after_delivery':any(r['equals_actual_receipt'] and not r['before_actual_context_delivery'] for r in mentions),
        'any_colour_shape_started_before_delivery':any(r['before_actual_context_delivery'] for r in mentions),
        'other_colour_shape_mentions':[r for r in mentions if not r['equals_actual_receipt']],
        'spoken_result_use_verified':False,'task_success_verified':False,
        'scope':'sampled text literal/timing only; negation, claims, audio and complete task require separate grading'}


def audit(run, tokenizer, framework):
    from lychee_fd.runtime.vllm_generation import StreamingDecoder
    report=json.loads((run/'report.json').read_text())
    deliveries=[e for e in report['worker_events'] if e['event']=='actual_result_context_applied']
    if len(deliveries)!=1:raise RuntimeError('Literal value audit requires exactly one actual result delivery')
    delivery=deliveries[0]
    receipts=[r for r in delivery['result']['receipts'] if r['tool']=='lookup_record']
    if len(receipts)!=1:raise RuntimeError('Missing actual record receipt')
    decoder=StreamingDecoder(tokenizer,framework,tts_chunk_size=1);snapshots=[]
    for line in (run/'incremental-events.jsonl').read_text().splitlines():
        record=json.loads(line)
        for event in decoder.feed(record['event']):
            if event['type']=='text_delta':
                snapshots.append({'event_id':event['event_id'],'text':event['snapshot'],'at_s':record['available_s']})
    if not snapshots:raise RuntimeError('No sampled native text snapshots')
    result=inspect_snapshots(snapshots,receipts[0]['value'],delivery['observed_at_s'])
    result.update(schema='dl-lychee-text-value-diagnostic-v1',
        native_report_sha256=sha256(run/'report.json'),
        native_arrival_trace_sha256=sha256(run/'incremental-events.jsonl'))
    (run/'text-value-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
