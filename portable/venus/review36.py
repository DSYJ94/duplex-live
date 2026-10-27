"""Listen to all36 recorded native cases locally; no model or ASR runs here."""
import argparse,hashlib,html,json,re,shutil,struct
from pathlib import Path
from .review import write_float_wav


def pcm_bytes(path):
    raw=path.read_bytes();assert raw[:4]==b'RIFF' and raw[8:12]==b'WAVE';pos=12;fmt=None;pcm=None
    while pos+8<=len(raw):
        kind=raw[pos:pos+4];size=struct.unpack_from('<I',raw,pos+4)[0];data=raw[pos+8:pos+8+size];pos+=8+size+(size%2)
        if kind==b'fmt ':fmt=struct.unpack_from('<HHIIHH',data)
        if kind==b'data':pcm=data
    assert fmt and fmt[0]==3 and fmt[1]==1 and fmt[2]==24000 and fmt[5]==32 and pcm is not None
    return pcm


def main(a):
    root=Path(a.collection);timelines=Path(a.timelines);out=Path(a.output);out.mkdir(exist_ok=False)
    result=json.loads((root/'results.json').read_text());assert result['all36_complete'] and len(result['cases'])==36
    cards=[];audit=[]
    for case in result['cases']:
        ident=case['id'];assert re.fullmatch('[a-z0-9-]+',ident);dest=out/ident;dest.mkdir()
        rows=[json.loads(s) for s in (root/ident/'trace.jsonl').read_text().splitlines()]
        parts=[pcm_bytes(root/ident/r['chunk_file']) for r in rows];write_float_wav(dest/'native-joined.wav',b''.join(parts))
        source=timelines/ident/'user.wav';assert hashlib.sha256(source.read_bytes()).hexdigest()==case['input_sha256'];shutil.copy2(source,dest/'user.wav')
        shutil.copy2(root/ident/'trace.jsonl',dest/'trace.jsonl')
        timeline=json.loads((timelines/ident/'timeline.json').read_text());turns=''.join('<p>'+html.escape(t['text'])+'</p>' for t in timeline['user_turns'])
        text=''.join(r['text'] for r in rows);queries=[e['query'] for e in case['actual_worker_events'] if e['type']=='native_delegate'];results=[e['result'] for e in case['actual_worker_events'] if e['type']=='worker_completed']
        cards.append(f'<section><h2>{html.escape(ident)}</h2>{turns}<p>User input</p><audio controls preload="none" src="{ident}/user.wav"></audio><p>Native emitted output, exact float samples joined in order</p><audio controls preload="none" src="{ident}/native-joined.wav"></audio><p>Generated text; independent ASR pending: {html.escape(text)}</p><p>Actual queries: {html.escape(json.dumps(queries))}</p><p>Actual worker results: {html.escape(json.dumps(results))}</p><a href="{ident}/trace.jsonl">Original frame trace</a></section>')
        audit.append(dict(id=ident,input_sha256=case['input_sha256'],joined_emitted_sha256=hashlib.sha256((dest/'native-joined.wav').read_bytes()).hexdigest(),frames=len(rows),qualified=False))
    (out/'review.json').write_text(json.dumps(dict(all36_complete=True,qualified=False,adoption='REJECTED',source_sha256=hashlib.sha256((root/'results.json').read_bytes()).hexdigest(),cases=audit),indent=2)+'\n')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Venus step190 recorded review</title><style>body{font:16px system-ui;max-width:960px;margin:30px auto;padding:0 20px;color:#182531}section{border-top:1px solid #ccc;padding:18px 0}audio{width:100%;max-width:650px}p{line-height:1.5}</style><h1>Venus step190 — all36 recorded native cases</h1><p>190 training updates, five complete epochs. Candidate rejected for adoption: silence and cancellation checks fail. This page plays recorded synthetic tests; it is not a live model. No gain or normalization changes. Joined playback does not establish real-time delivery timing. All original chunks and traces remain preserved.</p><a href="review.json">Recording inventory</a>'+''.join(cards))
    print(json.dumps(dict(cases=36,index=str(out/'index.html'),qualified=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser();[p.add_argument('--'+x,required=True) for x in ('collection','timelines','output')];main(p.parse_args())
