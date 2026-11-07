"""All64 original recorded outputs, blind A/B per case; never live qualification."""
import argparse,hashlib,html,json,random,shutil
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(root,artifacts,out):
    root,artifacts,out=map(Path,(root,artifacts,out));report=json.loads((root/'paired-native-report.json').read_text());results=json.loads((root/'results.json').read_text())
    if not report['complete'] or report['conditions']!=['main','control'] or len(report['cases'])!=64 or len(results['cases'])!=64:raise ValueError('Complete original paired64 required')
    ids=sorted({r['id'] for r in report['cases']})
    if len(ids)!=32:raise ValueError('All32 paired scenes required')
    out.mkdir(parents=True,exist_ok=False);media=out/'media';media.mkdir();cards=[];key=[];inventory=[]
    by={(x['condition'],x['id']):x for x in results['cases']}
    for ident in ids:
        rows=[x for x in report['cases'] if x['id']==ident]
        if len(rows)!=2 or {x['condition'] for x in rows}!={'main','control'}:raise ValueError('Both outputs required for every scene')
        original=by['main',ident];other=by['control',ident]
        if original['input_sha256']!=other['input_sha256']:raise ValueError('Paired inputs differ')
        timeline='paced-timelines-v22' if original['evaluation_set']=='fresh24' else 'timelines-v7-repaired'
        source=artifacts/timeline/ident/'user.wav';t=json.loads((source.parent/'timeline.json').read_text())
        if sha(source)!=original['input_sha256']:raise ValueError('Original user WAV differs')
        name=ident+'-input.wav';shutil.copy2(source,media/name);inventory.append(dict(file='media/'+name,sha256=sha(source),role='synthetic_input'))
        random.Random('capacity-review-526-v1:'+ident).shuffle(rows);players=[]
        for label,row in zip('AB',rows):
            p=root/row['condition']/ident/'joined-emitted.wav'
            if sha(p)!=row['joined_wav_sha256']:raise ValueError('Original output WAV differs')
            name=ident+'-'+label+'.wav';shutil.copy2(p,media/name)
            inventory.append(dict(file='media/'+name,sha256=row['joined_wav_sha256'],role='original_output'))
            key.append(dict(id=ident,label=label,condition=row['condition'],sha256=row['joined_wav_sha256']))
            metrics=[]
            for metric,title in [('naturalness','Humanlike delivery'),('clarity','Clarity'),('british_fit','British English fit')]:
                options='<option value="">Not rated</option>'+''.join(f'<option value="{i}">{i}</option>' for i in range(1,6))
                metrics.append(f'<label>{title}<select data-case="{ident}" data-label="{label}" data-metric="{metric}">{options}</select></label>')
            players.append(f'<div class="clip"><h3>Recording {label}</h3><audio controls preload="none" src="media/{name}"></audio><div class="ratings">'+''.join(metrics)+'</div></div>')
        turns=''.join('<li>'+html.escape(u['text'])+'</li>' for u in t['user_turns'])
        family=html.escape(original['family']);cards.append(f'<section data-family="{family}"><span class="family">{family}</span><h2>{html.escape(ident)}</h2><details><summary>Read the synthetic user input</summary><ol>{turns}</ol></details><p>Listen to the input</p><audio controls preload="none" src="media/{ident}-input.wav"></audio><div class="pair">'+''.join(players)+'</div></section>')
    source_sha=sha(root/'paired-native-report.json')
    page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Recorded native voice comparison</title><style>
*{box-sizing:border-box}body{font:17px/1.5 system-ui,sans-serif;background:#f5f6f3;color:#182a28;margin:0}main{max-width:1120px;margin:40px auto;padding:0 24px}h1{font-size:34px;line-height:1.2}h2{font-size:17px;overflow-wrap:anywhere;margin:6px 0 16px}h3{font-size:19px}p{max-width:850px}section{background:white;border:1px solid #dbe2dc;border-radius:16px;margin:24px 0;padding:24px}.badge,.family{background:#fff0ce;color:#705117;border-radius:16px;padding:5px 10px;font-size:14px}.family{background:#e9f2ed;color:#245941}.pair{display:grid;grid-template-columns:1fr 1fr;gap:28px}.clip{min-width:0}.ratings{display:grid;gap:8px;margin-top:14px}label{display:flex;justify-content:space-between;align-items:center;gap:12px}select,button{font:inherit;border:1px solid #9caea5;border-radius:8px;padding:8px;background:white}button{background:#225e4c;color:white;cursor:pointer}audio{width:100%;max-width:490px}.toolbar{display:flex;align-items:center;gap:18px;flex-wrap:wrap;position:sticky;top:0;background:#f5f6f3;padding:14px 0;z-index:1}.muted{color:#4d625a}details{margin-bottom:16px}li{margin:7px 0}@media(max-width:720px){.pair{grid-template-columns:1fr}main{padding:0 16px}section{padding:18px}h1{font-size:28px}}
</style></head><body><main><span class="badge">Recorded synthetic tests · experimental models</span><h1>Compare the native voices</h1><p>Listen to two recordings of each case and rate their delivery. Both use the RP Ben reference. A and B are shuffled separately for every case. All 32 cases and 64 outputs are available.</p><p class="muted">These recordings are from checkpoint 526, before the current continuation. They play locally without a GPU or microphone. Original words, audio levels and samples are unchanged. Joined recordings preserve output chunks; they do not reproduce actual live processing delays.</p><p>Ratings: 1 = poor, 3 = usable with noticeable issues, 5 = excellent. Rate British English fit against your preferred accent. Your ratings stay in this browser until you save the file; they do not approve a model.</p><div class="toolbar"><label>Cases <select id="family"><option value="all">All families</option>'''+''.join(f'<option value="{f}">{f.title()}</option>' for f in ('ordinary','quiet','delegate','pending','cancel','pause'))+'''</select></label><span id="visible">32 of 32 cases</span><button id="export">Save listening ratings</button><span id="rated" class="muted"></span></div>'''+''.join(cards)+'''<p class="muted">Both candidates remain unqualified. Speech correctness, ordinary reasoning, silence, cancellation, delegation and live latency require their separate complete tests.</p></main><script>
const sourceSHA=__SHA__;const key='native-capacity-british64-526-v1';let ratings={};try{ratings=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}
function count(){document.getElementById('rated').textContent=Object.values(ratings).filter(Boolean).length+' ratings saved locally'}
for(const el of document.querySelectorAll('select[data-case]')){const k=[el.dataset.case,el.dataset.label,el.dataset.metric].join('|');el.value=ratings[k]||'';el.onchange=()=>{ratings[k]=el.value;try{localStorage.setItem(key,JSON.stringify(ratings))}catch(e){}count()}}
const filter=document.getElementById('family');filter.onchange=()=>{let visible=0;for(const card of document.querySelectorAll('section')){card.hidden=filter.value!=='all'&&card.dataset.family!==filter.value;if(!card.hidden)visible++}document.getElementById('visible').textContent=visible+' of 32 cases'};
document.getElementById('export').onclick=()=>{const payload={schema:'dl-native-capacity-voice-ratings-v1',source_report_sha256:sourceSHA,checkpoint_global_updates:526,at:new Date().toISOString(),ratings,qualified:false,adopted:false};const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));const link=document.createElement('a');link.href=url;link.download='native-capacity-voice-listening-ratings.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};count();
</script></body></html>'''.replace('__SHA__',json.dumps(source_sha))
    (out/'index.html').write_text(page)
    manifest=dict(source_report_sha256=source_sha,checkpoint_global_updates=526,cases=32,recordings=64,synthetic_inputs=32,files=inventory,word_or_gain_changes=False,live_model=False,qualified=False,adopted=False)
    (out/'recording-inventory.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out.parent/(out.name+'-condition-key.json')).write_text(json.dumps(dict(source_report_sha256=source_sha,rows=key),indent=2)+'\n')
    return dict(cases=32,recordings=64,synthetic_inputs=32,files=len(inventory),source_report_sha256=source_sha,index=str(out/'index.html'),qualified=False)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','artifacts','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();print(json.dumps(build(a.root,a.artifacts,a.output)))
