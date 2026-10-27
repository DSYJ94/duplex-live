"""Local, blind-labelled listening sheet for all24 native comparison recordings."""
import argparse,hashlib,html,json,random,shutil
from pathlib import Path

def build(root,out,source_plan):
    root,out=Path(root),Path(out);a=json.loads((root/'voice-comparison.json').read_text())
    assert a['complete'] and len(a['cases'])==24 and a['cases_per_condition']==8
    questions={s['id']:s['user_turns'][0] for s in json.loads(Path(source_plan).read_text())['scenes']}
    ids=sorted({c['id'] for c in a['cases']});assert len(ids)==8
    out.mkdir(exist_ok=False);media=out/'media';media.mkdir();sections=[];key=[]
    for ident in ids:
        rows=[c for c in a['cases'] if c['id']==ident];assert len(rows)==3 and {c['condition'] for c in rows}==set(a['conditions'])
        random.Random('voice-review-v1:'+ident).shuffle(rows);players=[]
        for label,row in zip('ABC',rows):
            p=root/row['condition']/ident/'joined-emitted.wav'
            assert hashlib.sha256(p.read_bytes()).hexdigest()==row['joined_wav_sha256']
            name=ident+'-'+label+'.wav';shutil.copy2(p,media/name)
            key.append(dict(id=ident,label=label,condition=row['condition'],source_sha256=row['joined_wav_sha256']))
            players.append('<div class="clip"><strong>'+label+'</strong><audio controls preload="none" src="media/'+name+'"></audio><label>Naturalness <select data-case="'+ident+'" data-label="'+label+'" data-metric="naturalness"><option value="">Rate</option>'+''.join(f'<option>{i}</option>' for i in range(1,6))+'</select></label><label>Clarity <select data-case="'+ident+'" data-label="'+label+'" data-metric="clarity"><option value="">Rate</option>'+''.join(f'<option>{i}</option>' for i in range(1,6))+'</select></label></div>')
        sections.append('<section><h2>'+html.escape(questions[ident])+'</h2>'+''.join(players)+'</section>')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Native voice comparison</title><style>
body{font:17px/1.55 system-ui,sans-serif;background:#f5f5f1;color:#192624;margin:0}main{max-width:960px;margin:48px auto;padding:0 24px}h1{font-size:34px;line-height:1.2}h2{font-size:21px}p{max-width:760px}section{background:white;border:1px solid #d9dfda;border-radius:16px;padding:22px;margin:24px 0}.clip{display:flex;align-items:center;gap:16px;flex-wrap:wrap;padding:14px 0;border-top:1px solid #eee}audio{max-width:100%;width:340px}select,button{font:inherit;padding:8px;border:1px solid #a3b4ac;border-radius:8px;background:white}button{background:#245f4d;color:white;cursor:pointer}.badge{color:#6b4d1d;background:#fff1cf;padding:6px 12px;border-radius:20px;font-size:14px}.muted{color:#53635e}
</style><main><span class="badge">Recorded experiment · not an adopted model</span><h1>Which voice sounds most natural?</h1><p>Each question has three native recordings, with the labels shuffled separately for each question. Listen before rating. All eight questions and all three conditions are included. Audio levels and words have not been edited.</p><p class="muted">Naturalness:1=very artificial,3=plausible but noticeably synthetic,5=like a person speaking naturally. Clarity:1=hard to understand,5=effortless to understand. Ratings stay in this browser until you export them.</p><button id="export">Save listening ratings</button>'''+''.join(sections)+'''<p class="muted">These are synthetic recorded tests, not a live microphone session. Word accuracy, correct answers, silence and cancellation are assessed separately.</p></main><script>
const storageKey='duplex-native-voice-review-v1';let ratings={};try{ratings=JSON.parse(localStorage.getItem(storageKey)||'{}')}catch(e){}
for(const el of document.querySelectorAll('select')){const k=[el.dataset.case,el.dataset.label,el.dataset.metric].join('|');el.value=ratings[k]||'';el.addEventListener('change',()=>{ratings[k]=el.value;try{localStorage.setItem(storageKey,JSON.stringify(ratings))}catch(e){}})}
document.getElementById('export').onclick=()=>{const payload={schema:'dl-blind-native-voice-ratings-v1',at:new Date().toISOString(),ratings,qualified:false};const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='native-voice-listening-ratings.json';a.click();URL.revokeObjectURL(url)};
</script></html>'''
    (out/'index.html').write_text(page)
    (out.parent/(out.name+'-condition-key.json')).write_text(json.dumps(dict(source_report_sha256=hashlib.sha256((root/'voice-comparison.json').read_bytes()).hexdigest(),conditions=key),indent=2)+'\n')
    return dict(questions=8,recordings=24,audio_unchanged=True,qualified=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--plan',required=True);a=p.parse_args();print(json.dumps(build(a.root,a.output,a.plan)))
