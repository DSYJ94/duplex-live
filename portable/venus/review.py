"""Build an offline review of all ten original native recordings; never qualifies."""
import argparse,array,hashlib,html,json,math,shutil,struct,sys
from pathlib import Path

def read_float_wav(path):
 raw=path.read_bytes();assert raw[:4]==b'RIFF' and raw[8:12]==b'WAVE';pos=12;fmt=None;pcm=None
 while pos+8<=len(raw):
  kind=raw[pos:pos+4];size=struct.unpack_from('<I',raw,pos+4)[0];data=raw[pos+8:pos+8+size];pos+=8+size+(size%2)
  if kind==b'fmt ':fmt=struct.unpack_from('<HHIIHH',data)
  elif kind==b'data':pcm=data
 assert fmt and fmt[0]==3 and fmt[1]==1 and fmt[2]==24000 and fmt[5]==32 and pcm is not None
 samples=array.array('f');samples.frombytes(pcm)
 if sys.byteorder!='little':samples.byteswap()
 assert all(math.isfinite(x) for x in samples)
 return pcm,samples

def write_float_wav(path,pcm):
 fmt=struct.pack('<HHIIHH',3,1,24000,96000,4,32)
 path.write_bytes(b'RIFF'+struct.pack('<I',36+len(pcm))+b'WAVEfmt '+struct.pack('<I',16)+fmt+b'data'+struct.pack('<I',len(pcm))+pcm)

def main():
 p=argparse.ArgumentParser();p.add_argument('--recordings',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 root=Path(a.recordings).resolve();manifest=Path(a.manifest).resolve();plan=json.loads(manifest.read_text());out=Path(a.output).resolve();out.mkdir(exist_ok=False)
 results=json.loads((root/'results.json').read_text());assert results['all10_complete'] and len(results['cases'])==10
 metrics=[];cards=[]
 for case in plan['cases']:
  ident=case['id'];rows=[json.loads(x) for x in (root/ident/'trace.jsonl').read_text().splitlines()];dest=out/ident;dest.mkdir();parts=[];peak=0.;clipped=0;samplecount=0;blocks=[];elapsed=[]
  for row in rows:
   pcm,samples=read_float_wav(root/ident/row['chunk_file']);parts.append(pcm);peak=max(peak,max(map(abs,samples),default=0));clipped+=sum(abs(x)>=.995 for x in samples);samplecount+=len(samples);elapsed.append(row['elapsed'])
   blocks.extend(max(map(abs,samples[i:i+480]),default=0) for i in range(0,len(samples),480))
  write_float_wav(dest/'native-joined.wav',b''.join(parts));shutil.copy2(root/ident/'trace.jsonl',dest/'trace.jsonl')
  turnstarts={t:min(x['sequence'] for x in rows if x['input_turn']==t) for t in sorted(set(x['input_turn'] for x in rows))}
  worker=[e for row in rows for e in row['events']];text=''.join(x['text'] for x in rows)
  m=dict(case=ident,chunks=len(rows),native_generated_text=text,peak=peak,samples_at_or_above_0995=clipped,sample_count=samplecount,native_audio_seconds=samplecount/24000,compute_median_seconds=sorted(elapsed)[len(elapsed)//2],compute_p95_seconds=sorted(elapsed)[math.ceil(.95*len(elapsed))-1],turn_start_sequences=turnstarts,worker_events=worker,qualified=False,independent_audio_review_pending=True)
  if ident.startswith('quiet-'):
   cue=turnstarts[2];m['later_release_cue_sequence']=cue;m['nonzero_audio_before_release_sequences']=[x['sequence'] for x in rows if x['sequence']<cue and x['peak']>.001]
  metrics.append(m)
  inputs=[]
  for i,turn in enumerate(case['turns']):
   declared=Path(turn['path']);src=(manifest.parent/declared.relative_to('/inputs')).resolve() if declared.is_absolute() else Path(turn['path']).resolve()
   assert hashlib.sha256(src.read_bytes()).hexdigest()==turn['sha256'];shutil.copy2(src,dest/f'input-{i}.wav')
   inputs.append(f'<p>Input{i+1}: {html.escape(turn.get("authored_text") or "Original independently audited ordinary question")}<br><audio controls preload="none" src="{ident}/input-{i}.wav"></audio></p>')
  points=' '.join(f'{i/max(1,len(blocks)-1)*900:.2f},{90-min(1,b)*75:.2f}' for i,b in enumerate(blocks))
  # Amplitude is displayed at its original scale; no gain/threshold alteration.
  svg=f'<svg viewBox="0 0 900 100" aria-label="Native waveform absolute peak per20ms, original scale"><path d="M0 90 H900" stroke="#888"/><polyline points="{points}" fill="none" stroke="#005a9c" stroke-width="1"/></svg>'
  cards.append(f'<section><h2>{html.escape(ident)}</h2>'+''.join(inputs)+f'<p>Native output, original float chunks joined in order; this playback is not a wall-clock latency proof.<br><audio controls preload="none" src="{ident}/native-joined.wav"></audio></p>'+svg+f'<p>Peak {peak:.4f}; samples ≥0.995: {clipped}; compute p95 {m["compute_p95_seconds"]:.3f}s per input second.</p><p>Generated text (not independent ASR): {html.escape(text)}</p><p>Worker events: {len(worker)}. <a href="{ident}/trace.jsonl">Original trace</a></p></section>')
 report=dict(all10_collected=True,qualified=False,recording_root=str(root),cases=metrics,notes=['No normalization/clipping/gain modification','New14 input word audits and independent output listening remain pending','Joined playback does not prove actual delivery timing','Diagnostic arithmetic backend only; no external reasoning-model calls'])
 (out/'review.json').write_text(json.dumps(report,indent=2)+'\n')
 (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Venus native10 review</title><style>body{font:16px system-ui;max-width:1050px;margin:30px auto;padding:0 20px;color:#17212d}section{border-top:1px solid #aaa;padding:18px 0}audio{width:100%;max-width:650px}svg{width:100%;background:#f3f6f9}p{line-height:1.5}</style><h1>Venus native10 — recorded evidence</h1><p>All10 cases collected. Candidate UNQUALIFIED. Original native audio, no keyword mute or gain changes. Outputs and new synthetic inputs still require independent listening. Arithmetic test worker only; this is not a live microphone demo.</p><a href="review.json">Full measured review</a>'+''.join(cards))
 print(json.dumps(dict(review=str(out/'index.html'),all10_collected=True,qualified=False,global_peak=max(x['peak'] for x in metrics),clipped_samples=sum(x['samples_at_or_above_0995'] for x in metrics))))
if __name__=='__main__':main()
