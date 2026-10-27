"""Mac system TTS only; finite authored inputs, no external speech API or CPU caps."""
import argparse,array,concurrent.futures,hashlib,json,subprocess,wave
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True);p.add_argument('--voice',default='Samantha');p.add_argument('--rate',type=int,default=175);a=p.parse_args()
 assert 80<=a.rate<=220,'Pin an ordinary source speaking rate; no gain manipulation'
 src=Path(a.plan);plan=json.loads(src.read_text());root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
 texts=sorted({t for s in plan['scenes'] for t in s['user_turns']});receipts=[]
 def render(text):
  ident=hashlib.sha256(text.encode()).hexdigest();tmp=root/(ident+'.aiff');out=root/(ident+'.wav')
  subprocess.run(['say','-v',a.voice,'-r',str(a.rate),'-o',str(tmp),text],check=True,capture_output=True,timeout=60)
  subprocess.run(['/usr/bin/afconvert','-f','WAVE','-d','LEI16@16000','-c','1',str(tmp),str(out)],check=True,capture_output=True,timeout=30)
  with wave.open(str(out)) as w:assert (w.getframerate(),w.getnchannels(),w.getsampwidth())==(16000,1,2);duration=w.getnframes()/16000;samples=array.array('h',w.readframes(w.getnframes()))
  assert samples and duration>0,'Empty source audio is a failed render, not usable silence'
  peak=max(abs(x) for x in samples)/32768
  assert peak<=.995,'Clipping requires source diagnosis; gain changes forbidden'
  # AIFF source retained; no gain/normalization/clipping repair.
  return dict(id=ident,text=text,wav=out.name,original_aiff=tmp.name,seconds=duration,sha256=hashlib.sha256(out.read_bytes()).hexdigest(),aiff_sha256=hashlib.sha256(tmp.read_bytes()).hexdigest(),peak=peak,voice='macOS '+a.voice,rate=a.rate,independent_word_audit='PENDING')
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  for receipt in pool.map(render,texts):
   receipts.append(receipt);(root/'render-receipts.json').write_text(json.dumps(dict(plan_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),complete=len(receipts)==len(texts),utterances=receipts),indent=2)+'\n')
 print(json.dumps(dict(unique_utterances=len(receipts),total_unique_audio_seconds=sum(x['seconds'] for x in receipts),status='PENDING_INDEPENDENT_AUDIT',external_api_calls=0)))
if __name__=='__main__':main()
