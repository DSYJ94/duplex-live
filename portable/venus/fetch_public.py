"""Fetch only the35 approved public files at a pinned upstream revision."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,subprocess
from pathlib import Path
from urllib.parse import quote
REVISION='4681004131ea1f1fb9aca9f0db76be322657cc79'
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 rows=json.loads(Path(a.manifest).read_text());root=Path(a.output).resolve()
 assert len(rows)==35 and sum(x['bytes'] for x in rows)==19993772218
 def fetch(row):
  path=(root/row['path']).resolve();assert root in path.parents
  if path.exists() and path.stat().st_size==row['bytes'] and digest(path)==row['sha256']:return
  path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.public-download-part')
  url='https://huggingface.co/inclusionAI/Realtime-Venus/resolve/'+REVISION+'/'+quote(row['path'],safe='/')
  subprocess.run(['curl','--silent','--show-error','--fail','--location','--retry','2','--connect-timeout','15','--max-time','600','--output',str(temp),url],check=True,timeout=620)
  assert temp.stat().st_size==row['bytes'] and digest(temp)==row['sha256'],row['path']
  temp.replace(path);print(json.dumps(dict(file=row['path'],bytes=row['bytes'],sha256_verified=True)),flush=True)
 with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(fetch,rows))
 print(json.dumps(dict(all_hashes_verified=True,files=len(rows),bytes=sum(x['bytes'] for x in rows),revision=REVISION)),flush=True)
if __name__=='__main__':main()
