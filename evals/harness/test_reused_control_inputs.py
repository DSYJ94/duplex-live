"""Local lineage/PCM fixture; no external generation or recognition claim."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import soundfile as sf
from evals.duplex_fresh_inputs import generate,specifications


class ReusedControlInputs(unittest.TestCase):
 def test_exact_approved_clips_are_copied_and_only_new_control_is_synthesized(self):
  from data.synthetic.recovery_curriculum import MODEL
  with tempfile.TemporaryDirectory() as d:
   runs=Path(d)/'runs';runs.mkdir();scene=next(s for s in specifications() if s['id']=='hybrid-fresh-046');rows={};expected=[]
   for i,job,quality,case in [(0,'7515617','7515618','hybrid-fresh-026'),(1,'7517443','7517444','hybrid-fresh-031')]:
    folder=runs/('duplex-fresh-inputs-'+job)/case;folder.mkdir(parents=True);pcm=np.arange(1920*(i+1),dtype=np.float32)/100000;path=folder/'user-0.wav';sf.write(path,pcm,24000,subtype='FLOAT');expected.append(pcm)
    ref=folder/'reference.json';ref.write_text(json.dumps(dict(utterances=[dict(source_file=path.name,text=scene['turns'][i][1])])));(folder.parent/'inputs.json').write_text('{}')
    q=runs/('duplex-fresh-quality-'+quality);q.mkdir();(q/'quality.json').write_text('{}')
    receipt=dict(voice='coral',source_model=MODEL,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),file=path.name)
    rows[case]=dict(path=str(folder/'conversation.wav'),reference_path=str(ref),generation_receipts=[receipt])
   called=[]
   def synthesize(text,voice,path):
    called.append(text);sf.write(path,np.ones(1920,dtype=np.float32)*.01,24000,subtype='FLOAT')
    return dict(voice=voice,source_model=MODEL,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
   def cases(data,quality,ids,source,commit):return [rows[ids[0]]],dict(fixture_only=True,commit=commit)
   out=runs/'new-evaluation'
   with patch('evals.duplex_input_validator.cases',cases),patch('data.synthetic.recovery_curriculum.synthesize',synthesize),patch('builtins.print'):
    generate(out,cases=['hybrid-fresh-046'],user_voice='coral')
   self.assertEqual(called,[scene['turns'][2][1]])
   manifest=json.loads((out/'inputs.json').read_text());self.assertFalse(manifest['training_approved']);self.assertTrue(manifest['do_not_train']);row=manifest['rows'][0]
   self.assertEqual(row['evaluation_kind'],'reused-approved-clips-new-cancellation-control')
   for i,pcm in enumerate(expected):
    actual,rate=sf.read(out/scene['id']/('user-'+str(i)+'.wav'),dtype='float32');self.assertEqual(rate,24000);np.testing.assert_array_equal(actual,pcm)
    self.assertTrue(row['generation_receipts'][i]['reused_evaluation_clip'])
   # No fallback regeneration when a purported approved source no longer matches.
   rows['hybrid-fresh-026']['generation_receipts'][0]['voice']='wrong'
   with patch('evals.duplex_input_validator.cases',cases),patch('data.synthetic.recovery_curriculum.synthesize',synthesize),patch('builtins.print'),self.assertRaises(RuntimeError):
    generate(runs/'rejected',cases=['hybrid-fresh-046'],user_voice='coral')
   self.assertEqual(len(called),1)
