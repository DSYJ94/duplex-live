import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from data.venus import render_context_assistant as r

class AssistantAdmissionTests(unittest.TestCase):
 def test_whole_exact34_admitted_under_carried_caps(self):
  p=r.admission();self.assertEqual(len(p['requests']),34);self.assertEqual(p['conservative_cost_upper_usd'],2.2245)
 def test_payload_change_cannot_bypass_exact_admission(self):
  with tempfile.TemporaryDirectory() as directory:
   path=Path(directory)/'payload';p=json.loads(r.PAYLOAD.read_text());p['requests'][0]['body']['text']='changed';path.write_text(json.dumps(p))
   with patch.object(r,'PAYLOAD',path):
    with self.assertRaises(ValueError):r.admission()
