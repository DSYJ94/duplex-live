"""Compare reconstruction with the actual saved approved states, without CUDA."""
import hashlib
import json
from pathlib import Path
import unittest
import torch
from portable.context.restore_adapter_state import PINS, metadata_from_report

ROOT=Path(__file__).resolve().parents[2]
ARCHIVE=ROOT/'runs/cloud-export-20261016/tree/runs'


class RestoreMetadata(unittest.TestCase):
    @unittest.skipUnless((ARCHIVE/'context-training-proof-7510331/training-state.pt').exists(),'Verified native archive needed')
    def test_both_independent_metadata_recipes_match_the_real_saved_state(self):
        for name,pin in PINS.items():
            with self.subTest(proof=name):
                folder=ARCHIVE/name;report=json.loads((folder/'report.json').read_text())
                for filename,digest in pin['hashes'].items():
                    self.assertEqual(hashlib.sha256((folder/filename).read_bytes()).hexdigest(),digest)
                saved=torch.load(folder/'training-state.pt',weights_only=True,map_location='cpu')
                self.assertEqual(metadata_from_report(ROOT,folder,report),saved['metadata'])
                self.assertEqual(saved['completed_steps'],pin['steps'])
                self.assertEqual(sorted({int(v['step']) for v in saved['optimizer']['state'].values()}),[pin['adam']])


if __name__=='__main__':unittest.main()
