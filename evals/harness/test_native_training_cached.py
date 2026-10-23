import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from portable.context.native_training_cached import CODE,PROFILE,verify_receipt


class CachedTrainingReceipt(unittest.TestCase):
    def test_failed_changed_or_incomplete_verification_never_qualifies_a_cached_stage(self):
        committed=b'committed fixture source'
        expected=hashlib.sha256(committed).hexdigest()
        value=dict(schema='dl-native-immutable-input-hash-reuse-v1',passed=True,stage_kind='training',
                   source_commit='builder',code_sha256={p:expected for p in CODE},
                   cache=dict(mode=PROFILE,unique_hashed_files=30,all_cached_identities_rechecked=True,
                              validation_results_or_model_outputs_cached=False))
        with tempfile.TemporaryDirectory() as directory, \
             patch('portable.context.native_training_cached.subprocess.check_output',return_value=committed):
            path=Path(directory)/'receipt.json';path.write_text(json.dumps(value))
            self.assertTrue(verify_receipt(path,'training','builder')['passed'])
            for mode in ('failed','kind','source','code','empty','unchecked','results_cached'):
                r=copy.deepcopy(value)
                if mode=='failed':r['passed']=False
                if mode=='kind':r['stage_kind']='readiness'
                if mode=='source':r['source_commit']='other'
                if mode=='code':r['code_sha256'][CODE[0]]='changed'
                if mode=='empty':r['cache']['unique_hashed_files']=0
                if mode=='unchecked':r['cache']['all_cached_identities_rechecked']=False
                if mode=='results_cached':r['cache']['validation_results_or_model_outputs_cached']=True
                path.write_text(json.dumps(r))
                with self.subTest(mode=mode),self.assertRaises(ValueError):verify_receipt(path,'training','builder')


if __name__=='__main__':unittest.main()
