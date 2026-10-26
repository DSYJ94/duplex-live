import copy
import hashlib
import json
from pathlib import Path
import unittest
from data.venus.build_control_corpus import build
from data.venus.asr import wer
from training.venus.guard import AUTH,validate,stamp


class DataTests(unittest.TestCase):
    def test_all_folds_and_counterfactual_inputs(self):
        p=build();assert len(p['scenes'])==224
        grouped={}
        for s in p['scenes']:
            if s['family']=='delegate':grouped.setdefault(s['split_group'],[]).append(s)
        for rows in grouped.values():
            assert len(rows)==4
            assert len({tuple(s['user_turns']) for s in rows})==1
            assert len({s['teacher_program']['result_answer'] for s in rows})==4
            assert len({s['teacher_program']['worker_delay_seconds'] for s in rows})==1

    def test_word_audit_never_equates_cues_or_homophones(self):
        assert wer('What colour is it?','what color is it')==0
        assert wer('Room sign','Rune sign')==.5
        assert wer('Marigold','Marble')==1
        assert wer('Go.','Go. Go.')==1
        assert wer('Stay silent','')==1

    def test_repair_orthography_never_accepts_lexical_errors(self):
        from data.venus.repair import wer as repaired_wer
        assert repaired_wer('No acknowledgment now.','No acknowledgement now.')==0
        assert repaired_wer('What is two plus four?','What is2 plus4?')>0 # missing spaces still differ
        assert repaired_wer('What is two plus four?','What is 2 plus 4?')==0
        assert repaired_wer('marker in bottle','marker and bottle')==1/3
        assert repaired_wer('granite','granted')==1

    def test_training_guard_preserves_budget_and_one_gpu(self):
        raw=AUTH.read_bytes();a=json.loads(raw);start=stamp('2026-10-26T20:00:00Z')
        lease=dict(resource_key='venus-training-v1',vm_id='123',disk_id='456',zone='us-central1-a',disk_size_gib=100,creation_timestamp='2026-10-26T19:59:00Z',stop_unix=start+7200,prior_new_gpu_seconds_upper=4600,prior_gpu_seconds=40351,prior_scope_usd=a['prior_original_scope_cost_upper_usd'],authorization_sha256=hashlib.sha256(raw).hexdigest())
        vm=dict(id='123',name='duplex-venus-train-20261026-v1',zone='/us-central1-a',status='RUNNING',machineType='/a2-highgpu-1g',creationTimestamp=lease['creation_timestamp'],lastStartTimestamp='2026-10-26T20:00:00Z',tags=dict(items=['duplex-venus-train-bounded']),disks=[dict(boot=True,autoDelete=False,source='/disks/duplex-venus-train-20261026-v1')],scheduling=dict(provisioningModel='FLEX_START',maxRunDuration=dict(seconds=7200),instanceTerminationAction='STOP',automaticRestart=False))
        assert validate(vm,[vm],[],lease,start+10)==start+7200
        for bad in ({'stop_unix':start+9000},{'prior_new_gpu_seconds_upper':0},{'disk_size_gib':250},{'authorization_sha256':'bad'}):
            with self.assertRaises((AssertionError,ValueError)):validate(vm,[vm],[],dict(lease,**bad),start+10)
        with self.assertRaises(AssertionError):validate(vm,[vm],[dict(status='RUNNING')],lease,start+10)
        with self.assertRaises(AssertionError):validate(vm,[vm,dict(vm,id='999')],[],lease,start+10)

    def test_same_vm_restart_carries_closed_interval_and_shorter_stop(self):
        from training.venus.restart_guard import AUTH as authority,validate as check
        raw=authority.read_bytes();a=json.loads(raw);r=a['existing_resource'];start=stamp('2026-10-26T21:00:00Z')
        lease=dict(resource_key='venus-native-gradient-v8',vm_id=r['vm_id'],disk_id=r['disk_id'],zone=r['zone'],disk_size_gib=100,creation_timestamp='2026-10-26T20:02:00Z',stop_unix=start+7200,prior_new_gpu_seconds_upper=6400,prior_gpu_seconds=40351,prior_scope_usd=a['prior_original_scope_cost_upper_usd'],authorization_sha256=hashlib.sha256(raw).hexdigest(),hourly_admission_ceiling_usd=4,verified_hourly_price_usd=3.68708363)
        vm=dict(id=r['vm_id'],name=r['name'],zone='/'+r['zone'],status='RUNNING',machineType='/'+r['machine'],creationTimestamp=lease['creation_timestamp'],lastStartTimestamp='2026-10-26T21:00:00Z',tags=dict(items=[r['tag']]),disks=[dict(boot=True,autoDelete=False,source='/disks/'+r['disk_name'])],scheduling=dict(provisioningModel='FLEX_START',maxRunDuration=dict(seconds=7200),instanceTerminationAction='STOP',automaticRestart=False))
        assert check(vm,[vm],[],lease,start+10)==start+7200
        for bad in ({'prior_new_gpu_seconds_upper':4600},{'stop_unix':start+9000},{'verified_hourly_price_usd':4.01},{'vm_id':'123'}):
            with self.assertRaises(AssertionError):check(vm,[vm],[],dict(lease,**bad),start+10)
