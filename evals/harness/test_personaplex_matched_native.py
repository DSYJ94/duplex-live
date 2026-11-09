import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from training.personaplex import matched_native as m


class PairedAdmission(unittest.TestCase):
    def packet(self,root):
        source='a'*40;data='b'*64;entries=[]
        for base in m.BASES:
            for capacity in m.CAPACITIES:
                stem=base+'-'+capacity
                admission=dict(schema='dl-moshi-matched-admission-v1',source=source,actual_CUDA=True,
                    passed=True,base=base,capacity=capacity,data_report_sha256=data,
                    trainable_parameters=m.CAPACITIES[capacity],zero_B_native_outputs_exact=True,
                    original_all323_input_rows=True,all_four_acoustic_gradients=[dict(kind=k,
                        all_trainable_gradients_present_finite=True) for k in ('private','simple','clarify','cancel')],
                    new_four_worst_depth_checks=[dict(depth_parity=dict(passed=True)) for _ in range(4)],
                    complete_worst_update=[dict(group=g) for g in m.SIZES],
                    materialized_Adam=dict(optimizer_updates=0),optimizer_updates=0,
                    checkpoint_sha256=stem+'checkpoint',execution_id=stem+'-admit',
                    estimated_complete196_seconds=3600.)
                restored=dict(source=source,actual_CUDA=True,passed=True,
                    checkpoint_sha256=admission['checkpoint_sha256'],execution_id=stem+'-restore',
                    reference_execution_id=admission['execution_id'],restored_training_step=0,
                    all_Adam_RNG_OPAL_fields_exact=True,native_logits_exact=True)
                row=dict(base=base,capacity=capacity)
                for kind,value in [('admission',admission),('restore',restored)]:
                    path=root/(stem+'-'+kind+'.json');path.write_text(json.dumps(value))
                    terminal=root/(stem+'-'+kind+'-terminal.json');terminal.write_text(json.dumps(dict(source=source,
                        execution_id=value['execution_id'],actual_exit_code=0)))
                    row.update({kind:str(path),kind+'_sha256':m.sha(path),kind+'_terminal':str(terminal)})
                entries.append(row)
        path=root/'joint.json';path.write_text(json.dumps(dict(source=source,data_report_sha256=data,entries=entries)))
        args=SimpleNamespace(source=source,inputs_sha=data,joint_admission=str(path),joint_admission_sha=m.sha(path),lease='fixture')
        return args,entries,path

    def run_packet(self,args,now=1000):
        with patch.object(m.guard,'check',return_value=dict(prior_GPU_seconds=327,start_unix=1000)),patch.object(m.time,'time',return_value=now):
            return m.joint_schedule(args)

    def replace(self,args,path,value):
        path.write_text(json.dumps(value));args.joint_admission_sha=m.sha(path)

    def test_complete_four_fresh_lineages_fit_real_carried_budget(self):
        with tempfile.TemporaryDirectory() as root:
            args,entries,path=self.packet(Path(root));value=self.run_packet(args)
            self.assertEqual(value['estimated_serial_seconds_per_base'],{b:7200. for b in m.BASES})
            self.assertEqual(value['required_GPU_seconds'],2*7200+6*3600)
            self.assertEqual(value['available_GPU_seconds'],96*3600-327)

    def test_no_training_from_three_arms_or_a_nonterminal_restore(self):
        with tempfile.TemporaryDirectory() as root:
            args,entries,path=self.packet(Path(root));packet=json.loads(path.read_text())
            packet['entries']=entries[:-1];self.replace(args,path,packet)
            with self.assertRaisesRegex(ValueError,'All four'):self.run_packet(args)
            packet['entries']=entries;self.replace(args,path,packet)
            terminal=Path(entries[0]['restore_terminal']);value=json.loads(terminal.read_text());value['actual_exit_code']=1
            terminal.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'successful native'):self.run_packet(args)

    def test_admission_hash_and_own_fresh_process_binding_cannot_be_swapped(self):
        with tempfile.TemporaryDirectory() as root:
            args,entries,path=self.packet(Path(root));row=entries[0];restore=Path(row['restore'])
            value=json.loads(restore.read_text());value['checkpoint_sha256']='wrong';restore.write_text(json.dumps(value))
            with self.assertRaises(ValueError):self.run_packet(args)
            row['restore_sha256']=m.sha(restore);packet=json.loads(path.read_text());packet['entries']=entries
            self.replace(args,path,packet)
            with self.assertRaisesRegex(ValueError,'own exact independent'):self.run_packet(args)

    def test_insufficient_complete_schedule_time_stops_without_subset(self):
        with tempfile.TemporaryDirectory() as root:
            args,entries,path=self.packet(Path(root))
            with self.assertRaisesRegex(RuntimeError,'complete paired schedule'):self.run_packet(args,now=1000+48*3600)

    def test_worst_probe_selection_considers_actual_geometry_in_every_group(self):
        class Shape:
            def __init__(self,n):self.shape=(1,17,n)
        rows={g:[dict(id='short',inputs=Shape(5)),dict(id='long',codes=Shape(7))] for g in m.SIZES}
        self.assertEqual(m.worst_rows(rows),{g:1 for g in m.SIZES})
        self.assertEqual(m.RECIPE['updates'],196)
        self.assertEqual(m.RECIPE['reference_hidden_MSE'],10.)
        self.assertEqual(m.RECIPE['word_anchor'],64.)


if __name__=='__main__':unittest.main()
