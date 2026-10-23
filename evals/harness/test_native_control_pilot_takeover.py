import copy
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from portable.context.native_control_development_wait import STAGES
from portable.context.native_control_pilot_takeover import original_gate, gradient_gate, guard_gate
from portable.context.native_control_training_followup import digest


class PilotTakeoverGates(unittest.TestCase):
    def original(self):
        commands = {n:['python','-m',n] for n in STAGES}
        stages = [dict(name=n, command=commands[n], pid=10+i, birth=20+i) for i,n in enumerate(STAGES[:3])]
        for s in stages[:2]:s['exit_code']=0
        return dict(schema='dl-native-control40-precise-pilot-controller-v1',pid=1, source_commit='source', pod_id='pod', phase=STAGES[2], passed=False,
                    optimizer_update_limit=8, stages=stages, planned_commands=commands)

    def test_original_requires_passed_predecessors_exact_live_child_and_complete_commands(self):
        c=self.original();self.assertIs(original_gate(c,1,12,'source','pod'),c)
        for field,value in [('pid',2),('pod_id','other'),('source_commit','other'),
                            ('optimizer_update_limit',9),('phase','bounded8_training')]:
            bad=copy.deepcopy(c);bad[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):original_gate(bad,1,12,'source','pod')
        for mutate in (lambda x:x['stages'][0].update(exit_code=1),
                       lambda x:x['stages'][2].update(exit_code=0),
                       lambda x:x['stages'][2].update(pid=13),
                       lambda x:x['planned_commands'].pop('independent_restore'),
                       lambda x:x['stages'][2].update(command=['changed'])):
            bad=copy.deepcopy(c);mutate(bad)
            with self.assertRaises(ValueError):original_gate(bad,1,12,'source','pod')

    def test_later_child_preserves_all_completed_stages_and_never_replays_an_optimizer(self):
        for index in range(3,7):
            c=self.original();c.update(schema='dl-native-control40-pilot-supervisor-takeover-v1',
                                      old_supervisor_retired_after_child_exit=True,phase=STAGES[index])
            c['stages'][-1]['exit_code']=0
            for i in range(3,index+1):
                r=dict(name=STAGES[i],command=c['planned_commands'][STAGES[i]],pid=10+i,birth=20+i)
                if i<index:r['exit_code']=0
                c['stages'].append(r)
            self.assertIs(original_gate(c,1,10+index,'source','pod'),c)
            self.assertEqual(list(c['planned_commands'])[index+1:],list(STAGES[index+1:]))
            bad=copy.deepcopy(c);bad['stages'][index-1]['exit_code']=1
            with self.assertRaises(ValueError):original_gate(bad,1,10+index,'source','pod')
        nested=self.original();nested['schema']='dl-native-control40-pilot-supervisor-takeover-v1'
        nested['old_supervisor_retired_after_child_exit']=True
        with self.assertRaises(ValueError):original_gate(nested,1,12,'source','pod')

    def test_guard_requires_fresh_same_pod_and_longer_deadline(self):
        now=datetime.now(timezone.utc);deadline=(now+timedelta(hours=1)).timestamp()
        g=dict(phase='armed',pod_id='pod',at_utc=now.isoformat(),
               deadline_utc=(now+timedelta(hours=2)).isoformat())
        guard_gate(g,'pod',deadline)
        for field,value in [('pod_id','other'),('phase','stopped'),
                            ('at_utc',(now-timedelta(minutes=2)).isoformat()),
                            ('deadline_utc',(now+timedelta(minutes=61)).isoformat())]:
            bad=dict(g);bad[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):guard_gate(bad,'pod',deadline)

    def test_full_forty_gradient_cannot_be_approved_from_subset_or_changed_ownership(self):
        with tempfile.TemporaryDirectory() as t:
            engine=Path(t)/'engine';numerical=Path(t)/'numerical'
            engine.mkdir();numerical.mkdir()
            ids=['case-'+str(i) for i in range(40)]
            (engine/'engine-inputs.json').write_text(json.dumps(dict(cases=[dict(id=i) for i in ids])))
            (numerical/'numerical.json').write_text('passed numerical evidence')
            gradient=dict(all26_gradients_present_and_finite=True,frozen_parameter_gradients=0)
            combined=dict(gradient,acoustic_text_head_gradient_delta_exactly_zero=True,
                          acoustic_temporal_gradient_delta_norm=.2)
            g=dict(schema='dl-native-control40-gradient-readiness-v1',source_commit='source',passed=True,
                   complete40=True,all_parameters_and_execution_unchanged=True,optimizer_updates=0,
                   checkpoint_written=False,training_approved=False,behavior_qualified=False,
                   original154_and_old49_readiness_still_required=True,trainable_parameters=1894400,
                   peak_allocated_bytes=20_000_000_000,engine_report_sha256=digest(engine/'engine-inputs.json'),
                   numerical_report_sha256=digest(numerical/'numerical.json'),
                   cases=[dict(id=i,passed=True,exact_native_streamed_text_outputs=True,
                               exact_native_acoustic_outputs=True,text_gradient=dict(gradient),
                               combined_gradient=dict(combined)) for i in ids])
            self.assertIs(gradient_gate(g,engine,numerical,'source'),g)
            for mutate in (lambda x:x['cases'].pop(),lambda x:x.update(passed=False),
                           lambda x:x.update(optimizer_updates=1),lambda x:x.update(checkpoint_written=True),
                           lambda x:x.update(all_parameters_and_execution_unchanged=False),
                           lambda x:x.update(peak_allocated_bytes=80_000_000_000),
                           lambda x:x['cases'][0]['combined_gradient'].update(frozen_parameter_gradients=1),
                           lambda x:x['cases'][0]['combined_gradient'].update(acoustic_temporal_gradient_delta_norm=float('nan')),
                           lambda x:x['cases'][0].update(exact_native_acoustic_outputs=False),
                           lambda x:x['cases'][0].update(id='wrong')):
                bad=copy.deepcopy(g);mutate(bad)
                with self.assertRaises(ValueError):gradient_gate(bad,engine,numerical,'source')
            (engine/'engine-inputs.json').write_text('changed')
            with self.assertRaises(ValueError):gradient_gate(g,engine,numerical,'source')


if __name__=='__main__':unittest.main()
