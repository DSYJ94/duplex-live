import copy
from pathlib import Path
import unittest
from portable.context.native_control_development_wait import STAGES
from portable.context.native_control_pilot_recovery import recovered_commands


def previous():
    commands={n:['python','-m',n]for n in STAGES}
    commands[STAGES[3]]+=['--out','/old/ready','--actual-fit-updates','8','--weight','64']
    for n in STAGES[4:6]:commands[n]+=['--out','/old/train','--native-audio49-preflight','/old/ready','--actual-fit-updates','8']
    commands[STAGES[6]]=['python','-c',"verify(Path('/old/train'))"]
    stages=[dict(name=n,command=commands[n],exit_code=0)for n in STAGES[:3]]
    stages.append(dict(name=STAGES[3],command=commands[STAGES[3]]))
    return dict(source_commit='c9efaaac24d983342f7eded46617e6450bc102ce',pod_id='4yx8vahdmjsylx',
        phase='failed',error_type='TimeoutError',error='Extended adopted-stage deadline reached',
        optimizer_update_limit=8,passed=False,stages=stages,planned_commands=commands)


class Recovery(unittest.TestCase):
    def test_only_output_paths_change_all_seven_and_old_actual_exits_retained(self):
        p=previous();before=copy.deepcopy(p);c=recovered_commands(p,Path('/new'))
        self.assertEqual(p,before);self.assertEqual(list(c),list(STAGES))
        self.assertEqual(c[STAGES[2]],p['planned_commands'][STAGES[2]])
        self.assertEqual(c[STAGES[3]][-4:],['--actual-fit-updates','8','--weight','64'])
        self.assertIn('/new/combined-readiness',c[STAGES[4]])
        self.assertIn("'/new/training'",c[STAGES[6]][-1])
    def test_rejects_failed_predecessor_and_non_deadline_model_failure(self):
        for change in ('predecessor','reason','source','updates'):
            p=previous()
            if change=='predecessor':p['stages'][2]['exit_code']=1
            elif change=='reason':p['error']='nonfinite gradient'
            elif change=='source':p['source_commit']='other'
            else:p['optimizer_update_limit']=32
            with self.assertRaises(ValueError):recovered_commands(p,Path('/new'))
    def test_rejects_changed_recorded_actual_command(self):
        p=previous();p['stages'][1]['command']=['replacement']
        with self.assertRaises(ValueError):recovered_commands(p,Path('/new'))


if __name__=='__main__':unittest.main()
