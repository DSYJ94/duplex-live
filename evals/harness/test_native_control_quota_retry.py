import copy
from pathlib import Path
import unittest

from portable.context.native_control_development_wait import STAGES
from portable.context.native_control_quota_retry import retry_commands, COMMIT, POD


def previous():
    commands={n:['python','-u','-m',n]for n in STAGES}
    for n in STAGES[4:6]:
        commands[n]+=['--out','/old/training','--native-audio49-preflight','/old/ready',
                      '--actual-fit-updates','8','--warm-start-proof','/parent520','--weight','64']
    commands[STAGES[6]]=['python','-u','-c',"verify(Path('/old/training'))"]
    stages=[dict(name=n,command=commands[n],exit_code=0)for n in STAGES[:4]]
    stages.append(dict(name=STAGES[4],command=commands[STAGES[4]],pid=97250,birth=25672616))
    return dict(pid=79780,birth=24607807,source_commit=COMMIT,pod_id=POD,
                phase='bounded8_training',passed=False,optimizer_update_limit=8,
                stages=stages,planned_commands=commands)


class QuotaRetry(unittest.TestCase):
    def test_reuses_all_four_passed_stages_and_changes_only_three_output_paths(self):
        p=previous();before=copy.deepcopy(p);c=retry_commands(p,Path('/new'))
        self.assertEqual(p,before)
        for name in STAGES[:4]:self.assertEqual(c[name],p['planned_commands'][name])
        for name in STAGES[4:6]:
            old=p['planned_commands'][name]
            self.assertEqual([(a,b)for a,b in zip(old,c[name])if a!=b],[('/old/training','/new/training')])
        self.assertEqual(c[STAGES[6]][-1],"verify(Path('/new/training'))")

    def test_rejects_failed_readiness_changed_actual_command_extension_and_partial_resume(self):
        for change in ('readiness','command','updates','resume','identity','finished'):
            p=previous()
            if change=='readiness':p['stages'][3]['exit_code']=1
            elif change=='command':p['stages'][4]['command']=[]
            elif change=='updates':
                command=p['planned_commands'][STAGES[4]]
                command[command.index('--actual-fit-updates')+1]='32'
            elif change=='resume':p['planned_commands'][STAGES[4]]+=['--resume-training','/partial']
            elif change=='identity':p['pid']=1
            else:p['stages'][4]['exit_code']=0
            with self.subTest(change=change),self.assertRaises(ValueError):retry_commands(p,Path('/new'))


if __name__=='__main__':unittest.main()
