import copy
import unittest
from evals.native_v2_teacher_repair import FAILED, check_lineage, CLARITY_FAILED, check_clarity_lineage


class TeacherRepairLineage(unittest.TestCase):
    def fixture(self):
        records=[dict(dir=f'native2-train-{i:03d}', source_row=dict(split='train', wav='unchanged'),
                      audio_sha256=str(i)) for i in range(24)]
        parent=dict(recordings=records, teacher_protocol='native-dialogue-v4',
                    source_ingestion_sha256='source', source_conversion='original')
        checks=dict(results=[dict(scenario_id=r['dir'], speech_and_fact_checks_pass=r['dir'] not in FAILED)
                             for r in records])
        return parent, checks

    def test_all24_retained_only_failed_raw_recordings_may_change(self):
        parent, checks=self.fixture();fresh=copy.deepcopy(parent)
        for r in fresh['recordings']:
            if r['dir'] in FAILED:r['audio_sha256']='new'
        check_lineage(fresh,parent,checks)

    def test_passing_recording_source_edit_or_subset_is_rejected(self):
        parent, checks=self.fixture()
        for mutation in ('passing','source','subset','different_diagnosis'):
            fresh=copy.deepcopy(parent);decisions=copy.deepcopy(checks)
            if mutation=='passing':fresh['recordings'][0]['audio_sha256']='changed'
            elif mutation=='source':fresh['recordings'][1]['source_row']['wav']='changed'
            elif mutation=='subset':fresh['recordings'].pop()
            else:decisions['results'][1]['speech_and_fact_checks_pass']=True
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                check_lineage(fresh,parent,decisions)

    def test_clarity_protocol_is_explicit_only_for_five_actual_failures(self):
        parent, checks=self.fixture()
        for row in checks['results']:
            row['speech_and_fact_checks_pass']=row['scenario_id'] not in CLARITY_FAILED
        fresh=copy.deepcopy(parent)
        fresh['record_protocols']={sid:'native-dialogue-v5' for sid in CLARITY_FAILED}
        for row in fresh['recordings']:
            if row['dir'] in CLARITY_FAILED:
                row.update(audio_sha256='new',teacher_protocol='native-dialogue-v5')
        check_clarity_lineage(fresh,parent,checks)
        for change in ('passing','undeclared','source','subset'):
            bad=copy.deepcopy(fresh)
            if change=='passing':bad['recordings'][0]['audio_sha256']='changed'
            elif change=='undeclared':bad['record_protocols']['native2-train-000']='native-dialogue-v5'
            elif change=='source':bad['recordings'][1]['source_row']['wav']='changed'
            else:bad['recordings'].pop()
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                check_clarity_lineage(bad,parent,checks)


if __name__=='__main__':unittest.main()
