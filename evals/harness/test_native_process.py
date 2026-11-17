import unittest
from training.personaplex.native_process import validate_restore

class NativeProcessTest(unittest.TestCase):
    def fixture(self):
        saved=dict(schema='dl-native-zero-ret-restore-v1',identity={'checkpoint':'pinned'},source='original-source',step=0,writer_pid=50,execution='original-execution')
        parent=dict(job_id='8535017',state='FAILED',cache_source='original-source',cache_independently_verified=True)
        return saved,dict(job_id='8535999',restore_parent=parent),dict(pid=50,pid_namespace='pid:[new]',start_ticks=200,node='node2')
    def test_equal_container_pid_in_different_terminal_parent_job_is_fresh(self):
        saved,lease,current=self.fixture();validate_restore(saved,saved['identity'],'original-source','new-execution',lease,current)
    def test_same_allocation_without_os_process_proof_is_rejected(self):
        saved,lease,current=self.fixture();lease['job_id']='8535017'
        with self.assertRaises(ValueError):validate_restore(saved,saved['identity'],'original-source','new-execution',lease,current)
    def test_parent_or_cache_provenance_cannot_be_skipped(self):
        for change in ({'state':'RUNNING'},{'cache_independently_verified':False},{'cache_source':'wrong'}):
            saved,lease,current=self.fixture();lease['restore_parent'].update(change)
            with self.assertRaises(ValueError):validate_restore(saved,saved['identity'],'original-source','new-execution',lease,current)
    def test_saved_execution_and_identity_remain_required(self):
        saved,lease,current=self.fixture()
        with self.assertRaises(ValueError):validate_restore(saved,saved['identity'],'original-source',saved['execution'],lease,current)
        with self.assertRaises(ValueError):validate_restore(saved,{},'original-source','new-execution',lease,current)

    def test_same_job_requires_distinct_namespace_or_start_ticks(self):
        saved,lease,current=self.fixture();lease.pop('restore_parent');saved['process_identity']=dict(current,pid_namespace='pid:[old]',start_ticks=100)
        validate_restore(saved,saved['identity'],'original-source','new-execution',lease,current)
        saved['process_identity']=dict(current)
        with self.assertRaises(ValueError):validate_restore(saved,saved['identity'],'original-source','new-execution',lease,current)
