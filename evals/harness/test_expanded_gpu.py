import os,sys,unittest
from unittest.mock import patch
from training.personaplex import expanded_gpu

class DriverBeforeImport(unittest.TestCase):
    def test_actual_driver_link_is_set_before_native_import(self):
        def run(module,run_name):
            self.assertEqual(os.environ['TRITON_LIBCUDA_PATH'],'/verified/driver')
            self.assertEqual(sys.argv,['native.module','build'])
        with patch.dict(os.environ,{'DUPLEX_LIVE_LEASE_FILE':'/phase/job/lease.json'}),patch.object(sys,'argv',['bootstrap','native.module','build']),patch.object(expanded_gpu.subprocess,'check_output',return_value='/verified/driver\n')as link,patch.object(expanded_gpu.runpy,'run_module',side_effect=run):
            expanded_gpu.main()
            first=link.call_args.args[0][-1]
            expanded_gpu.main()
            self.assertNotEqual(first,link.call_args.args[0][-1])
            self.assertEqual(link.call_args.args[0][2:4],['-m','evals.cuda_linker_path'])

if __name__=='__main__':unittest.main()
