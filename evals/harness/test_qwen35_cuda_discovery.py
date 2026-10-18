import tempfile
from pathlib import Path
import unittest
from evals.qwen35_cuda_discovery import mapped_driver_directory


class LoadedDriverScope(unittest.TestCase):
    def test_same_mapped_driver_bytes_allow_its_directory_and_duplicate_segments(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);a=folder/'libcuda.so.1';a.write_bytes(b'fixture-current-driver');(folder/'libcuda.so').write_bytes(a.read_bytes())
            maps='000-111 r-xp 000 00:00 1 '+str(a)+'\n111-222 rw-p 000 00:00 1 '+str(a)
            result,digest=mapped_driver_directory(maps);self.assertEqual(result,folder);self.assertEqual(len(digest),64)
    def test_wrong_linker_bytes_or_ambiguous_driver_never_select_a_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);a=folder/'libcuda.so.1';a.write_bytes(b'actual');(folder/'libcuda.so').write_bytes(b'wrong-compat-library')
            maps='000-111 r-xp 000 00:00 1 '+str(a)
            with self.assertRaisesRegex(RuntimeError,'do not match'):mapped_driver_directory(maps)
            with self.assertRaisesRegex(RuntimeError,'exactly one'):mapped_driver_directory(maps+'\n000-111 r-xp 000 00:00 2 /other/libcuda.so.1')
            with self.assertRaisesRegex(RuntimeError,'exactly one'):mapped_driver_directory('000-111 r-xp 000 00:00 1 /other/libcudart.so')


if __name__=='__main__':unittest.main()
