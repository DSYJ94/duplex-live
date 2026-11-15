import unittest
from unittest.mock import patch
from pathlib import Path
from training.personaplex.expanded_launch import container_command

class ContainerStartupTest(unittest.TestCase):
    def test_exact_external_worktree_metadata_is_visible_read_only(self):
        common='/home/u6ada/test/source.git'
        with patch('training.personaplex.expanded_launch.subprocess.check_output',return_value=common+'\n'):
            cmd=container_command(Path('/projects/u6ada/work'),gpu=True,environment={},backup=Path('/home/u6ada/test/backup'))
        self.assertIn(common+':'+common+':ro',cmd)
        self.assertEqual(cmd.count(common+':'+common+':ro'),1)
        self.assertNotIn('/home:/home',cmd)
        self.assertIn('--containall',cmd)
