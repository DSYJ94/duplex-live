from pathlib import Path
import unittest
from evals.duplex_input_validator import cases


class HistoricalInputApproval(unittest.TestCase):
    def test_wrong_commit_or_incomplete_source_never_opens_input_artifacts(self):
        root=Path(__file__).resolve().parents[2]
        missing=Path('/deliberately-absent-evaluation-input')
        with self.assertRaisesRegex(ValueError,'both source and commit'):
            cases(missing,missing,[],source=root)
        with self.assertRaisesRegex(RuntimeError,'declared clean worktree'):
            cases(missing,missing,[],source=root,commit='0'*40)


if __name__=='__main__':unittest.main()
