from pathlib import Path
import tempfile
import unittest
from evals.context_event_log import EventLog


class EventLogTest(unittest.TestCase):
    def test_deferred_logging_performs_no_filesystem_write_during_input(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'events.jsonl'
            with EventLog(path,'deferred') as log:
                log.write('{"frame":0}\n');log.flush()
                self.assertFalse(path.exists())
                log.write('{"frame":1}\n');log.flush()
            self.assertEqual(path.read_text(),'{"frame":0}\n{"frame":1}\n')
            self.assertFalse(log.summary()['filesystem_writes_during_input'])

    def test_modes_preserve_identical_complete_records_even_on_exception(self):
        with tempfile.TemporaryDirectory() as root:
            paths=[]
            for mode in ('immediate','deferred'):
                p=Path(root)/mode;paths.append(p)
                with self.assertRaisesRegex(RuntimeError,'source failed'):
                    with EventLog(p,mode) as log:
                        log.write('complete first row\n');log.flush()
                        raise RuntimeError('source failed')
                self.assertEqual(log.summary()['in_stream_calls'],2)
            self.assertEqual(paths[0].read_bytes(),paths[1].read_bytes())


if __name__=='__main__':unittest.main()
