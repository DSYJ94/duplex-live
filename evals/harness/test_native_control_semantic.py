"""Synthetic API-response fixtures; these are not independent label judgments."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from data.synthetic.native_control_repair import build
from evals import native_control_semantic as semantic


class CompleteSemanticBinding(unittest.IsolatedAsyncioTestCase):
    async def test_no_partial_or_rewritten_semantic_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch = root / 'batch'
            batch.mkdir()
            for path in (batch / 'teacher.json', root / 'quality.json', root / 'source-audit.json'):
                path.write_text('{}')
            messages = {s['id']: [dict(role='user', content='Artificial test only.')]
                        for s in build()['cases']}

            def request(sent, tools, path):
                call = dict(type='function_call', name=tools[0]['name'],
                            arguments=json.dumps(dict(verdict='pass', reason='Artificial test only.')))
                data = dict(id=path.stem, model='fixture-reviewer', status='completed', output=[call])
                path.write_text(json.dumps(data))
                return data, call, 0.

            args = (root / 'source', root / 'source-audit.json', batch, root / 'quality.json')
            with patch.object(semantic, 'messages', return_value=messages), patch.object(semantic, 'request', request):
                report = await semantic.review(*args, root / 'semantic', 'fixture-key', root / 'lease')
                approval = root / 'semantic/semantic.json'
                self.assertEqual(len(report['results']), 40)
                self.assertFalse(report['training_approved'])
                semantic.verify(*args, approval)
                missing = dict(report, results=report['results'][:-1])
                approval.write_text(json.dumps(missing))
                with self.assertRaisesRegex(ValueError, 'complete unchanged'):
                    semantic.verify(*args, approval)
                approval.write_text(json.dumps(report))
                response = approval.parent / report['results'][0]['response_file']
                data = json.loads(response.read_text())
                data['output'][0]['arguments'] = json.dumps(dict(verdict='fail', reason='Artificial test only.'))
                response.write_text(json.dumps(data))
                report['results'][0]['response_sha256'] = semantic.sha256(response)
                approval.write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError, 'response or exact'):
                    semantic.verify(*args, approval)


if __name__ == '__main__':
    unittest.main()
