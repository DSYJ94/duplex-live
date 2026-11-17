import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from training.personaplex.broad_corpus_download import acquire


class BroadCorpusDownloadTest(unittest.TestCase):
    def opener(self, url, timeout):
        if url.endswith('.jsonl.gz'):
            payload = gzip.compress(b'{"message_tree_id":"fixture","prompt":{}}\n')
        elif url.endswith('.jsonl'):
            payload = (json.dumps(dict(instruction='Question', context='Source context', response='Answer', category='closed_qa')) + '\n').encode()
        elif url.endswith('.parquet'):
            payload = b'PAR1fixturePAR1'
        else:
            payload = b'fixture owner card/licence'
        response = io.BytesIO(payload)
        response.headers = {'ETag': 'opaque-not-a-content-hash'}
        return response

    def test_all_originals_licences_and_official_validation_remain_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'staging'
            with patch('builtins.print'):
                result = acquire(root, opener=self.opener)
            self.assertEqual(len(result['sources']), 7)
            self.assertTrue((root / 'download-inventory.json').exists())
            self.assertEqual(sum(r['official_split'] == 'validation' for r in result['sources']), 1)
            self.assertFalse(result['decontamination_complete'])
            self.assertFalse(result['native_audio_created'])
            self.assertFalse(result['training_started'])
            self.assertEqual({r['licence'] for r in result['sources']}, {'Apache-2.0', 'CC-BY-SA-3.0'})
            with self.assertRaises(FileExistsError): acquire(root, opener=self.opener)

    def test_invalid_complete_source_cannot_be_published_as_a_corpus(self):
        def malformed(url, timeout):
            response = self.opener(url, timeout)
            if url.endswith('.jsonl.gz'):
                response = io.BytesIO(gzip.compress(b'not JSON\n'))
                response.headers = {}
            return response
        with tempfile.TemporaryDirectory() as temp, patch('builtins.print'):
            root = Path(temp) / 'staging'
            with self.assertRaises(json.JSONDecodeError): acquire(root, opener=malformed)
            self.assertFalse((root / 'download-inventory.json').exists())
