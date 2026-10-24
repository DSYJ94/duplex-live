"""Inspect saved-export structure without importing Torch or executing pickle."""
import json
from pathlib import Path
import pickle
import struct
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from evals.native_saved_export_metadata import (
    adapter_layout, checkpoint_prefix, primitive_prefix, sha)


class NeverUnpickle:
    def __reduce__(self):
        return (exec, ("raise AssertionError('Executable checkpoint was evaluated')",))


class SavedExportMetadataTests(unittest.TestCase):
    def prefix(self, names=None):
        return dict(schema='dl-context-training-state-v1',
            metadata=dict(recipe=dict(rank=8, lr=2e-7), repo_commit='a'*40),
            completed_steps=8, parameter_names=['a', 'b'] if names is None else names,
            adapters=NeverUnpickle())

    def test_complete_names_before_executable_tensor_section(self):
        for protocol in (2, 3, 4, 5):
            for count in (0, 1, 26, 1001):
                with self.subTest(protocol=protocol, count=count):
                    expected=self.prefix([str(i) for i in range(count)])
                    actual=primitive_prefix(pickle.dumps(expected, protocol=protocol))
                    self.assertEqual(actual, {k:v for k,v in expected.items() if k!='adapters'})

    def test_executable_opcode_inside_metadata_is_rejected(self):
        for protocol in (2, 4):
            value=self.prefix();value['metadata']['injected']=NeverUnpickle()
            with self.assertRaisesRegex(ValueError, 'Unsupported opcode'):
                primitive_prefix(pickle.dumps(value, protocol=protocol))

    def test_no_pickle_deserializer_is_called(self):
        payload=pickle.dumps(self.prefix(), protocol=4)
        with patch('pickle.loads', side_effect=AssertionError('unsafe loads')), \
                patch('pickle.load', side_effect=AssertionError('unsafe load')):
            self.assertEqual(primitive_prefix(payload)['completed_steps'], 8)

    def test_incomplete_or_wrong_prefix_is_rejected(self):
        value=self.prefix();del value['adapters']
        for payload in (pickle.dumps(value, protocol=4), b'not a pickle', b'x'*2_000_001):
            with self.assertRaises(ValueError):primitive_prefix(payload)
        for key, replacement in [('schema', 'other'), ('metadata', []),
                ('completed_steps', True), ('parameter_names', [3])]:
            value=self.prefix();value[key]=replacement
            with self.subTest(key=key), self.assertRaises(ValueError):
                primitive_prefix(pickle.dumps(value, protocol=4))

    def test_nonfinite_metadata_is_rejected(self):
        for number in (float('nan'), float('inf'), float('-inf')):
            value=self.prefix();value['metadata']['number']=number
            with self.assertRaises(ValueError):primitive_prefix(pickle.dumps(value, protocol=4))

    def test_checkpoint_zip_requires_hash_and_one_index(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'state.pt'
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('training/data.pkl', pickle.dumps(self.prefix(), protocol=2))
            self.assertEqual(checkpoint_prefix(path, sha(path))['completed_steps'], 8)
            with self.assertRaisesRegex(ValueError, 'Checkpoint bytes changed'):
                checkpoint_prefix(path, '0'*64)
            with zipfile.ZipFile(path, 'a') as archive:
                archive.writestr('other/data.pkl', pickle.dumps(self.prefix(), protocol=2))
            with self.assertRaisesRegex(ValueError, 'One bounded checkpoint index'):
                checkpoint_prefix(path, sha(path))

    def write_adapter(self, path, header=None, raw=None, body=b'\0'*32):
        if raw is None:
            if header is None:
                header={'layer.lora_A.weight':dict(dtype='F32', shape=[2, 4], data_offsets=[0, 32])}
            raw=json.dumps(header, separators=(',', ':')).encode()
        path.write_bytes(struct.pack('<Q', len(raw))+raw+body)
        return sha(path)

    def test_adapter_parameter_layout_and_string_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'adapters.safetensors'
            header={'a':dict(dtype='F32', shape=[2, 4], data_offsets=[0, 32]),
                '__metadata__':{'purpose':'fixture'}}
            digest=self.write_adapter(path, header=header)
            result=adapter_layout(path, digest)
            self.assertEqual(result['parameter_count'], 8)
            self.assertEqual(set(result['tensors']), {'a'})
            with self.assertRaisesRegex(ValueError, 'Adapter bytes changed'):
                adapter_layout(path, '0'*64)

    def test_adapter_rejects_duplicate_keys_holes_and_overlaps(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'adapters.safetensors'
            duplicate=b'{"a":{"dtype":"F32","shape":[2,4],"data_offsets":[0,32]},"a":{}}'
            with self.assertRaisesRegex(ValueError, 'Duplicate object key'):
                adapter_layout(path, self.write_adapter(path, raw=duplicate))
            for offsets in ([4, 36], [0, 28], [-4, 28]):
                header={'a':dict(dtype='F32', shape=[2,4], data_offsets=offsets)}
                with self.subTest(offsets=offsets), self.assertRaises(ValueError):
                    adapter_layout(path, self.write_adapter(path, header=header))
            overlap={n:dict(dtype='F32', shape=[2,4], data_offsets=[0,32]) for n in ('a','b')}
            with self.assertRaisesRegex(ValueError, 'overlap or hole'):
                adapter_layout(path, self.write_adapter(path, header=overlap))

    def test_adapter_rejects_wrong_types_and_unindexed_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'adapters.safetensors'
            for shape, dtype, offsets in [([True,4],'F32',[0,16]), ([2,4],'F16',[0,16]),
                    ([2,4],'F32',[False,32]), ([0,4],'F32',[0,0])]:
                header={'a':dict(dtype=dtype, shape=shape, data_offsets=offsets)}
                with self.subTest(header=header), self.assertRaises(ValueError):
                    adapter_layout(path, self.write_adapter(path, header=header))
            with self.assertRaisesRegex(ValueError, 'Unindexed adapter bytes'):
                adapter_layout(path, self.write_adapter(path, body=b'\0'*36))
            with self.assertRaisesRegex(ValueError, 'String-only'):
                adapter_layout(path, self.write_adapter(path, header={'__metadata__':{'x':1}}))


if __name__=='__main__':unittest.main()
