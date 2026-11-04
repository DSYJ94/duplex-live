import hashlib,json
from pathlib import Path
import pytest
from training.venus.checkpoint_stream import CheckpointStream


@pytest.mark.parametrize('corrupt',[False,True])
def test_only_hash_verified_immutable_checkpoint_is_recorded(tmp_path,corrupt):
    receipts=tmp_path/'receipts';receipts.mkdir();backup=tmp_path/'backup'
    stream=CheckpointStream(['ssh','host'],'/native/run',backup,receipts,'source')
    content=b'actual synthetic checkpoint bytes'
    row=dict(path='main/step-0060.pt',bytes=len(content),sha256=hashlib.sha256(content).hexdigest())
    def invoke(args,seconds):
        if args[0]=='ssh':return json.dumps([row])
        Path(args[-1]).write_bytes(b'corrupt' if corrupt else content);return ''
    stream.invoke=invoke
    original=stream.record
    def record(value):original(value);stream.stop_event.set()
    stream.record=record;stream.run()
    records=[json.loads(x) for x in (receipts/'stream-backup.jsonl').read_text().splitlines()]
    if corrupt:
        assert not stream.verified and records[-1]['final_full_copyback_required']
    else:
        assert stream.verified[row['path']]==row and stream.bytes==len(content)
        assert records[-1]['checkpoint_verified']['sha256']==row['sha256']
    assert json.loads((receipts/'stream-backup-binding.json').read_text())['source']=='source'
    # A second run cannot silently take over the same backup destination.
    with pytest.raises(FileExistsError):CheckpointStream(['ssh','host'],'/other',backup,receipts,'other')


def test_storage_ceiling_rejects_before_transfer(tmp_path):
    receipts=tmp_path/'receipts';receipts.mkdir()
    stream=CheckpointStream(['ssh','host'],'/native/run',tmp_path/'backup',receipts,'source')
    def invoke(args,seconds):
        assert args[0]=='ssh','Unauthorized file transfer'
        return json.dumps([dict(path='expanded-init.pt',bytes=100_000_000_001,sha256='unused')])
    stream.invoke=invoke;stream.run()
    assert not stream.verified
    record=json.loads((receipts/'stream-backup.jsonl').read_text())
    assert record['error_type']=='ValueError'
