"""Pack an explicit native inventory and independently read back every S3 member.

No directory-wide upload, credential output, extraction, execution or deployment.
The original manifest and archive index bind every regular file and archived link.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import fcntl
import hashlib
import json
import os
import re
import shlex
from pathlib import Path
import sys
import subprocess
import tarfile
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evals.native_migration_inventory import digest, safe_name


def inventory(path, wanted):
    if digest(path) != wanted:
        raise ValueError('Inventory SHA changed')
    data = json.loads(path.read_text())
    names = [safe_name(r['path']) for r in data['files'] + data['symlinks']]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate inventory path')
    return data


class HashReader:
    def __init__(self, stream):
        self.stream = stream
        self.h = hashlib.sha256()

    def read(self, count=-1):
        block = self.stream.read(count)
        self.h.update(block)
        return block


def pack(args):
    data = inventory(args.manifest, args.manifest_sha256)
    args.output.mkdir(parents=True, exist_ok=True)
    root = args.tree.resolve()
    groups, group, size = [], [], 0
    # Published weights and complete candidate state are packed first; all rows
    # remain mandatory, and no subset can produce a successful final receipt.
    rows = sorted(data['files'], key=lambda r: (not r['path'].startswith('models/'),
                  'context-training-proof-7590052' not in r['path'], r['path']))
    for row in rows:
        if group and size + row['bytes'] > args.batch_bytes:
            groups.append(group)
            group, size = [], 0
        group.append(row)
        size += row['bytes']
    if group:
        groups.append(group)
    index = {'schema': 'dl-native-s3-archive-index-v1',
             'manifest_sha256': args.manifest_sha256, 'archives': [],
             'files': len(rows), 'bytes': sum(r['bytes'] for r in rows),
             'symlinks': len(data['symlinks']), 'source_deployment': False}
    for number, group in enumerate(groups):
        name = 'native-v3-%03d.tar' % number
        target = args.output / name
        with tarfile.open(target, 'x', format=tarfile.PAX_FORMAT) as archive:
            for row in group:
                source = root / row['path']
                if source.is_symlink() or not source.is_file() or not source.resolve().is_relative_to(root):
                    raise ValueError('Regular source escapes inventory tree: ' + row['path'])
                info = archive.gettarinfo(str(source), arcname=row['path'])
                if not info.isfile() or info.size != row['bytes']:
                    raise ValueError('Source type or size changed')
                info.uid = info.gid = 0
                info.uname = info.gname = ''
                with source.open('rb') as stream:
                    reader = HashReader(stream)
                    archive.addfile(info, reader)
                if reader.h.hexdigest() != row['sha256']:
                    raise ValueError('Source SHA mismatch: ' + row['path'])
            links = data['symlinks'] if number == len(groups)-1 else []
            for row in links:
                source = root / row['path']
                if not source.is_symlink() or str(source.readlink()) != row['target']:
                    raise ValueError('Archived link changed')
                info = tarfile.TarInfo(row['path'])
                info.type = tarfile.SYMTYPE
                info.linkname = row['target']
                archive.addfile(info)
        record = {'name': name, 'bytes': target.stat().st_size,
                  'sha256': digest(target), 'files': [r['path'] for r in group],
                  'symlinks': [r['path'] for r in links]}
        index['archives'].append(record)
        print(json.dumps({'packed': name, 'bytes': record['bytes'],
                          'files': len(group)}), flush=True)
    with (args.output/'index.json').open('x') as stream:
        json.dump(index, stream, indent=2)
        stream.write('\n')


def verify_members(stream, spec, files, links):
    expected_files, expected_links = set(spec['files']), set(spec['symlinks'])
    seen_files, seen_links = set(), set()
    with tarfile.open(fileobj=stream, mode='r|') as archive:
        for member in archive:
            name = safe_name(member.name)
            if name in seen_files or name in seen_links:
                raise ValueError('Duplicate archive member')
            if member.isfile() and name in expected_files:
                row = files[name]
                if member.size != row['bytes']:
                    raise ValueError('Destination member size changed')
                h = hashlib.sha256()
                with archive.extractfile(member) as reader:
                    for block in iter(lambda: reader.read(8*1024*1024), b''):
                        h.update(block)
                if h.hexdigest() != row['sha256']:
                    raise ValueError('Destination member SHA mismatch: ' + name)
                seen_files.add(name)
            elif member.issym() and name in expected_links:
                if member.linkname != links[name]['target']:
                    raise ValueError('Destination archived link changed')
                seen_links.add(name)
            else:
                raise ValueError('Unexpected archive member: ' + name)
    if seen_files != expected_files or seen_links != expected_links:
        raise ValueError('Incomplete archive read-back')


def upload(args):
    import boto3
    from boto3.s3.transfer import TransferConfig
    from botocore.config import Config
    from botocore.exceptions import ClientError
    shared_config = Config(connect_timeout=15, read_timeout=120,
        retries={'mode': 'standard', 'max_attempts': 8},
        s3={'payload_signing_enabled': True},
        request_checksum_calculation='when_required', response_checksum_validation='when_required')
    data = inventory(args.manifest, args.manifest_sha256)
    index = json.loads((args.archive_dir/'index.json').read_text())
    assert index['manifest_sha256'] == args.manifest_sha256
    files = {r['path']: r for r in data['files']}
    links = {r['path']: r for r in data['symlinks']}
    listed_files = [n for r in index['archives'] for n in r['files']]
    listed_links = [n for r in index['archives'] for n in r['symlinks']]
    assert len(listed_files) == len(set(listed_files)) and set(listed_files) == set(files)
    assert len(listed_links) == len(set(listed_links)) and set(listed_links) == set(links)
    if args.key_file.stat().st_mode & 0o077:
        raise ValueError('Credential file must have mode0600')
    credentials = json.loads(args.key_file.read_text())
    client = boto3.client('s3', endpoint_url='https://s3api-'+args.region.lower()+'.runpod.io',
        region_name=args.region, aws_access_key_id=credentials['access_key'],
        aws_secret_access_key=credentials['secret_key'],
        config=shared_config)
    client.head_bucket(Bucket=args.bucket)
    transfer = TransferConfig(multipart_threshold=64*1024*1024,
        multipart_chunksize=64*1024*1024, max_concurrency=2)
    lock = threading.Lock()
    complete = []
    source_ready = set()
    existing_verified = set()
    if args.verified_existing_report:
        prior = json.loads(args.verified_existing_report.read_text())
        assert prior['manifest_sha256'] == args.manifest_sha256
        expected = {r['name']: r for r in index['archives']}
        for row in prior['archives']:
            assert row['passed'] and row['sha256'] == expected[row['name']]['sha256']
            assert row['bytes'] == expected[row['name']]['bytes']
            existing_verified.add(row['name'])
    if args.source_pack_receipt and not args.source_helper:
        raise ValueError('A Git-deployed source helper path is required')
    if args.source_helper and (not str(args.source_helper).startswith(
            '/projects/u6si/duplex-live/worktrees/migration-') or '..' in args.source_helper.parts):
        raise ValueError('Unexpected source helper path')
    signing_client = boto3.client('s3', endpoint_url='https://s3api-'+args.region.lower()+'.runpod.io',
        region_name=args.region, aws_access_key_id=credentials['access_key'],
        aws_secret_access_key=credentials['secret_key'], config=shared_config)
    prepared = threading.local()
    class CapturedRequest(Exception):
        pass
    def capture_request(request, **kwargs):
        prepared.value = {'url': request.url, 'headers': {
            k: v.decode() if isinstance(v, bytes) else str(v)
            for k, v in request.headers.items()}}
        raise CapturedRequest()
    signing_client.meta.events.register('before-send.s3.UploadPart', capture_request)
    destination = json.loads(args.destination_ssh.read_text()) if args.destination_ssh else None

    def destination_tar_hash(spec):
        if destination is None:
            raise RuntimeError('Independent destination hash connection not configured')
        name = safe_name(spec['name'])
        assert '/' not in name and name.startswith('native-v3-') and name.endswith('.tar')
        code = ('import hashlib;from pathlib import Path;'
                'p=Path("/workspace/native-archive-v3")/' + repr(name) + ';'
                'h=hashlib.sha256();f=p.open("rb");'
                '[h.update(b) for b in iter(lambda:f.read(16*1024*1024),b"")];'
                'print(h.hexdigest())')
        print(json.dumps({'checking_destination_tar_sha256': name}), flush=True)
        proc = subprocess.run(['ssh', '-i', destination['identity'], '-o', 'IdentitiesOnly=yes',
            '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', '-o', 'StrictHostKeyChecking=yes',
            '-o', 'UserKnownHostsFile='+destination['known_hosts'], '-p', str(destination['port']),
            destination['username']+'@'+destination['host'], 'python3 -c '+shlex.quote(code)],
            capture_output=True, text=True, timeout=600)
        if proc.returncode or proc.stdout.strip() != spec['sha256']:
            raise RuntimeError('Independent destination tar SHA256 failed: '+name)
        return True
    if args.source_pack_receipt:
        expected = {r['name']: r for r in index['archives']}
        for line in args.source_pack_receipt.read_text().splitlines():
            row = json.loads(line)
            if row['passed']:
                assert row['sha256'] == expected[row['name']]['sha256']
                assert row['bytes'] == expected[row['name']]['bytes']
                source_ready.add(row['name'])

    def signed_source_upload(spec, key):
        created = client.create_multipart_upload(Bucket=args.bucket, Key=key)
        upload_id = created['UploadId']
        chunk = 64*1024*1024
        parts = []
        with (args.archive_dir/spec['name']).open('rb') as reader:
            for offset in range(0, spec['bytes'], chunk):
                number = len(parts)+1
                block = reader.read(min(chunk, spec['bytes']-offset))
                try:
                    signing_client.upload_part(Bucket=args.bucket, Key=key,
                        UploadId=upload_id, PartNumber=number, Body=block)
                except CapturedRequest:
                    pass
                request = prepared.value
                assert request['headers']['X-Amz-Content-SHA256'] == hashlib.sha256(block).hexdigest()
                parts.append({'number': number, 'offset': offset,
                              'bytes': len(block), **request})
        # The supported API uses SDK-prepared authorization headers bound to
        # one exact payload/object part. No reusable account key is exported.
        payload = {'name': spec['name'], 'bytes': spec['bytes'],
                   'bucket': args.bucket, 'parts': parts}
        try:
            proc = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2',
                'u6si.aip2.isambard', 'python3 '+shlex.quote(str(args.source_helper))+' upload'],
                input=json.dumps(payload), capture_output=True, text=True, timeout=1800)
            if proc.returncode:
                detail = proc.stderr.splitlines()[-1] if proc.stderr.splitlines() else 'no stderr'
                detail = re.sub(r'https?://\S+', '<redacted-url>', detail)
                detail = re.sub(r'(rps_|user_)[A-Za-z0-9_-]+', '<redacted>', detail)
                raise RuntimeError('Source signed upload failed; exit '+str(proc.returncode)+': '+detail)
            result = json.loads(proc.stdout)
            assert result['name'] == spec['name'] and result['all_signed_parts_uploaded']
            assert [p['PartNumber'] for p in result['parts']] == list(range(1,len(parts)+1))
            try:
                client.complete_multipart_upload(Bucket=args.bucket, Key=key,
                    UploadId=upload_id, MultipartUpload={'Parts': result['parts']})
            except ClientError as error:
                if error.response['Error']['Code'] != 'NoSuchUpload':
                    raise
                # A retry after the provider finalized a large archive can return
                # NoSuchUpload. Size alone is never accepted: independently hash
                # the actual destination file before treating it as transferred.
                destination_tar_hash(spec)
                print(json.dumps({'provider_completion_recovered_by_destination_sha256':
                                  spec['name']}), flush=True)
        except BaseException:
            try:
                client.abort_multipart_upload(Bucket=args.bucket, Key=key, UploadId=upload_id)
            except ClientError as cleanup_error:
                if cleanup_error.response['Error']['Code'] != 'NoSuchUpload':
                    raise
            raise

    def one(spec):
        path = args.archive_dir / spec['name']
        assert path.is_file() and path.stat().st_size == spec['bytes']
        assert digest(path) == spec['sha256']
        key = 'native-archive-v3/' + spec['name']
        print(json.dumps({'starting_archive': spec['name']}), flush=True)
        if spec['name'] in existing_verified:
            destination_tar_hash(spec)
        elif spec['name'] in source_ready:
            signed_source_upload(spec, key)
        else:
            client.upload_file(str(path), args.bucket, key, Config=transfer)
        head = client.head_object(Bucket=args.bucket, Key=key)
        assert head['ContentLength'] == spec['bytes']
        if not args.defer_to_independent_filesystem_verifier:
            response = client.get_object(Bucket=args.bucket, Key=key)
            try:
                verify_members(response['Body'], spec, files, links)
            finally:
                response['Body'].close()
        result = {'archive': spec['name'], 'object_key': key,
                  'archive_bytes': spec['bytes'], 'source_archive_sha256': spec['sha256'],
                  'files': len(spec['files']), 'symlinks': len(spec['symlinks']),
                  'upload_source': 'isambard-single-object-signed-parts'
                                   if spec['name'] in source_ready else 'local-mac',
                  'reused_independently_verified_container': spec['name'] in existing_verified,
                  'independent_s3_readback_all_member_hashes_passed':
                       not args.defer_to_independent_filesystem_verifier,
                  'independent_destination_verification_pending':
                       args.defer_to_independent_filesystem_verifier,
                  'at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        with lock:
            with args.journal.open('a') as journal:
                journal.write(json.dumps(result)+'\n')
                journal.flush()
                os.fsync(journal.fileno())
            complete.append(result)
            print(json.dumps(result), flush=True)
        return result

    # Prove the supported signed-header source route on one tiny archive before
    # starting concurrent large copies. Failures cancel remaining queued work.
    one(index['archives'][0])
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one, spec) for spec in index['archives'][1:]]
        try:
            for future in as_completed(futures):
                future.result()
        except BaseException as error:
            for future in futures:
                future.cancel()
            print(json.dumps({'transfer_failed': type(error).__name__,
                              'message': str(error)[:500]}), flush=True)
            raise
    assert sum(r['files'] for r in complete) == len(files)
    assert sum(r['symlinks'] for r in complete) == len(links)
    for path, key in [(args.manifest, 'native-archive-v3/manifest.json'),
                      (args.archive_dir/'index.json', 'native-archive-v3/index.json')]:
        client.upload_file(str(path), args.bucket, key)
        body = client.get_object(Bucket=args.bucket, Key=key)['Body']
        try:
            assert hashlib.sha256(body.read()).hexdigest() == digest(path)
        finally:
            body.close()
    report = {'schema': ('dl-native-s3-upload-awaiting-full-verification-v1'
                        if args.defer_to_independent_filesystem_verifier
                        else 'dl-native-s3-complete-verification-v1'),
              'at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'bucket': args.bucket, 'region': args.region,
              'manifest_sha256': args.manifest_sha256, 'files': len(files),
              'bytes': sum(r['bytes'] for r in files.values()), 'symlinks': len(links),
              'all_members_independently_read_back_and_verified':
                   not args.defer_to_independent_filesystem_verifier,
              'independent_destination_verification_pending':
                   args.defer_to_independent_filesystem_verifier,
              'archives': complete, 'whole_remote_project_backup': False,
              'model_quality_or_native_numerical_approved': False}
    with args.report.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    client.upload_file(str(args.report), args.bucket, 'native-archive-v3/'+
        ('upload-pending-verification.json' if args.defer_to_independent_filesystem_verifier
         else 'verification.json'))
    print(json.dumps({k: v for k, v in report.items() if k != 'archives'}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for command in ('pack', 'upload-verify'):
        q = sub.add_parser(command)
        q.add_argument('--manifest', type=Path, required=True)
        q.add_argument('--manifest-sha256', required=True)
        if command == 'pack':
            q.add_argument('--tree', type=Path, required=True)
            q.add_argument('--output', type=Path, required=True)
            q.add_argument('--batch-bytes', type=int, default=1024*1024*1024)
        else:
            q.add_argument('--archive-dir', type=Path, required=True)
            q.add_argument('--key-file', type=Path, required=True)
            q.add_argument('--bucket', required=True)
            q.add_argument('--region', required=True)
            q.add_argument('--journal', type=Path, required=True)
            q.add_argument('--report', type=Path, required=True)
            q.add_argument('--workers', type=int, default=3)
            q.add_argument('--source-pack-receipt', type=Path)
            q.add_argument('--source-helper', type=Path)
            q.add_argument('--defer-to-independent-filesystem-verifier', action='store_true')
            q.add_argument('--verified-existing-report', type=Path)
            q.add_argument('--destination-ssh', type=Path)
    args = p.parse_args()
    if args.command == 'pack':
        pack(args)
    else:
        # One active writer per destination. A preexisting journal requires an
        # explicit recovery review rather than a duplicate upload job.
        with args.journal.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.journal.exists() or args.report.exists():
                raise RuntimeError('Existing transfer receipt: inspect before restarting')
            upload(args)


if __name__ == '__main__':
    main()
