import json
from pathlib import Path
import subprocess
import threading
import time

import pytest

from training.venus.opal_supervisor import GitSupervisor, atomic, digest, existing_ssh
from training.venus.opal_adapter import load_upstream

SOURCE=Path('/private/tmp/duplex-opal-upstream-20261028')


def commit(repo, files):
    repo.mkdir(exist_ok=True)
    if not (repo/'.git').exists():
        subprocess.run(['git','init','-q',str(repo)],check=True)
    for name, value in files.items():
        path=repo/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value)
    subprocess.run(['git','-C',str(repo),'add','training/venus/opal_programs'],check=True)
    subprocess.run(['git','-C',str(repo),'-c','user.name=OPAL fixture','-c','user.email=fixture@example.invalid',
                    'commit','-qm','Fixture optimizer program'],check=True)
    return subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()


def wait_for(predicate, timeout=10):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        value=predicate()
        if value:return value
        time.sleep(.01)
    raise TimeoutError('Expected actual mailbox state did not arrive')


def test_git_program_bytes_override_uncommitted_files_and_stale_or_unscoped_decisions_fail(tmp_path):
    repo=tmp_path/'repo';path='training/venus/opal_programs/fixture/updater.py'
    source=commit(repo,{path:'value = 1\n'})
    (repo/path).write_text('value = 999\n')
    backend=GitSupervisor(tmp_path/'decisions',repo)
    request=dict(request_id='current')
    decision=dict(request_id='current',request_sha256='exact',action='adapt',source=source,
                  programs={'updater.py':path},reason='A fixture change, not a training recommendation.')
    programs,actual=backend.verify(decision,request,'exact')
    assert programs=={'updater.py':'value = 1\n'} and actual==source
    with pytest.raises(ValueError):backend.verify(dict(decision,request_id='old'),request,'exact')
    with pytest.raises(ValueError):backend.verify(dict(decision,request_sha256='old'),request,'exact')
    with pytest.raises(ValueError):backend.verify(dict(decision,programs={'updater.py':'API keys.md'}),request,'exact')
    with pytest.raises(ValueError):backend.verify(dict(decision,action='keep'),request,'exact')


@pytest.mark.parametrize('failure',['missing-deployment','cancelled-during-verification'])
def test_missing_remote_git_deployment_cannot_change_any_program(tmp_path,failure):
    repo=tmp_path/'repo';path='training/venus/opal_programs/fixture/updater.py'
    source=commit(repo,{path:'value = 2\n'})
    workspace=tmp_path/'workspace';workspace.mkdir()
    (workspace/'updater.py').write_text('value = 1\n')
    (workspace/'analysis.py').write_text('analysis = True\n')
    def not_deployed(*args):
        if failure=='missing-deployment':raise RuntimeError('Exact remote Git source is absent')
        backend.cancel()
    backend=GitSupervisor(tmp_path/'decisions',repo,verify_deployment=not_deployed)
    backend.preflight();errors=[]
    def serve():
        try:backend.decide(workspace,'fixture observations','fixture system',tmp_path)
        except Exception as error:errors.append(error)
    thread=threading.Thread(target=serve);thread.start()
    request=wait_for(lambda:next((tmp_path/'decisions').glob('*/identity.json'),None))
    identity=json.loads(request.read_text())
    atomic(request.with_name('decision.json'),dict(**identity,action='adapt',source=source,
           programs={'updater.py':path},reason='Fixture rejected until exact source deployment.'))
    thread.join(timeout=2)
    assert not thread.is_alive() and isinstance(errors[0],RuntimeError if failure=='missing-deployment' else TimeoutError)
    assert (workspace/'updater.py').read_text()=='value = 1\n'
    assert not request.with_name('committed-decision.json').exists()


@pytest.mark.skipif(not SOURCE.exists(),reason='Pinned OPAL checkout required')
def test_existing_iap_argv_is_used_unchanged_without_new_alias_or_upstream_edits():
    load_upstream(SOURCE)
    from opal.backends import mailbox
    original=mailbox._Folder
    argv=['ssh','-o','StrictHostKeyChecking=yes','-o','BatchMode=yes','-o','IdentitiesOnly=yes',
          '-o','UserKnownHostsFile=/tmp/fixture-known-hosts','-o',
          'ProxyCommand=gcloud compute start-iap-tunnel fixture 22','fixture@127.0.0.1']
    with existing_ssh(mailbox,argv):
        folder=mailbox._Folder('/workspace/fixture','existing-native-host')
        assert folder.base==argv
        with pytest.raises(ValueError):mailbox._Folder('/workspace/fixture','other-host')
    assert mailbox._Folder is original
    with pytest.raises(ValueError):
        with existing_ssh(mailbox,['ssh','unverified-host']):pass


@pytest.mark.skipif(not SOURCE.exists(),reason='Pinned OPAL checkout required')
def test_real_upstream_mailbox_applies_committed_edit_after_native_steps_continue(monkeypatch,tmp_path):
    from evals.harness.test_venus_native_opal import setup,make
    from training.venus import train
    from training.venus.native_optimizer import NativeOpal
    from training.venus.opal_adapter import checkpoint_manifest
    load_upstream(SOURCE)
    from opal.backends import mailbox
    recipe=setup(monkeypatch);model,base,rows=make()
    # Tiny fixture only: the generated program must already be in Git before
    # its reply may return. No Git changes are made to the actual project.
    repo=tmp_path/'fixture-repo'
    commit(repo,{'training/venus/opal_programs/seed/updater.py':'seed = True\n'})
    supervisor=GitSupervisor(tmp_path/'decisions',repo,timeout=120)
    runtime=NativeOpal(base,model,source=SOURCE,run_dir=tmp_path/'train',recipe=recipe,
                       backend='mailbox:'+str(tmp_path/'box'),agent_budget=.01)
    errors=[]
    def serve():
        try:mailbox.run_broker(str(tmp_path/'box'),supervisor,tmp_path/'broker',poll=.01,
                              give_up_after=2,log=lambda _:None)
        except BaseException as error:errors.append(error)
    broker=threading.Thread(target=serve,daemon=True);broker.start()
    try:
        for step in range(16):train.train_step(model,runtime.optimizer,None,rows,step,opal=runtime)
        def actual_request():
            if errors:raise errors[0]
            return next((tmp_path/'decisions').glob('*/request.json'),None)
        request_file=wait_for(actual_request)
        request=json.loads(request_file.read_text());identity=json.loads(request_file.with_name('identity.json').read_text())
        assert 'train_batch_ordinary_ce' in request['message']
        # There is a real unanswered request. Training remains nonblocking.
        for step in (16,17):train.train_step(model,runtime.optimizer,None,rows,step,opal=runtime)
        assert checkpoint_manifest(runtime.optimizer)['update']==18
        assert checkpoint_manifest(runtime.optimizer)['program_version']==0
        text=request['programs']['updater.py'];marker='        for o in self.opts:\n            o.step()'
        assert text.count(marker)==1
        changed=text.replace(marker,"        for group in self.param_groups:\n            group['weight_decay'] = .02\n"+marker)
        path='training/venus/opal_programs/fixture/updater.py'
        source=commit(repo,{path:changed})
        atomic(request_file.with_name('decision.json'),dict(**identity,action='adapt',source=source,
               programs={'updater.py':path},reason='Deterministic mailbox fixture; no optimizer quality claim.'))
        wait_for(lambda:request_file.with_name('committed-decision.json').exists())
        wait_for(lambda:runtime.optimizer.worker.ready is not None)
        train.train_step(model,runtime.optimizer,None,rows,18,opal=runtime)
        manifest=checkpoint_manifest(runtime.optimizer)
        assert manifest['program_version']==1 and manifest['update']==19
        assert base.param_groups[0]['weight_decay']==.02
        audit=json.loads(request_file.with_name('committed-decision.json').read_text())
        assert audit['source']==source and audit['external_api_calls']==0 and audit['cost_usd']==0
        assert digest(changed.encode())==audit['program_hashes']['updater.py']
    finally:
        runtime.close();broker.join(timeout=4)
    assert not broker.is_alive() and not errors


@pytest.mark.skipif(not SOURCE.exists(),reason='Pinned OPAL checkout required')
def test_old_attempt_answer_cannot_change_current_program(tmp_path):
    load_upstream(SOURCE)
    from opal.backends.mailbox import MailboxBackend
    backend=MailboxBackend('mailbox:'+str(tmp_path/'box'))
    workspace=tmp_path/'workspace';workspace.mkdir();program=workspace/'updater.py';program.write_text('value = 1\n')
    backend.setup(tmp_path,workspace);backend._ensure_folder()
    atomic(backend.folder/'response-0001.json',dict(run_id=backend.run_id,attempt='stale',files={'updater.py':'value = 999\n'}))
    assert backend._answer(1,workspace,time.time()) is None
    assert program.read_text()=='value = 1\n'
