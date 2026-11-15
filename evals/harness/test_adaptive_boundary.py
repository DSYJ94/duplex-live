"""Real pinned OPAL CPU transaction tests; never loads a voice model."""
import copy,os,tempfile,unittest
from pathlib import Path
import torch
from training.venus.native_optimizer import NativeOpal,assert_optimizer_equal
from training.personaplex.adaptive_boundary import BoundaryController,state_hash
from evals.harness.test_adaptive_policy import request

class AdaptiveBoundary(unittest.TestCase):
    def test_keep_parent_restores_exact_before_schedule_transition(self):
        source=os.environ['DUPLEX_OPAL_SOURCE']
        with tempfile.TemporaryDirectory()as root:
            model=torch.nn.Linear(2,1);base=torch.optim.AdamW(model.parameters(),lr=1e-3)
            recipe=dict(updates=196,warmup_updates=16,learning_rate=1e-3,gradient_clip=None)
            parent=NativeOpal(base,model,source=source,run_dir=Path(root)/'keep',recipe=recipe,backend='keep')
            try:
                for i in range(49):
                    base.zero_grad(set_to_none=True);loss=model(torch.ones(2,2)).square().mean();loss.backward();parent.step(float(loss.detach()),i)
                base.zero_grad(set_to_none=True)
                saved=copy.deepcopy(parent.optimizer.state_dict());weights=copy.deepcopy(model.state_dict())
            finally:parent.close()
            model2=torch.nn.Linear(2,1);model2.load_state_dict(weights);base2=torch.optim.AdamW(model2.parameters(),lr=1e-3)
            active=NativeOpal(base2,model2,source=source,run_dir=Path(root)/'adaptive',recipe=recipe,backend='boundary-local')
            try:
                active.optimizer.load_state_dict(saved);assert_optimizer_equal(saved,active.optimizer.state_dict())
                counters=(saved['decisions'],saved['cost'],saved['ledger'])
                ctrl=BoundaryController(active)
                receipt=ctrl.enable(dict(resume_step=49,checkpoint_sha256='b'*64,backup_sha256='c'*64,data_sha256='a'*64))
                after=active.optimizer.state_dict()
                self.assertEqual((after['decisions'],after['cost'],after['ledger']),counters)
                self.assertEqual(receipt['previous_decision_schedule'],saved['decision_schedule'])
                self.assertEqual(active.optimizer.schedule,[])
            finally:active.close()

    def test_actual_install_adam_scheduler_rng_and_restore(self):
        source=os.environ['DUPLEX_OPAL_SOURCE']
        with tempfile.TemporaryDirectory() as root:
            torch.manual_seed(21)
            model=torch.nn.Linear(2,1);base=torch.optim.AdamW(model.parameters(),lr=1e-3)
            recipe=dict(updates=196,warmup_updates=16,learning_rate=1e-3,gradient_clip=None)
            active=NativeOpal(base,model,source=source,run_dir=Path(root)/'first',recipe=recipe,backend='boundary-local')
            try:
                self.assertEqual(active.optimizer.worker.backend.name,'keep')
                self.assertEqual(active.optimizer.schedule,[])
                def update(owner,net,opt,step):
                    opt.zero_grad(set_to_none=True);loss=net(torch.ones(3,2)).square().mean();loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
                    owner.step(float(loss.detach()),step);opt.zero_grad(set_to_none=True)
                for step in range(49):update(active,model,base,step)
                controller=BoundaryController(active)
                context=dict(resume_step=49,checkpoint_sha256='b'*64,backup_sha256='c'*64,data_sha256='a'*64)
                before=controller._snapshot();receipt=controller.enable(context)
                self.assertTrue(receipt['preserved_state']);self.assertEqual(receipt['new_optimizer_updates'],0)
                self.assertEqual(state_hash(before),state_hash(controller._snapshot()))
                decision=controller.evaluate(request(dev=dict(text=2.2,audio=3.)),context,now=100.)
                self.assertTrue(decision['applied']);self.assertEqual(controller.state['current_multiplier'],.75)
                saved=copy.deepcopy(active.optimizer.state_dict());ctrl=controller.state_dict();weights=copy.deepcopy(model.state_dict())
                bundle=Path(root)/'fresh-input.pt'
                from training.venus.state import rng_state
                torch.save(dict(opal=saved,controller=ctrl,model=weights,rng=rng_state(),recipe=recipe),bundle)
                nominal=base.param_groups[0]['lr'];update(active,model,base,49)
                self.assertEqual(active.optimizer.updater.last_effective_rates,[nominal*.75])
                expected=copy.deepcopy(model.state_dict());expected_adam=copy.deepcopy(base.state_dict())
                import torch.multiprocessing as mp
                mp.spawn(fresh_worker,args=(str(bundle),root,source),nprocs=1,join=True)
                fresh=torch.load(Path(root)/'fresh-output.pt',weights_only=False)
                self.assertNotEqual(fresh['pid'],os.getpid())
                self.assertEqual(state_hash(expected),state_hash(fresh['model']))
                self.assertEqual(state_hash(expected_adam),state_hash(fresh['adam']))
                self.assertFalse(controller.evaluate(request(),dict(context,resume_step=50),now=100.).get('applied',False))
                model2=torch.nn.Linear(2,1);model2.load_state_dict(weights);base2=torch.optim.AdamW(model2.parameters(),lr=1e-3)
                other=NativeOpal(base2,model2,source=source,run_dir=Path(root)/'second',recipe=recipe,backend='boundary-local')
                try:
                    other.optimizer.load_state_dict(saved);assert_optimizer_equal(saved,other.optimizer.state_dict())
                    restored=BoundaryController(other);restored.load_state_dict(ctrl)
                    update(other,model2,base2,49)
                    self.assertEqual(state_hash(expected),state_hash(model2.state_dict()))
                    self.assertEqual(state_hash(expected_adam),state_hash(base2.state_dict()))
                finally:other.close()
            finally:active.close()



def fresh_worker(rank,bundle,root,source):
    from training.venus.state import restore_rng
    saved=torch.load(bundle,weights_only=False)
    model=torch.nn.Linear(2,1);model.load_state_dict(saved['model']);base=torch.optim.AdamW(model.parameters(),lr=1e-3)
    active=NativeOpal(base,model,source=source,run_dir=Path(root)/'fresh-process',recipe=saved['recipe'],backend='boundary-local')
    try:
        active.optimizer.load_state_dict(saved['opal']);assert_optimizer_equal(saved['opal'],active.optimizer.state_dict())
        ctrl=BoundaryController(active);ctrl.load_state_dict(saved['controller']);restore_rng(saved['rng'])
        base.zero_grad(set_to_none=True);loss=model(torch.ones(3,2)).square().mean();loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);active.step(float(loss.detach()),49)
        torch.save(dict(pid=os.getpid(),model=model.state_dict(),adam=base.state_dict()),Path(root)/'fresh-output.pt')
    finally:active.close()


def collective_worker(rank,root,source):
    import datetime
    import torch.distributed as dist
    dist.init_process_group('gloo',init_method='file://'+str(Path(root)/'rendezvous'),rank=rank,world_size=2,timeout=datetime.timedelta(seconds=60))
    try:
        torch.manual_seed(42);model=torch.nn.Linear(2,1);base=torch.optim.AdamW(model.parameters(),lr=1e-3)
        from training.personaplex.distributed_opal import DistributedOpal
        class Runtime:
            wrapped=model
            clips=0
            def clip_grad_norm(self,limit):
                self.clips+=1
                return torch.nn.utils.clip_grad_norm_(model.parameters(),limit)
        runtime=Runtime()
        driver=DistributedOpal(runtime,base,source=source,run_dir=Path(root)/'collective',recipe=dict(updates=196,warmup_updates=16,learning_rate=1e-3,gradient_clip=1.),adaptive=True)
        active=driver.active
        try:
            for i in range(49):
                base.zero_grad(set_to_none=True);loss=model(torch.ones(2,2)).square().mean();loss.backward();active.step(float(loss.detach()),i)
            base.zero_grad(set_to_none=True);torch.manual_seed(100+rank)
            ctrl=driver.boundary;context=dict(resume_step=49,checkpoint_sha256='b'*64,backup_sha256='c'*64,data_sha256='a'*64)
            receipt=ctrl.enable(context)
            assert len({r['rng_sha256']for r in receipt['rank_checks']})==2
            result=ctrl.evaluate(request(dev=dict(text=2.2,audio=3.))if rank==0 else None,context,now=100.)
            assert result['applied']and ctrl.state['current_multiplier']==.75
            before=ctrl._snapshot()
            bad=dict(context,checkpoint_sha256=('d'if rank else'b')*64)
            try:ctrl.evaluate(request(),bad,now=100.)
            except ValueError:pass
            else:raise AssertionError('Disagreeing checkpoint was accepted')
            assert state_hash(before)==state_hash(ctrl._snapshot())
            loss=model(torch.ones(2,2)).square().mean();loss.backward()
            result=driver.step(float(loss.detach()),49)
            assert runtime.clips==1 and result['adaptive_multiplier']==.75
            assert driver.rank_state({})['adaptive_controller']==ctrl.state_dict()
        finally:active.close()
    finally:dist.destroy_process_group()


class CollectiveBoundary(unittest.TestCase):
    def test_two_real_gloo_ranks_program_agreement_and_distinct_rng(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory()as root:
            mp.spawn(collective_worker,args=(root,os.environ['DUPLEX_OPAL_SOURCE']),nprocs=2,join=True)

if __name__=='__main__':unittest.main()
