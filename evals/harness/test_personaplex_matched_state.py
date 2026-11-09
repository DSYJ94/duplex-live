import copy
import unittest
import torch
from torch import nn

from evals.personaplex_adapters import AdapterLinear,adapter_parameters
from training.personaplex.matched_state import adapter_state,load_adapter,materialize_adam
from training.venus.state import exact_state


class StateOwnership(unittest.TestCase):
    def model(self):
        base=nn.Sequential(nn.Linear(3,4),nn.Linear(4,2)).requires_grad_(False)
        base[0]=AdapterLinear(base[0],rank=2);base[1]=AdapterLinear(base[1],rank=2)
        return base

    def test_native_adapter_selector_is_complete_and_excludes_source_weights(self):
        model=self.model();state=adapter_state(model)
        self.assertEqual(set(state),{'0.lora_A.weight','0.lora_B.weight','1.lora_A.weight','1.lora_B.weight'})
        self.assertEqual(sum(x.numel() for x in state.values()),26)
        with self.assertRaises(ValueError):adapter_state(nn.Linear(2,2))

    def test_roundtrip_restores_every_named_adapter_but_never_frozen_weights(self):
        model=self.model();saved=adapter_state(model);frozen={n:p.clone() for n,p in model.named_parameters() if not p.requires_grad}
        with torch.no_grad():
            for p in adapter_parameters(model).values():p.add_(1)
        load_adapter(model,saved)
        self.assertTrue(exact_state(saved,adapter_state(model)))
        self.assertTrue(all(torch.equal(frozen[n],p) for n,p in model.named_parameters() if n in frozen))

    def test_bad_partial_or_nonfinite_state_is_rejected_before_any_copy(self):
        model=self.model();saved=adapter_state(model)
        for kind in ('missing','shape','nan'):
            changed=copy.deepcopy(saved)
            if kind=='missing':changed.pop('1.lora_B.weight')
            elif kind=='shape':changed['1.lora_B.weight']=torch.zeros(1)
            else:changed['1.lora_B.weight'].fill_(float('nan'))
            with self.subTest(kind=kind),self.assertRaises(ValueError):load_adapter(model,changed)
            self.assertTrue(exact_state(saved,adapter_state(model)))

    def test_materialized_adam_has_no_update_and_exactly_matches_lazy_first_update(self):
        torch.manual_seed(5);left=self.model();right=copy.deepcopy(left)
        a=torch.optim.AdamW(adapter_parameters(left).values(),lr=2e-6,foreach=False,fused=False)
        b=torch.optim.AdamW(adapter_parameters(right).values(),lr=2e-6,foreach=False,fused=False)
        before=adapter_state(left);report=materialize_adam(a)
        self.assertEqual(report['optimizer_updates'],0)
        self.assertTrue(exact_state(before,adapter_state(left)))
        self.assertEqual(len(a.state),4)
        self.assertTrue(all(float(s['step'])==0 and not s['exp_avg'].any() and not s['exp_avg_sq'].any() for s in a.state.values()))
        for x,y in zip(adapter_parameters(left).values(),adapter_parameters(right).values()):
            x.grad=torch.full_like(x,.25);y.grad=x.grad.clone()
        a.step();b.step()
        self.assertTrue(exact_state(adapter_state(left),adapter_state(right)))
        self.assertTrue(exact_state(a.state_dict(),b.state_dict()))
        with self.assertRaises(ValueError):materialize_adam(a)


if __name__=='__main__':unittest.main()
