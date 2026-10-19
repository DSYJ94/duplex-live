import unittest
from evals.context_teacher_native_approval import prepared, complete_inventory


class TeacherNativeApprovalTest(unittest.TestCase):
    def test_full_approval_rejects_subsets_while_control_is_explicit(self):
        teachers=[dict(id=str(i)) for i in range(25)]
        cases=[dict(teacher=t) for t in teachers]
        self.assertTrue(complete_inventory(cases,teachers))
        with self.assertRaises(RuntimeError):complete_inventory(cases[:4],teachers)
        self.assertFalse(complete_inventory(cases[:4],teachers,diagnostic=True))
        with self.assertRaises(RuntimeError):complete_inventory(cases+[cases[0]],teachers,diagnostic=True)

    def scene(self):
        import torch
        from contextspan.model.sequence_convention import assemble_training_sequence
        class Tokenizer:
            def encode(self,text):return [21,22]
        spm=Tokenizer();prefix=torch.full((17,2),13,dtype=torch.int64);prefix[0]=3
        clean=torch.arange(17*7,dtype=torch.int64).reshape(17,7)+50
        clean[0]=torch.tensor([3,0,31,3,32,3,3])
        spans=[dict(inject_frame=3,reference='actual record')]
        codes,mask,_=assemble_training_sequence(clean,torch.ones(7,dtype=torch.bool),spans,spm)
        e=dict(clean_codes=clean,codes=torch.cat([prefix,codes],dim=1),
            text_mask=torch.cat([torch.zeros(2,dtype=torch.bool),mask]),record={'spans':spans})
        return e,prefix,spm

    def test_streamed_user_changes_no_assistant_or_text_inputs(self):
        import torch
        e,prefix,spm=self.scene();user=e['clean_codes'][None,9:17].clone()
        old=prepared(e,prefix,user,spm);user[0,6,2]+=7
        new=prepared(e,prefix,user,spm)
        self.assertTrue(torch.equal(old[:,:9],new[:,:9]))
        self.assertFalse(torch.equal(old[:,9:17],new[:,9:17]))

    def test_word_mask_prefix_and_invalid_codec_changes_are_rejected(self):
        e,prefix,spm=self.scene();user=e['clean_codes'][None,9:17].clone()
        with self.assertRaises(RuntimeError):prepared(e,prefix,user[:,:7],spm)
        user[0,0,0]=-1
        with self.assertRaises(RuntimeError):prepared(e,prefix,user,spm)
        e,prefix,spm=self.scene();user=e['clean_codes'][None,9:17].clone();e['text_mask'][0]=True
        with self.assertRaises(RuntimeError):prepared(e,prefix,user,spm)


if __name__=='__main__':unittest.main()
