import unittest
from types import SimpleNamespace
import torch
from training.personaplex.corpus_review_batch import complete_ids,encode


class Processor:
    def __init__(self):self.tokenizer=SimpleNamespace(padding_side='right',pad_token_id=0)
    def apply_chat_template(self,messages,**kwargs):
        batch=messages if isinstance(messages[0],list)else[messages]
        lengths=[len(item[-1]['content'])for item in batch];width=max(lengths)
        masks=[[0]*(width-length)+[1]*length for length in lengths]
        return dict(input_ids=torch.tensor(masks),attention_mask=torch.tensor(masks))


class CorpusReviewBatchTest(unittest.TestCase):
    def test_actual_eos_and_padding_cannot_hide_nonpadding_tokens(self):
        self.assertEqual(complete_ids([7,2,0,0],[2],0),[7,2])
        self.assertEqual(complete_ids([7,8,9],2,0),[7,8,9])
        with self.assertRaises(ValueError):complete_ids([7,2,8],2,0)

    def test_complete_inputs_left_padded_and_only_oversize_row_excluded(self):
        processor=Processor()
        batch,valid,lengths=encode(processor,['a','long','x'*4097])
        self.assertEqual(valid,[0,1]);self.assertEqual(lengths,[1,4,4097])
        self.assertEqual(processor.tokenizer.padding_side,'left')
        self.assertEqual(batch['attention_mask'].tolist(),[[0,0,0,1],[1,1,1,1]])
