from types import SimpleNamespace
import unittest
from training.personaplex.language_documents import compile_candidate,offsets,render


class CharacterTokenizer:
    def is_byte(self,identity):return False
    def encode(self,text,out_type):
        ids=[5+ord(c) for c in text]
        if out_type is int:return ids
        return SimpleNamespace(pieces=[SimpleNamespace(id=value,begin=i,end=i+1,surface=text[i]) for i,value in enumerate(ids)])


class LanguageDocumentsTest(unittest.TestCase):
    def row(self):
        return dict(id='source-1',admitted_for_training=False,source_dataset='public/test',source_revision='pinned',
                    source_licence='test',source_group_id='family-1',source_record_ids=['record-1'],content_sha256='a'*64,
                    turns=[dict(role='user',text='Why café?'),dict(role='assistant',text='Éclair.'),
                           dict(role='user',text='And then?'),dict(role='assistant',text='Explain.')])

    def test_one_shot_full_context_masks_only_original_assistant_content(self):
        row=self.row();value=compile_candidate(row,CharacterTokenizer(),tokenizer_sha256='b'*64,max_tokens=1000)
        supervised=''.join(c for c,active in zip(value['text'],value['mask']) if active)
        self.assertEqual(supervised,'Éclair.Explain.')
        self.assertIn('Why café?',value['text'])
        self.assertFalse(value['admitted_for_training']);self.assertFalse(value['semantic_qualified'])
        self.assertIsNone(value['fold']);self.assertEqual(value['structural_flags'],[])
        long=compile_candidate(row,CharacterTokenizer(),tokenizer_sha256='b'*64,max_tokens=1)
        self.assertEqual(long['tokens'],value['tokens'])
        self.assertIn('complete_document_exceeds_declared_context',long['structural_flags'])

    def test_utf8_offsets_require_matching_complete_surfaces(self):
        text='café long'
        piece=SimpleNamespace(begin=3,end=5,surface='é')
        self.assertEqual(offsets(text,piece),(3,4))
        with self.assertRaises(ValueError):offsets(text,SimpleNamespace(begin=3,end=5,surface='wrong'))

    def test_zero_width_unicode_byte_prefix_targets_belong_to_answer_not_user(self):
        class ByteTokenizer(CharacterTokenizer):
            def is_byte(self,identity):return identity==300
            def encode(self,text,out_type):
                pieces=[]
                for i,c in enumerate(text):
                    if c in 'éÉ':pieces.append(SimpleNamespace(id=300,begin=i,end=i,surface=''))
                    pieces.append(SimpleNamespace(id=5+ord(c),begin=i,end=i+1,surface=c))
                return [p.id for p in pieces] if out_type is int else SimpleNamespace(pieces=pieces)
        value=compile_candidate(self.row(),ByteTokenizer(),tokenizer_sha256='b'*64,max_tokens=1000)
        flags=[active for token,active in zip(value['tokens'],value['mask']) if token==300]
        self.assertEqual(flags,[False,True])
        self.assertEqual(value['zero_width_byte_fallback_targets'],1)
        self.assertEqual(value['boundary_crossing_tokens_excluded'],0)

    def test_no_partial_conversations_or_qualification_upgrade(self):
        row=self.row();row['admitted_for_training']=True
        with self.assertRaises(ValueError):compile_candidate(row,CharacterTokenizer(),tokenizer_sha256='b'*64,max_tokens=100)
        with self.assertRaises(ValueError):render(self.row()['turns'][:-1])


if __name__=='__main__':unittest.main()
