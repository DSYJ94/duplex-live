"""Contrast actual TRAIN-world references at the first private spoken subword.

Wrong references are negative conditioning only, never new positive speech or
word-time labels. Each reference retains its actual recorded arrival clock.
"""
import json
from evals.personaplex_pairs import contrast_candidates


class PairedGrounding:
    def __init__(self, rows, prepare, template, prefix, tokenizer):
        import torch
        selected = [r for r in rows if r['split'] == 'train' and r['scenario_id'].startswith('teacher-cf-')]
        sidecars = {r['scenario_id']:json.loads((r['conversion']/r['scenario_id']/'conversation.json').read_text()) for r in selected}
        samples = {r['scenario_id']:prepare(r) for r in selected}
        self.pairs, self.receipts = [], []
        self.conditioning = template['inputs'][:,1:9,prefix-1:prefix].cuda()
        self.prefix = prefix
        for a,b in contrast_candidates(sidecars):
            positive,_,phrase = samples[a]
            negative,_,other_phrase = samples[b]
            if not phrase or not other_phrase:
                raise RuntimeError('Approved paired world omits measured private subwords')
            frame = min(phrase)
            token = int(positive['targets'][0,0,frame])
            other = int(negative['targets'][0,0,min(other_phrase)])
            first_word = sidecars[a]['timeline']['meta']['phrase'].split()[0].casefold()
            piece = tokenizer.id_to_piece(token).replace('▁','').casefold()
            if not piece or not first_word.startswith(piece) or token < 5 or token == other:
                raise RuntimeError('Contrast must predict the first distinct native private subword')
            # Only complete, already-arrived negative conditioning is useful.
            active = negative['reference'][0,prefix:].abs().sum(-1).nonzero().flatten()
            ready = sidecars[b]['reference_arrivals'][0]['ready_at_s']
            if not active.numel() or int(active.max()) > frame or ready > frame/12.5:
                continue
            cut = prefix+frame+1
            inputs = positive['inputs'][...,:cut].clone()
            # Remove nearby teacher assistant-acoustic clues in this contrast
            # branch using captured native prefix context. User input is intact.
            first = max(prefix,cut-4)
            inputs[:,1:9,first:cut] = self.conditioning.expand(-1,-1,cut-first)
            wrong = torch.zeros_like(positive['reference'][:,:cut])
            length = min(cut,negative['reference'].shape[1])
            wrong[:,:length] = negative['reference'][:,:length]
            self.pairs.append((inputs,positive['reference'][:,:cut],wrong,
                positive['control_states'][:,:cut],token))
            self.receipts.append({'positive':a,'negative':b,'first_private_frame':frame,
                'native_target':token,'negative_actual_ready_s':ready,
                'masked_assistant_context_frames':cut-first,
                'reference_clocks_shifted':False,'negative_used_as_positive_speech':False})
        if len(self.pairs) < 4:
            raise RuntimeError('Too few causal different-value TRAIN pairs for grounding experiment')

    def loss(self, bundle, index):
        import torch.nn.functional as F
        inputs,correct,wrong,states,token = self.pairs[index % len(self.pairs)]
        def nll(reference):
            embedded = bundle.model.embed_codes(inputs)
            embedded = embedded + bundle.bridge.transform(reference).to(embedded)
            embedded = embedded + bundle.control(states).to(embedded)
            _, text = bundle.model.forward_embeddings(embedded)
            return -text[0,0,-1].log_softmax(-1)[token]
        right,other = nll(correct),nll(wrong)
        loss = right+F.relu(1.+right-other)
        return loss, {'pair':self.receipts[index % len(self.pairs)],'correct_nll':float(right.detach()),
            'wrong_nll':float(other.detach()),'margin':float((other-right).detach())}
