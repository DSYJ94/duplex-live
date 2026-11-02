#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2026 The OpenBMB Team. All rights reserved.
# Modifications Copyright 2026 The Realtime-Venus Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Exact generate_chunk method extracted from pinned public Venus source.
# Source: inclusionAI/Realtime-Venus@4681004131ea1f1fb9aca9f0db76be322657cc79
# Full source SHA256 b7b4070cead9b88347f6e39d8333706ec007e2a832d6bcde5755063bf1453774
# Executed only with scripted tiny CPU fixtures, never native model weights.
@torch.inference_mode()
def generate_chunk(
    self,
    inputs_embeds: torch.Tensor,
    temperature: torch.Tensor,
    repetition_penalty: float,
    eos_token: Union[int, torch.Tensor],
    force_no_stop=False,
    max_new_token=500,
    min_new_tokens=0,
    past_key_values=None,
    logits_processors=None,
    text_start_pos=None,
):
    """For inputs_embeds, it should be like [bs=1, seq_len, hidden_dim], its content is like:
    |Text BOS|Spk embeds|Text-Hidden states Interleave (if applicable)|Audio BOS|
    where the last position is the audio BOS token.
    So, the first iteration in generation directly forward the model with inputs_embeds, and
    the last hidden states of the last position (Audio BOS) will be decoded to get the first audio token.
    """
    logits_warpers, logits_processors = gen_logits(
        num_code=self.config.num_audio_tokens, repetition_penalty=repetition_penalty
    )

    # We only support batch size `1` for now
    assert inputs_embeds.shape[0] == 1
    eos_token = eos_token.to(inputs_embeds.device)
    finish = torch.zeros(inputs_embeds.shape[0], device=inputs_embeds.device).bool()

    temperature = (temperature.unsqueeze(0).expand(inputs_embeds.shape[0], -1).contiguous().view(-1, 1)).to(
        inputs_embeds.device
    )

    condition_length = inputs_embeds.shape[1]

    new_tokens = torch.zeros(
        inputs_embeds.shape[0],
        max_new_token,
        self.num_vq,
        device=inputs_embeds.device,
        dtype=torch.long,
    )

    for t in range(max_new_token):
        audio_bos = False

        # If this is the first audio token, the case is special
        if t == 0:
            audio_bos = True
            inputs_embeds_ = inputs_embeds
            position_ids = torch.tensor(
                list(range(text_start_pos, text_start_pos + condition_length)),
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(0)
        else:
            # Generate the following audio tokens, it is applicable to all other cases, including second and the following calling of `generate`
            inputs_embeds_ = self.emb_code[0](new_tokens[:, t - 1 : t, 0])

            position_ids = torch.tensor(
                [text_start_pos + condition_length + t - 1],  # prefill the previous token
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(0)

        outputs: BaseModelOutputWithPast = self.model(
            position_ids=position_ids,
            # cache_position=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds_,
            use_cache=True,
            output_attentions=False,
            # return_dict=True,  # Add this to ensure returns dict with past_key_values
        )

        del position_ids
        del inputs_embeds_

        hidden_states = outputs.last_hidden_state
        past_key_values = outputs.past_key_values

        with P.cached():
            logits = torch.empty(
                hidden_states.size(0),
                hidden_states.size(1),
                self.num_audio_tokens,
                self.num_vq,
                dtype=torch.float,
                device=self.device,
            )
            for num_vq_iter in range(self.num_vq):
                x: torch.Tensor = self.head_code[num_vq_iter](hidden_states)
                logits[..., num_vq_iter] = x
                del x

        del hidden_states

        logits = logits[:, -1].float()
        logits = logits.permute(0, 2, 1)
        logits = logits.reshape(-1, logits.size(2))

        logits /= temperature

        if not audio_bos:
            input_ids_sliced = new_tokens[:, 0:t].permute(0, 2, 1)  # get previous t new tokens

            logits_token = input_ids_sliced.reshape(
                input_ids_sliced.size(0) * input_ids_sliced.size(1),
                -1,
            ).to(self.device)

            del input_ids_sliced

            for logitsProcessors in logits_processors:
                logits = logitsProcessors(logits_token, logits)

            del logits_token

        if force_no_stop or t < min_new_tokens:
            logits[:, eos_token] = -torch.inf

        scores = F.softmax(logits, dim=-1)
        del logits

        idx_next = torch.multinomial(scores, num_samples=1).to(finish.device)
        del scores

        idx_next = idx_next.view(-1, self.num_vq)

        finish_or = idx_next.eq(eos_token).any(1)
        finish.logical_or_(finish_or)

        del finish_or
        new_tokens[:, t] = idx_next

        if t == 0 and finish.any():
            break

        del idx_next

        if finish.all():
            break

    # The latest generated token is not in the range returned this time. If it is an eos token, it is not returned. If it is a normal token, it is not returned.
    genrated_input_ids = new_tokens[:, 0:t, :]

    return genrated_input_ids, past_key_values
