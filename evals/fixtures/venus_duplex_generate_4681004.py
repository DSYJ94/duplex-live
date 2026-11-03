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
# Exact streaming_generate method extracted from pinned public Venus source.
# Source: inclusionAI/Realtime-Venus@4681004131ea1f1fb9aca9f0db76be322657cc79
# Full source SHA256 b7b4070cead9b88347f6e39d8333706ec007e2a832d6bcde5755063bf1453774
# Executed only with scripted tiny CPU fixtures, never native model weights.
def streaming_generate(
        self,
        prompt_wav_path=None,
        max_new_speak_tokens_per_chunk=20,
        decode_mode: str = "sampling",
        temperature=0.7,
        top_k=100,
        top_p=0.8,
        listen_prob_scale=1.0,
        listen_top_k=None,
        text_repetition_penalty=1.05,
        text_repetition_window_size=512,
    ):
        start_time = time.time()

        if self.is_session_stop_set() or self.is_break_set():
            return {
                "is_listen": True,
                "text": "",
                "audio_waveform": self._generate_silence_waveform(),
                "end_of_turn": True,
                "current_time": self.audio_chunk_idx,
                "cost_llm": 0.0,
                "cost_tts_prep": 0.0,
                "cost_tts": 0.0,
                "cost_token2wav": 0.0,
                "cost_all": time.time() - start_time,
                "n_tokens": 0,
                "n_tts_tokens": 0,
            }

        # check if there are pending logits to process
        if not hasattr(self, "pending_logits") or self.pending_logits is None:
            return {
                "is_listen": True,
                "text": "",
                "audio_waveform": self._generate_silence_waveform(),
                "end_of_turn": False,
                "current_time": self.audio_chunk_idx,
                "cost_llm": 0.0,
                "cost_tts_prep": 0.0,
                "cost_tts": 0.0,
                "cost_token2wav": 0.0,
                "cost_all": time.time() - start_time,
                "n_tokens": 0,
                "n_tts_tokens": 0,
            }

        # use pending logits generated in streaming_prefill
        logits = self.pending_logits
        self.pending_logits = None

        # Force listen: check if we should force listen for first N calls
        force_listen = self._streaming_generate_count < self.force_listen_count
        self._streaming_generate_count += 1

        total_hidden_in_unit = []
        total_ids_in_unit = []
        current_time = self.audio_chunk_idx
        is_listen = False
        end_of_turn = False

        llm_start_time = time.time()

        for j in range(max_new_speak_tokens_per_chunk):
            if j == max_new_speak_tokens_per_chunk - 1:
                if self.ls_mode == "explicit":
                    self.decoder.feed(self.decoder.embed_token(self.chunk_eos_token_id))
                    self.total_ids.append(self.chunk_eos_token_id)
                    break

            if force_listen:
                last_id = torch.tensor([self.listen_token_id], dtype=torch.long, device=self.device)
            else:
                last_id = self.decoder.decode(
                    logits=logits,
                    mode=decode_mode,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    listen_top_k=listen_top_k,
                    listen_prob_scale=listen_prob_scale,
                    text_repetition_penalty=text_repetition_penalty,
                    text_repetition_window_size=text_repetition_window_size,
                )

                # if current turn not ended, not allowed to listen (only check when not force_listen)
                if last_id.item() == self.listen_token_id and (not self.current_turn_ended):
                    last_id = torch.tensor([self.tts_bos_token_id], dtype=torch.long, device=self.device)

            self.total_ids.append(last_id.item())

            is_listen = last_id.item() == self.listen_token_id

            # termination condition detection
            if last_id.item() in self.chunk_terminator_token_ids:
                if self.ls_mode == "explicit":
                    logits, _ = self.decoder.feed(self.decoder.embed_token(last_id.item()), return_logits=True)
                break
            else:
                # normal speak
                self.current_turn_ended = False

                if last_id.item() in self.chunk_speak_token_ids:
                    pass
                else:
                    self.res_ids.append(last_id.item())
                    self.speak_count += 1

                logits, hidden = self.decoder.feed(self.decoder.embed_token(last_id.item()), return_logits=True)

                assert len(hidden.shape) == 3
                assert hidden.shape[0] == 1
                assert hidden.shape[1] == 1

                end_of_turn = last_id.item() in self.turn_terminator_token_ids

                if end_of_turn:
                    self.current_turn_ended = True

                if j != 0:
                    total_hidden_in_unit.append([last_id.item(), hidden, end_of_turn])
                    total_ids_in_unit.append(last_id.item())

        # Prefill </unit> token
        unit_end_id = self.tokenizer.convert_tokens_to_ids("</unit>")
        self.decoder.feed(self.decoder.embed_token(unit_end_id))
        self.total_ids.append(unit_end_id)

        # calculate generated text (for sliding window context preserve, filter out special tokens)
        generated_text = self.tokenizer.decode(total_ids_in_unit, skip_special_tokens=True) if total_ids_in_unit else ""

        # sliding window: register unit end, and check if sliding window is needed
        input_type = self.current_mode.lower() if self.current_mode else "audio"

        self.decoder.register_unit_end(
            input_type=input_type,
            generated_tokens=total_ids_in_unit,
            is_listen=is_listen,
            generated_text=generated_text,
        )
        # select sliding window method based on sliding window mode
        if self.decoder._window_config.sliding_window_mode == "context":
            self.decoder.enforce_window_with_context()
        elif self.decoder._window_config.sliding_window_mode == "basic":
            self.decoder.enforce_window()

        llm_end_time = time.time()

        if is_listen:
            self.total_hidden.append([])
            return {
                "is_listen": True,
                "text": "",
                "audio_waveform": self._generate_silence_waveform(),
                "end_of_turn": False,
                "current_time": current_time,
                "cost_llm": llm_end_time - llm_start_time,
                "cost_tts_prep": 0.0,
                "cost_tts": 0.0,
                "cost_token2wav": 0.0,
                "cost_all": time.time() - start_time,
                "n_tokens": len(total_ids_in_unit),
                "n_tts_tokens": 0,
            }

        self.total_hidden.append(total_hidden_in_unit)
        text = generated_text  # reuse already calculated text

        if not self.generate_audio:
            return {
                "is_listen": False,
                "text": text,
                "audio_waveform": None,
                "end_of_turn": end_of_turn,
                "current_time": current_time,
                "cost_llm": llm_end_time - llm_start_time,
                "cost_tts_prep": 0.0,
                "cost_tts": 0.0,
                "cost_token2wav": 0.0,
                "cost_all": time.time() - start_time,
                "n_tokens": len(total_ids_in_unit),
                "n_tts_tokens": 0,
            }

        # TTS generate
        tts_start_time = time.time()
        tts_prep_start_time = time.time()
        tts_condition = self._convert_results_to_tts_input(total_hidden_in_unit)
        tts_prep_end_time = time.time()

        max_token_per_chunk = 25 + 1
        min_token_per_chunk = 25 + 1

        if end_of_turn:
            min_token_per_chunk = 0
        force_flush = False
        if self.tts_text_start_pos == 0:  # this is the start of the turn
            min_token_per_chunk = 0  # allow decoding <1s audio
            force_flush = True

        if self.tts_current_turn_start_time is None:
            self.tts_current_turn_start_time = current_time

        new_tokens, old_kv = self.model.tts.generate_chunk(
            inputs_embeds=tts_condition,
            temperature=self.tts_temperature,
            repetition_penalty=self.tts_repetition_penalty,
            eos_token=self.tts_eos_token,
            force_no_stop=False,
            max_new_token=max_token_per_chunk,
            min_new_tokens=min_token_per_chunk,
            past_key_values=self.tts_past_key_values,
            logits_processors=self.tts_logits_processors,
            text_start_pos=self.tts_text_start_pos,
        )

        tts_end_time = time.time()

        # update TTS state (note: token2wav reset must be after audio generation, otherwise tokens in buffer will be lost)
        if end_of_turn:
            self.tts_text_start_pos = 0
            self.tts_past_key_values = None
            self.tts_current_turn_start_time = None
        else:
            self.tts_past_key_values = old_kv
            self.tts_text_start_pos += tts_condition.shape[1] + new_tokens.shape[1]

        # token2wav generation (must be before reset, otherwise tokens in the last but second chunk will be lost)
        token2wav_start_time = time.time()
        audio_waveform = self._generate_waveform_from_tokens(
            new_tokens, prompt_wav_path, end_of_turn, force_flush=force_flush
        )
        token2wav_end_time = time.time()

        # reset token2wav state after audio generation, ensure all tokens in buffer are processed
        if end_of_turn:
            self._reset_token2wav_for_new_turn()

        end_time = time.time()

        return {
            "is_listen": False,
            "text": text,
            "audio_waveform": audio_waveform,
            "end_of_turn": end_of_turn,
            "current_time": current_time,
            "cost_llm": llm_end_time - llm_start_time,
            "cost_tts_prep": tts_prep_end_time - tts_prep_start_time,
            "cost_tts": tts_end_time - tts_start_time,
            "cost_token2wav": token2wav_end_time - token2wav_start_time,
            "cost_all": end_time - start_time,
            "n_tokens": len(total_ids_in_unit),
            "n_tts_tokens": new_tokens.numel(),
        }
