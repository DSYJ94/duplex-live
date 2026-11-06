# Extracted verbatim methods from OpenBMB minicpmo-utils1.0.6 stepaudio2/token2wav.py.
# Upstream distribution: https://pypi.org/project/minicpmo-utils/1.0.6/
# Original archive SHA256:3b35de45a46db449079d96fc96723df4151ab8db319888016edfd9475f9406c3
# Original source SHA256:0598c25ad9883f75d6d5ca1e9276c03adbfa0357a29442164198ea6931a06efb
# Tiny scripted CPU dependency doubles only; never import speech models or CUDA.
def set_stream_cache(self, prompt_wav):
        self.cache = self._prepare_prompt(prompt_wav)
        prompt_speech_tokens, prompt_speech_tokens_lens, spk_emb, prompt_mels, prompt_mels_lens = self.cache

        print("prompt_speech_tokens_lens", prompt_speech_tokens_lens)
        print("prompt_mels.shape", prompt_mels.shape)
        print("prompt_mels_lens", prompt_mels_lens)
        
        right_pad_speech_tokens = torch.ones(1, 3, device=prompt_speech_tokens.device, dtype=prompt_speech_tokens.dtype) * 4218
        
        # self.stream_cache = self.flow.setup_cache(
        #     torch.cat([prompt_speech_tokens, prompt_speech_tokens[:, :3]], dim=1),
        #     prompt_mels, spk_emb, n_timesteps=self.n_timesteps)
        
        stream_cache = self.flow.setup_cache(
            torch.cat([prompt_speech_tokens, right_pad_speech_tokens], dim=1),
            prompt_mels, spk_emb, n_timesteps=self.n_timesteps)

        # hift cache
        hift_cache_dict = dict(
            mel = torch.zeros(1, prompt_mels.shape[2], 0, device='cuda'), 
            source = torch.zeros(1, 1, 0, device='cuda'),
            speech = torch.zeros(1, 0, device='cuda'),
        )

        return stream_cache, hift_cache_dict

def stream(self, generated_speech_tokens, prompt_wav, last_chunk=False, return_waveform=False):
        if self.cache is None:
            self.cache = self._prepare_prompt(prompt_wav)
        prompt_speech_tokens, prompt_speech_tokens_lens, spk_emb, prompt_mels, prompt_mels_lens = self.cache

        generated_speech_tokens = torch.tensor([generated_speech_tokens], dtype=torch.int32, device='cuda')
        generated_speech_tokens_lens = torch.tensor([generated_speech_tokens.shape[1]], dtype=torch.int32, device='cuda')

        if self.stream_cache is None:
            raise ValueError("stream_cache is not set")

        with torch.amp.autocast("cuda", dtype=torch.float16 if self.float16 else torch.float32):
            chunk_mel, self.stream_cache = self.flow.inference_chunk(
                token=generated_speech_tokens,
                spk=spk_emb,
                cache=self.stream_cache,
                last_chunk=last_chunk,
                n_timesteps=self.n_timesteps,
            )
        if self.stream_cache['estimator_att_cache'].shape[4] > (prompt_mels.shape[1] + 100):
            self.stream_cache['estimator_att_cache'] = torch.cat([
                self.stream_cache['estimator_att_cache'][:, :, :, :, :prompt_mels.shape[1]],
                self.stream_cache['estimator_att_cache'][:, :, :, :, -100:],
            ], dim=4)

        # bug fix - 20260107
        # 同样截断 conformer_att_cache，防止无限增长导致 position embedding 不匹配
        if self.stream_cache['conformer_att_cache'].shape[3] > (prompt_mels.shape[1] + 100):
            self.stream_cache['conformer_att_cache'] = torch.cat([
                self.stream_cache['conformer_att_cache'][:, :, :, :prompt_mels.shape[1], :],
                self.stream_cache['conformer_att_cache'][:, :, :, -100:, :],
            ], dim=3)

        # vocoder cache
        hift_cache_mel = self.hift_cache_dict['mel']
        hift_cache_source = self.hift_cache_dict['source']
        hift_cache_speech = self.hift_cache_dict['speech']
        mel = torch.concat([hift_cache_mel, chunk_mel], dim=2)

        speech, source = self.hift(mel, hift_cache_source)

        # overlap speech smooth
        if hift_cache_speech.shape[-1] > 0:
            speech = fade_in_out(speech, hift_cache_speech, self.speech_window)

        # 检测是否是第一个 chunk（没有有效的 speech cache）
        is_first_chunk = hift_cache_speech.shape[-1] == 0

        # update vocoder cache
        self.hift_cache_dict = dict(
            mel = mel[..., -self.mel_cache_len:].clone().detach(),
            source = source[:, :, -self.source_cache_len:].clone().detach(),
            speech = speech[:, -self.source_cache_len:].clone().detach(),
        )

        if not last_chunk:
            if is_first_chunk:
                # 第一个 chunk：截断尾部，在开头添加静音来补偿长度
                # 不做淡入，保持原始音频内容不变
                silence_padding = torch.zeros(1, self.source_cache_len, device=speech.device)
                speech = torch.cat([silence_padding, speech[:, :-self.source_cache_len]], dim=1)
            else:
                # 后续 chunk：正常截断尾部（由下一个 chunk 的 fade_in 补偿）
                speech = speech[:, :-self.source_cache_len]

        wav_np = speech.cpu().numpy()
        if return_waveform:
            return wav_np

        # Clip to [-1, 1] to avoid overflow, then scale to int16
        wav_np = np.clip(wav_np, -1.0, 1.0)
        wav_int16 = (wav_np * 32767.0).astype('<i2')  # 16-bit little-endian PCM
        pcm_bytes = wav_int16.tobytes()
        return pcm_bytes
