"""Phase-1 sampler: shards -> interleaved training sequences (docs/09 §1).

Plain words: builds the lesson sheet. Reading order mirrors the live
clock: for every 80 ms beat, first the sound-token the model HEARD, then
the token it should have SAID (a word, <sil>, <yield>, ...). The answer
tokens are the only thing graded — the sound is given, not guessed.

Sequence layout (event stream):
  [audio_pad x ~1 per tick][emission tokens for that tick]...
Words may span several BPE tokens; they are placed consecutively at their
tick. inputs_embeds = embedding(input_ids) with audio_pad positions
overwritten by the frozen ear-features from the shard.
"""

from __future__ import annotations

from pathlib import Path

import torch

IGNORE = -100  # standard "don't grade this position"


def build_from_grid(sc_dir: Path, tok) -> dict:
    """Preferred path: rebuild the sequence from the scenario's grid
    (word emissions tokenized properly) + shard features."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tokenizer.grid import from_scenario_dir
    from tokenizer.vocab import ALL_SPECIAL, SIL

    grid = from_scenario_dir(sc_dir)
    pad_id = tok.convert_tokens_to_ids("<|audio_pad|>")
    ids, labels, audio_pos = [], [], []
    for t in grid.ticks:
        ids.append(pad_id)           # what it heard this beat
        labels.append(IGNORE)
        audio_pos.append(len(ids) - 1)
        e = t.emission
        if e == SIL:
            ei = tok.convert_tokens_to_ids(SIL)
            ids.append(ei)
            labels.append(ei)
        elif e in ALL_SPECIAL:
            ei = tok.convert_tokens_to_ids(e)
            ids.append(ei)
            labels.append(ei)
        else:  # a word: BPE tokens placed consecutively
            for w in tok(e, add_special_tokens=False)["input_ids"]:
                ids.append(w)
                labels.append(w)
    return {"ids": ids, "labels": labels, "audio_pos": audio_pos,
            "scenario_id": grid.scenario_id}


def collate(seq: dict, shard: dict, embed_layer) -> dict:
    """inputs_embeds with frozen ear-features scattered into pad slots."""
    ids = torch.tensor([seq["ids"]], dtype=torch.long)
    embeds = embed_layer(ids).clone()
    feats = shard["audio_feats"]
    n = min(len(seq["audio_pos"]), feats.shape[0])
    pos = torch.tensor(seq["audio_pos"][:n])
    embeds[0, pos] = feats[:n].to(embeds.dtype).to(embeds.device)
    labels = torch.tensor([seq["labels"]], dtype=torch.long)
    return {"inputs_embeds": embeds, "labels": labels}
