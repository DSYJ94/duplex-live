"""Voice pool for the synthetic factory (Issue #3).

The v0 renderer had 3 fixed reference clips (MiniCPM-o snapshot assets).
Hours-scale generation needs speaker diversity so the model cannot key a
role to one timbre. This module stages a pool of diverse, permissively
licensed reference clips:

  - anchors: the 3 proven MiniCPM-o asset clips (Apache-2.0 repo assets)
  - pool: one 6–10 s clip per speaker from LibriSpeech test-clean
    (CC-BY-4.0), 16 kHz mono wav — exactly what MiniCPM-o voice cloning
    expects (refs loaded at 16 kHz)

Run on the Isambard login node inside the container (python 3.12, has
soundfile via pylibs; internet available there):

  apptainer exec --bind /projects ... $SIF \
      python -m data.synthetic.voices --dl /projects/u6si/duplex-live

Outputs under $DL/models/voices/:
  librispeech/<spk>.wav        one clip per speaker
  voices.json                  [{key, path, sex, dur_ms, license, source}]
  LICENSES.md                  attribution notes (CC-BY-4.0 text pointer)

render.py loads voices.json when present (--voices path override) and
assigns distinct voices per scenario channel deterministically.
"""

from __future__ import annotations

import argparse
import io
import json
import tarfile
import urllib.request
from pathlib import Path

URL = "https://us.openslr.org/resources/12/test-clean.tar.gz"
MIN_S, MAX_S = 6.0, 10.0
LICENSE = "LibriSpeech test-clean, CC-BY-4.0 (openslr.org/12)"


def stage_tar(voices_dir: Path) -> Path:
    tar = voices_dir / "test-clean.tar.gz"
    if not tar.exists():
        print(f"downloading {URL} -> {tar} (~350 MB)")
        urllib.request.urlretrieve(URL, tar)
    return tar


def extract(tar: Path, out_dir: Path) -> list[dict]:
    import soundfile as sf

    out_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    with tarfile.open(tar) as tf:
        by_spk: dict[str, list[tarfile.TarInfo]] = {}
        speakers_raw = None
        for m in tf:
            if m.name.endswith("SPEAKERS.txt"):
                speakers_raw = m
                continue
            if m.name.endswith(".flac") and "/test-clean/" in m.name:
                parts = Path(m.name).parts
                spk = parts[-3] if len(parts) >= 3 else None
                if spk and spk.isdigit():
                    by_spk.setdefault(spk, []).append(m)
        if speakers_raw is None:
            raise RuntimeError("SPEAKERS.txt not found in tar")
        sex_of: dict[str, str] = {}
        with io.TextIOWrapper(tf.extractfile(speakers_raw), encoding="utf-8") as fh:
            for line in fh:
                if "|" in line:
                    cells = [c.strip() for c in line.split("|")]
                    if len(cells) >= 2 and cells[0].isdigit():
                        sex_of[cells[0]] = cells[1]

        for spk in sorted(by_spk):
            # decode up to a few candidates, keep the first in [MIN_S, MAX_S]
            # (flac members are compressed — no cheap duration from size)
            picked = None
            for m in by_spk[spk][:6]:
                data = tf.extractfile(m).read()
                audio, sr = sf.read(io.BytesIO(data), dtype="float32")
                dur = len(audio) / sr
                if MIN_S <= dur <= MAX_S:
                    picked = (m, audio, sr, dur)
                    break
                if picked is None or abs(dur - 8.0) < abs(picked[3] - 8.0):
                    picked = (m, audio, sr, dur)
            if picked is None:
                continue
            m, audio, sr, dur = picked
            if not (4.0 <= dur <= 12.0):
                continue
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            wav = out_dir / f"ls{spk}.wav"
            sf.write(wav, audio, sr)
            entries.append({
                "key": f"ls{spk}",
                "path": str(wav),
                "sex": sex_of.get(spk, "?"),
                "dur_ms": round(dur * 1000),
                "license": LICENSE,
                "source_utt": m.name,
            })
            print(f"  ls{spk} ({sex_of.get(spk, '?')}) {dur:.1f}s <- {m.name}")
    return entries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dl", default="/projects/u6si/duplex-live")
    args = ap.parse_args()

    voices_dir = Path(args.dl) / "models" / "voices"
    voices_dir.mkdir(parents=True, exist_ok=True)
    tar = stage_tar(voices_dir)
    entries = extract(tar, voices_dir / "librispeech")

    with open(voices_dir / "voices.json", "w") as f:
        json.dump(entries, f, indent=2)
    (voices_dir / "LICENSES.md").write_text(
        "# Voice pool licenses\n\n"
        "- MiniCPM-o asset clips (system_ref_audio*.wav, HT_ref_audio.wav): "
        "from the Apache-2.0 openbmb/MiniCPM-o-4_5 snapshot.\n"
        f"- {LICENSE} — see https://www.openslr.org/12. Attributed per-clip "
        "in voices.json (source_utt, speaker key).\n")
    print(f"{len(entries)} voices -> {voices_dir / 'voices.json'}")


if __name__ == "__main__":
    main()
