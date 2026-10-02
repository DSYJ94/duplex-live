"""D1: the decisive diagnosis — is the turn-start signal IN the trunk
hidden state at all? (2026-10-02, owner-approved; the check that
arbitrates between root-cause theories BEFORE any more training.)

Three readout attacks on the same assumption failed identically (gate
MLP rounds 1-3, linear head-DAgger, head-DAgger CE): per-lesson pole
flips, <speak> substitution. When readouts keep failing the same way,
the assumption "the trunk state contains the signal" is the bug to
test — on the 4,374 SAVED rollout states, with no model, in minutes.

Tasks (all on runs/dagger-states/scale1.pt):
  T1 word-vs-quiet from h  (binary; gold < n_old = word/speech-content
     tick, else duplex-control tick) — linear vs MLP.
  T2 speech-vs-quiet from h (binary; gold word OR <speak> = 1 vs
     <sil>/timing = 0) — the perceptually-relevant version given the
     <speak> semantics.
  T3 lesson-identity from h (8-way) — if the state predicts WHICH
     lesson far better than WHAT should happen, history/voice dominates
     the state (the history-basin theory of the five-point law).

Splits (the whole point is the contrast):
  - per-lesson 5-fold: is there signal at all, locally?
  - pooled 5-fold: does it transfer across the mixed distribution?
  - leave-one-lesson-out (LOO): THE number — does a readout trained on
    7 lessons work on the 8th? (gate1 said per-lesson yes, pooled no.)

Verdict thresholds (printed, plain language):
  MLP-LOO strong (AUC >= 0.75) and >> linear-LOO -> signal present,
    nonlinearly: LoRA-readout training is viable; GRPO/RL could sit on
    top later.
  Per-lesson strong but LOO ~ chance (<0.65): signal is lesson-local
    (history basins) -> readout training is FOREVER moot; only
    objective-level fixes (self-history fine-tuning, D2) can help.
  Both weak everywhere: no signal in states at all -> same conclusion
    as row 2, via the stronger form.

Pure torch (container has no sklearn); AUC = Mann-Whitney from ranks.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def auc(scores: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney AUC; ties get 0.5."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks within tie groups
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2)
                 / (n_pos * n_neg))


class LinearProbe(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.lin = nn.Linear(d, 1)

    def forward(self, x):
        return self.lin(x).squeeze(-1)


class MLPProbe(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_probe(model: nn.Module, xtr: torch.Tensor, ytr: torch.Tensor,
                steps: int = 400, batch: int = 256) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    n_pos = float((ytr == 1).sum())
    n_neg = float((ytr == 0).sum())
    pos_weight = torch.tensor(n_neg / max(1.0, n_pos), device=ytr.device)
    for _ in range(steps):
        sel = torch.randint(0, xtr.shape[0], (min(batch, xtr.shape[0]),),
                            device=xtr.device)
        logits = model(xtr[sel])
        loss = F.binary_cross_entropy_with_logits(
            logits, ytr[sel].float(), pos_weight=pos_weight)
        opt.zero_grad()
        loss.backward()
        opt.step()


@torch.no_grad()
def eval_probe(model: nn.Module, x: torch.Tensor, y: torch.Tensor):
    s = torch.sigmoid(model(x)).cpu().numpy()
    yb = y.cpu().numpy()
    a = auc(s, yb)
    pred = (s >= 0.5).astype(int)
    tpr = pred[yb == 1].mean() if (yb == 1).any() else float("nan")
    tnr = (pred[yb == 0] == 0).mean() if (yb == 0).any() else float("nan")
    bal = (tpr + tnr) / 2
    return a, bal


def run_binary(name: str, y_all: torch.Tensor, H: torch.Tensor,
               sids: list, dev) -> dict:
    """Per-lesson / pooled / LOO x {linear, MLP}. Returns the table."""
    out = {}
    y = y_all.to(dev)
    lessons = sorted(set(sids))
    for model_name, cls in (("linear", LinearProbe), ("mlp", MLPProbe)):
        res = {}
        # per-lesson 5-fold
        per = []
        for sid in lessons:
            m = np.array([s == sid for s in sids])
            idx = np.where(m)[0]
            yl = y[torch.tensor(idx, device=dev)]
            if yl.float().std() == 0:
                per.append(float("nan"))
                continue
            fold_auc = []
            k = 5
            for f in range(k):
                te = idx[f::k]
                tr = np.concatenate([idx[f2::k] for f2 in range(k)
                                     if f2 != f])
                mo = cls(H.shape[1]).to(dev)
                train_probe(mo, H[torch.tensor(tr, device=dev)],
                            y[torch.tensor(tr, device=dev)])
                a, _ = eval_probe(mo, H[torch.tensor(te, device=dev)],
                                  y[torch.tensor(te, device=dev)])
                fold_auc.append(a)
            per.append(float(np.nanmean(fold_auc)))
        res["per-lesson(mean)"] = float(np.nanmean(per))
        res["per-lesson(each)"] = {
            s: (None if np.isnan(v) else round(v, 3))
            for s, v in zip(lessons, per)}
        # pooled 5-fold
        n = len(sids)
        idx = np.arange(n)
        rng = np.random.default_rng(0)
        rng.shuffle(idx)
        fold_auc = []
        for f in range(5):
            te = idx[f::5]
            tr = np.concatenate([idx[f2::5] for f2 in range(5) if f2 != f])
            mo = cls(H.shape[1]).to(dev)
            train_probe(mo, H[torch.tensor(tr, device=dev)],
                        y[torch.tensor(tr, device=dev)])
            a, _ = eval_probe(mo, H[torch.tensor(te, device=dev)],
                              y[torch.tensor(te, device=dev)])
            fold_auc.append(a)
        res["pooled"] = float(np.mean(fold_auc))
        # leave-one-lesson-out
        loo = []
        for sid in lessons:
            te = np.array([s == sid for s in sids])
            tr = ~te
            mo = cls(H.shape[1]).to(dev)
            train_probe(mo, H[torch.tensor(np.where(tr)[0], device=dev)],
                        y[torch.tensor(np.where(tr)[0], device=dev)])
            a, _ = eval_probe(mo, H[torch.tensor(np.where(te)[0], device=dev)],
                              y[torch.tensor(np.where(te)[0], device=dev)])
            loo.append(a)
        res["LOO(mean)"] = float(np.nanmean(loo))
        res["LOO(each)"] = {s: round(v, 3)
                            for s, v in zip(lessons, loo)}
        out[model_name] = res
    print(f"\n[T1/T2] {name}  (pos={int(y_all.sum())} / "
          f"neg={int((y_all == 0).sum())})", flush=True)
    for mn, res in out.items():
        print(f"  {mn:6s} per-lesson {res['per-lesson(mean)']:.3f}   "
              f"pooled {res['pooled']:.3f}   "
              f"LOO {res['LOO(mean)']:.3f}", flush=True)
        print(f"         LOO per lesson: {res['LOO(each)']}", flush=True)
        if "per-lesson(each)" in res:
            print(f"         per-lesson: {res['per-lesson(each)']}",
                  flush=True)
    return out


def run_lesson_id(H: torch.Tensor, sids: list, dev) -> None:
    """T3: how predictable is the LESSON from the state? Linear, LOO."""
    lessons = sorted(set(sids))
    lid = torch.tensor([lessons.index(s) for s in sids], device=dev)
    accs = []
    for sid in lessons:
        te = np.array([s == sid for s in sids])
        tr = ~te
        mo = nn.Linear(H.shape[1], len(lessons)).to(dev)
        opt = torch.optim.Adam(mo.parameters(), lr=1e-3, weight_decay=1e-4)
        tri = torch.tensor(np.where(tr)[0], device=dev)
        tei = torch.tensor(np.where(te)[0], device=dev)
        for _ in range(300):
            sel = tri[torch.randint(0, tri.shape[0], (256,), device=dev)]
            loss = F.cross_entropy(mo(H[sel]), lid[sel])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            p = mo(H[tei]).argmax(1)
            maj = lid[tei].mode().values
            accs.append((float((p == lid[tei]).float().mean()),
                         float((maj == lid[tei]).float().mean())))
    accs_a = np.array(accs)
    print(f"\n[T3] lesson-identity (linear, LOO): mean acc "
          f"{accs_a[:, 0].mean():.3f} vs predict-majority "
          f"{accs_a[:, 1].mean():.3f}  (chance ~0.125)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--steps", type=int, default=400)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    buf = torch.load(args.states, weights_only=False)
    H = torch.stack([h for h in buf["h"]]).to(dev).float()
    gold = torch.tensor(buf["gold"], device=dev)
    sids = list(buf["sid"])
    print(f"[diag] states {tuple(H.shape)} on {dev}, {len(set(sids))} "
          f"lessons ({time.time() - t0:.0f}s)", flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Omni-30B-A3B-Instruct")
    n_old = len(tok)
    tok_conv = tok.convert_tokens_to_ids
    sil = tok_conv("<sil>")
    speak = tok_conv("<speak>")
    timing_ids = {tok_conv(f"<t+{n}s>") for n in (1, 2, 3, 5, 10)} | \
        {tok_conv("<yield>")}
    print(f"[diag] n_old={n_old} sil={sil}", flush=True)

    y_word = (gold < n_old).long()
    y_speech = ((gold < n_old) | (gold == speak)).long()
    print(f"[diag] T1 word-vs-quiet: pos(quick check)={int(y_word.sum())}")
    run_binary("T1 word-vs-quiet (gold< n_old)", y_word, H, sids, dev)
    run_binary("T2 speech-vs-quiet (word|<speak> vs sil/timing)",
               y_speech, H, sids, dev)
    run_lesson_id(H, sids, dev)

    print("""
[diag] HOW TO READ THIS (plain language):
  per-lesson = trained and tested inside ONE lesson (is there any
               signal locally?)
  pooled     = trained on all lessons, tested on held-out states of
               the SAME lessons (does it survive the mix?)
  LOO        = trained on 7 lessons, tested on the 8th (the number
               that decides whether ANY readout training can work)
  DECISION:  MLP-LOO >= 0.75 AUC and >> linear-LOO -> LoRA-readout
             training viable; else only objective-level fixes (D2
             self-history fine-tuning); T3 >> chance confirms the
             history-basin theory of the five-point law.
""", flush=True)
    print(f"[diag] done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
