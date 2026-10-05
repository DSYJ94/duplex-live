"""Listening gallery for the synthetic factory (localhost, zero deps).

Audition everything that involves voice samples as the pipeline runs:
rendered batches (mixed / stems / every utterance with its WER) and the
voice pool (each reference clip with its per-voice WER from the latest
filter reports). Reads the local copies synced from Isambard — refresh
the page after a sync to see new state.

Sync (from the repo root, after each render/filter step on the cluster):
  rsync -a u6si.aip2.isambard:/projects/u6si/duplex-live/runs/synth/ runs/synth/
  rsync -a u6si.aip2.isambard:/projects/u6si/duplex-live/runs/synth-v02/ runs/synth-v02/
  rsync -a u6si.aip2.isambard:/projects/u6si/duplex-live/models/voices/ runs/voices/
  rsync -a u6si.aip2.isambard:'<assets-dir>/*.wav' runs/voices/anchors/   # 3 anchors

Run:  python -m viz.listen [--runs runs] [--port 8765]
      -> http://localhost:8765
"""

from __future__ import annotations

import argparse
import html
import json
import mimetypes
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

ANCHORS = ["assistant", "user", "background"]  # MiniCPM-o asset ref clips

CSS = """
:root { --bg:#0f1115; --card:#171a21; --line:#2a2f3a; --fg:#e6e9ef;
        --dim:#8b93a3; --ok:#34d399; --bad:#f87171; --warn:#fbbf24; }
* { box-sizing:border-box; }
body { background:var(--bg); color:var(--fg); font:14px/1.5 -apple-system,
       'Segoe UI', Roboto, sans-serif; margin:0; padding:24px; }
a { color:#7ab8ff; text-decoration:none; } a:hover { text-decoration:underline; }
h1 { font-size:20px; margin:0 0 4px; } h2 { font-size:16px; margin:28px 0 8px; }
.sub { color:var(--dim); font-size:12.5px; margin-bottom:20px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px;
        padding:14px 16px; margin:12px 0; }
.badge { display:inline-block; padding:1px 9px; border-radius:99px; font-size:11.5px;
         font-weight:600; margin-right:6px; }
.b-pass { background:rgba(52,211,153,.15); color:var(--ok); }
.b-fail { background:rgba(248,113,113,.15); color:var(--bad); }
.b-dim  { background:rgba(139,147,163,.15); color:var(--dim); }
.issues { color:var(--bad); font-size:12.5px; margin:4px 0; }
.meta { color:var(--dim); font-size:12px; margin:2px 0; }
table { border-collapse:collapse; width:100%; margin-top:8px; font-size:12.5px; }
th, td { text-align:left; padding:4px 8px; border-bottom:1px solid var(--line);
         vertical-align:middle; }
th { color:var(--dim); font-weight:500; }
audio { height:28px; width:240px; }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(340px,1fr));
        gap:12px; }
.num { font-variant-numeric:tabular-nums; }
.wer-hi { color:var(--bad); font-weight:600; }
.wer-md { color:var(--warn); }
"""


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def audio_tag(src: str) -> str:
    return f'<audio controls preload="none" src="/audio/{src}"></audio>'


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


# ---------------------------------------------------------------- pages

def page_index(runs: Path, voices_dir: Path) -> str:
    cards = []
    for mf in sorted(runs.glob("*/audio/manifest.jsonl")) + \
            sorted(runs.glob("*/manifest.jsonl")):
        audio_dir = mf.parent
        name = audio_dir.parent.name if audio_dir.name == "audio" else audio_dir.name
        rows = load_jsonl(mf)
        total_min = sum(r.get("duration_ms", 0) for r in rows) / 60000
        rep_p = audio_dir / "filter_report.jsonl"
        verdict_html = '<span class="badge b-dim">not filtered</span>'
        if rep_p.exists():
            rep = load_jsonl(rep_p)
            npass = sum(1 for r in rep if r["verdict"] == "pass")
            verdict_html = (f'<span class="badge b-pass">{npass}/{len(rep)} pass'
                            f'</span>')
        cards.append(
            f'<div class="card"><a href="/batch/{name}"><b>{esc(name)}</b></a>'
            f' {verdict_html}'
            f'<div class="meta">{len(rows)} scenarios · {total_min:.1f} min · '
            f'{esc(audio_dir)}</div></div>')
    evals = []
    for t1 in sorted(runs.glob("*/t1/model_out.wav")):
        ename = t1.parent.parent.name
        ntests = len([d for d in t1.parent.parent.iterdir()
                      if d.is_dir() and (d / "model_out.wav").exists()])
        evals.append(f'<div class="card"><a href="/batch/{ename}"><b>{ename}</b>'
                     f'</a> <span class="badge b-dim">eval session</span>'
                     f'<div class="meta">{ntests} tests · user_in vs model_out</div></div>')
    vcard = ""
    if (voices_dir / "voices.json").exists():
        n = len(json.loads((voices_dir / "voices.json").read_text()))
        n_anchor = sum(1 for a in ANCHORS if (voices_dir / "anchors" / f"{a}.wav").exists())
        vcard = (f'<div class="card"><a href="/voices"><b>voice pool</b></a>'
                 f'<div class="meta">{n} LibriSpeech + {n_anchor} anchors · '
                 f'{esc(voices_dir)}</div></div>')
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>duplex-live · listen</title><style>{CSS}</style></head><body>
<h1>🏭 Synthetic factory — listening gallery</h1>
<div class="sub">Local copies synced from Isambard · refresh after each
<code>rsync</code> · audio 24 kHz (pool refs 16 kHz)</div>
{''.join(cards)}{''.join(evals)}{vcard}
</body></html>"""


def page_batch(runs: Path, name: str) -> str | None:
    audio_dir = runs / name / "audio"
    if not (audio_dir / "manifest.jsonl").exists():
        audio_dir = runs / name
        if not (audio_dir / "manifest.jsonl").exists():
            return _page_eval_session(runs, name)
    rows = load_jsonl(audio_dir / "manifest.jsonl")
    rep = {r["scenario_id"]: r for r in load_jsonl(audio_dir / "filter_report.jsonl")} \
        if (audio_dir / "filter_report.jsonl").exists() else {}

    blocks = []
    for row in rows:
        sc = audio_dir / row["dir"]
        r = rep.get(row["scenario_id"], {})
        verdict = r.get("verdict")
        badge = ('<span class="badge b-pass">pass</span>' if verdict == "pass"
                 else '<span class="badge b-fail">fail</span>' if verdict == "fail"
                 else '<span class="badge b-dim">not filtered</span>')
        voices = " · ".join(f"{ch}→{esc(v)}" for ch, v in
                            sorted(row.get("voices", {}).items()))
        issues = "".join(f"<div>{esc(i)}</div>" for i in r.get("issues", []))
        players = " ".join(
            f'<div class="meta">{esc(p)} {audio_tag(quote(str(sc.relative_to(runs)) + "/" + p))}</div>'
            for p in (["model.wav", "mixed.wav"]
                      + sorted(x.name for x in sc.glob("stem_*.wav")))
            if (sc / p).exists())

        utt_rows = []
        tl_p = sc / "timeline.actual.json"
        detail = (r.get("asr_detail") or {}).get("per_utterance", [])
        if tl_p.exists():
            tl = json.loads(tl_p.read_text())
            speech = [e for e in tl["events"] if e["text"] and e["kind"] != "marker"]
            vmap = tl.get("meta", {}).get("voices", {})
            for i, ev in enumerate(speech):
                wav = f"{i:02d}_{ev['ch']}.wav"
                u = detail[i] if i < len(detail) else {}
                wer = u.get("wer")
                wer_html = (f'<span class="num wer-hi">{wer}</span>' if wer and wer > 0.3
                            else f'<span class="num wer-md">{wer}</span>' if wer
                            else '<span class="num">–</span>')
                note = "; ".join(x for x in [ev.get("note", ""),
                                             u.get("gate_fail", "")] if x)
                utt_rows.append(
                    f"<tr><td class='num'>{ev['actual_start_ms']}ms</td>"
                    f"<td>{esc(ev['ch'])}</td><td>{esc(ev['kind'])}</td>"
                    f"<td>{esc(ev.get('voice') or vmap.get(ev['ch']) or '')}</td>"
                    f"<td>{esc(ev['text'][:90])}</td><td>{wer_html}</td>"
                    f"<td>{esc(u.get('hyp', '')[:70])}</td>"
                    f"<td>{audio_tag(quote(str(sc.relative_to(runs)) + '/utterances/' + wav))}</td>"
                    f"<td class='meta'>{esc(note[:60])}</td></tr>")
        utt_table = (f"<table><tr><th>t</th><th>ch</th><th>kind</th><th>voice</th>"
                     f"<th>text</th><th>WER</th><th>whisper heard</th>"
                     f"<th>listen</th><th>note</th></tr>{''.join(utt_rows)}</table>"
                     if utt_rows else "")

        # model-audio manifests (render_model_audio) carry no class/duration
        cls_html = (f' <span class="badge b-dim">{esc(row["scenario_class"])}</span>'
                    if "scenario_class" in row else "")
        dur_html = (f'<span class="meta num">· {(row["duration_ms"] / 1000):.1f}s · '
                    f'overlap {row.get("overlap_ms", 0)}ms</span>'
                    if "duration_ms" in row else "")
        blocks.append(
            f'<div class="card" id="{esc(row["scenario_id"])}">'
            f'<b>{esc(row["scenario_id"])}</b>{cls_html} {badge}{dur_html}'
            f'<div class="meta">voices: {voices or "anchors (v0.1 batch)"}</div>'
            f'<div class="issues">{issues}</div>{players}{utt_table}</div>')

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>{esc(name)} · listen</title><style>{CSS}</style></head><body>
<a href="/">← batches</a>
<h1>{esc(name)}</h1>
<div class="sub">{len(rows)} scenarios · click a player to audition; the
per-utterance table shows exactly what the ASR judge heard</div>
{''.join(blocks)}
</body></html>"""


def _page_eval_session(runs: Path, name: str) -> str | None:
    """Harness live-session layout: <root>/<name>/t*/{user_in,model_out}.wav"""
    root = runs / name
    tests = sorted(d for d in root.iterdir()
                   if d.is_dir() and (d / "model_out.wav").exists()
                   and d.name.startswith("t"))
    if not tests:
        return None
    blocks = []
    for t in tests:
        players = []
        for wav in ["user_in.wav", "model_out.wav"]:
            p = t / wav
            if p.exists():
                players.append(
                    f'<div class="meta">{wav} {audio_tag(quote(str(p.relative_to(runs))))}</div>')
        blocks.append(f'<div class="card"><b>{t.name}</b> '
                      f'<span class="badge b-dim">eval session</span>'
                      f'{"".join(players)}</div>')
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>{esc(name)} · listen</title><style>{CSS}</style></head><body>
<a href="/">← batches</a>
<h1>{esc(name)}</h1>
<div class="sub">Live eval session — stimulus (user_in) vs system output
(model_out) per test; scored by evals/harness (see harness_report.json)</div>
{''.join(blocks)}
</body></html>"""


def page_voices(runs: Path, voices_dir: Path) -> str:
    entries = []
    vj = voices_dir / "voices.json"
    if vj.exists():
        entries = json.loads(vj.read_text())
    wers = {}
    for mf in sorted(runs.glob("*/audio/filter_report.jsonl")) + \
            sorted(runs.glob("*/filter_report.jsonl")):
        for r in load_jsonl(mf):
            for k, w in ((r.get("asr_detail") or {}).get("wer_by_voice") or {}).items():
                if w is not None:
                    wers.setdefault(k, []).append(w)
    rows = []
    for a in ANCHORS:
        wav = Path("anchors") / f"{a}.wav"
        if (voices_dir / wav).exists():
            rows.append((a, "anchor (MiniCPM-o asset)", wav, "Apache-2.0"))
    for e in entries:
        rows.append((e["key"], f"LibriSpeech {e.get('sex', '?')} · "
                    f"{e['dur_ms'] / 1000:.1f}s", Path(e["key"] + ".wav")
                    if (voices_dir / (e["key"] + ".wav")).exists() else None,
                    "CC-BY-4.0"))
    trs = []
    for key, desc, wav, lic in rows:
        w_list = wers.get(key, [])
        w = sum(w_list) / len(w_list) if w_list else None
        w_html = (f'<span class="num wer-hi">{w:.3f}</span>' if w and w > 0.15
                  else f'<span class="num">{w:.3f}</span>' if w is not None
                  else '<span class="num">untested</span>')
        player = audio_tag(quote("__voices__/" + str(wav))) if wav else ""
        trs.append(f"<tr><td><b>{esc(key)}</b></td><td>{esc(desc)}</td>"
                   f"<td>{w_html}</td><td>{player}</td>"
                   f"<td class='meta'>{esc(lic)}</td></tr>")
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>voice pool · listen</title><style>{CSS}</style></head><body>
<a href="/">← batches</a>
<h1>Voice pool</h1>
<div class="sub">Reference clips the renderer clones from · mean WER across
filtered batches (how reliably that voice round-trips through TTS+ASR)</div>
<table><tr><th>key</th><th>source</th><th>WER</th><th>listen</th><th>license</th></tr>
{''.join(trs)}</table>
</body></html>"""


# ---------------------------------------------------------------- server

class Handler(BaseHTTPRequestHandler):
    runs: Path = Path("runs")
    voices: Path = Path("runs/voices")

    def log_message(self, fmt, *args):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path == "/":
            self._send(200, page_index(self.runs, self.voices).encode(),
                       "text/html; charset=utf-8")
            return
        if path == "/voices":
            self._send(200, page_voices(self.runs, self.voices).encode(),
                       "text/html; charset=utf-8")
            return
        if path.startswith("/batch/"):
            page = page_batch(self.runs, path.removeprefix("/batch/").strip("/"))
            if page is None:
                self._send(404, b"batch not found", "text/plain")
            else:
                self._send(200, page.encode(), "text/html; charset=utf-8")
            return
        if path.startswith("/audio/"):
            rel = Path(path.removeprefix("/audio/"))
            root = self.voices if rel.parts and rel.parts[0] == "__voices__" else self.runs
            if rel.parts and rel.parts[0] == "__voices__":
                rel = Path(*rel.parts[1:])
            f = (root / rel).resolve()
            if (not str(f).startswith(str(root.resolve())) or not f.is_file()):
                self._send(404, b"not found", "text/plain")
                return
            ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
            self._send(200, f.read_bytes(), ctype)
            return
        self._send(404, b"not found", "text/plain")


class DualStackServer(ThreadingHTTPServer):
    """Listens on ::1 AND 127.0.0.1 with one socket. Some browsers
    resolve localhost to ::1 first and refuse to fall back
    (ERR_CONNECTION_REFUSED, owner report 2026-09-24) — the old
    IPv4-only bind locked exactly those browsers out."""

    # class attribute: the socket is created from it in __init__,
    # BEFORE server_bind runs
    address_family = socket.AF_INET6
    daemon_threads = True

    def server_bind(self):
        try:
            self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        except OSError:
            pass  # dual-stack unsupported here; ::1 still served
        super().server_bind()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--voices", default=None,
                    help="voice pool dir (default <runs>/voices)")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    Handler.runs = Path(args.runs)
    Handler.voices = Path(args.voices) if args.voices else Path(args.runs) / "voices"
    url = f"http://localhost:{args.port}"
    print(f"listening gallery: {url}  (runs={Handler.runs}, voices={Handler.voices})")
    try:
        DualStackServer(("::", args.port), Handler).serve_forever()
    except OSError:  # no IPv6 on this machine — plain IPv4 loopback
        ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
