"""Layer-2 real-audio collector (docs/02-data.md "collector targets").

Fetches candidate conversational audio with per-clip provenance. Modes:

  rss    podcast RSS enclosures — the sanctioned distribution path
         (stdlib only, runs anywhere: this Mac, login node, CPU job)
  ytdlp  YouTube CC-BY subset via the yt-dlp binary (must be installed);
         verifies the video's license field == Creative Commons before
         downloading, rate-limited, no cookies/circumvention

License classes (ledger `license_class`, gates training use per ADR
discipline — same review OpenDialog's NC flag gets):
  pd-usgov        US federal work (chamber proceedings, agency audio) — clean
  opl-uk          Open Parliament Licence — clean with attribution
  cc-by           Creative Commons BY (feed-declared or verified) — clean
  nc-attribution  e.g. C-SPAN policy: non-commercial use w/ attribution —
                  RESEARCH-ONLY, must not enter a commercial-release model
  unverified      license claim not machine-checked yet — cannot train

Usage:
  python -m data.real.collect --list-sources
  python -m data.real.collect --discover "booknotes"          # itunes API
  python -m data.real.collect --add id=... url=... license=... basis="..."
  python -m data.real.collect --collect --source cspan-wt --max 2
  python -m data.real.collect --purge-source cspan-wt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

UA = {"User-Agent": "Mozilla/5.0 (duplex-live research collector)"}
MAX_BYTES = 200 * 1024 * 1024  # per clip hard cap
NS = {"itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd",
      "cc": "http://backend.userland.com/creativeCommonsRssModule"}

DEFAULT_SOURCES = [
    {"id": "cspan-wt", "kind": "rss",
     "url": "https://feeds.megaphone.fm/cspanwashingtontoday",
     "license_class": "nc-attribution",
     "basis": "C-SPAN policy: non-commercial use permitted with attribution "
              "(research-only; not commercial-release clean)"},
    {"id": "cspan-booknotes", "kind": "rss",
     "url": "https://feeds.megaphone.fm/ADV9127984426",
     "license_class": "nc-attribution",
     "basis": "author interviews, two-speaker dynamics; same C-SPAN policy"},
]


def _get(url: str, dest: Path | None = None) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise ValueError(f"{url} exceeds {MAX_BYTES} cap")
        if dest:
            dest.write_bytes(data)
        return data


def _load_sources(root: Path) -> list[dict]:
    p = root / "sources.json"
    if not p.exists():
        p.write_text(json.dumps(DEFAULT_SOURCES, indent=1))
    return json.loads(p.read_text())


def _ledger_path(root: Path) -> Path:
    return root / "collected.jsonl"


def ledger_ids(root: Path) -> set[str]:
    p = _ledger_path(root)
    if not p.exists():
        return set()
    return {json.loads(l)["url"] for l in p.read_text().splitlines() if l.strip()}


# ---------------------------------------------------------------- rss mode

def fetch_feed(url: str) -> tuple[dict, list[dict]]:
    data = _get(url)
    ch = ET.fromstring(data).find("channel")
    meta = {"title": ch.findtext("title") or "",
            "cc_license": ch.findtext("cc:license", namespaces=NS) or ""}
    items = []
    for it in ch.findall("item"):
        enc = it.find("enclosure")
        if enc is None or not (enc.get("url") or "").startswith("http"):
            continue
        items.append({"title": it.findtext("title") or "",
                      "url": enc.get("url"),
                      "bytes": int(enc.get("length") or 0),
                      "duration_hint": it.findtext("itunes:duration",
                                                    namespaces=NS) or ""})
    return meta, items


def collect_rss(src: dict, out_dir: Path, ledger, max_n: int,
                seen: set[str]) -> None:
    meta, items = fetch_feed(src["url"])
    print(f"[{src['id']}] {meta['title']!r}: {len(items)} enclosures"
          + (f" (feed cc: {meta['cc_license']})" if meta["cc_license"] else ""))
    n = 0
    for it in items:
        if n >= max_n:
            break
        if it["url"] in seen:
            continue
        dest = out_dir / f"{src['id']}_{hashlib.sha1(it['url'].encode()).hexdigest()[:10]}.mp3"
        try:
            _get(it["url"], dest)
        except Exception as e:  # noqa: BLE001 — log and continue
            print(f"  skip {it['url'][:60]}: {e}")
            continue
        n += 1
        row = {"clip": dest.name, "source": src["id"], "url": it["url"],
               "title": it["title"][:120],
               "license_class": src["license_class"], "basis": src["basis"],
               "feed_license": meta["cc_license"], "bytes": dest.stat().st_size,
               "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()[:16]}
        ledger.write(json.dumps(row, ensure_ascii=False) + "\n")
        ledger.flush()
        print(f"  got {dest.name} ({dest.stat().st_size // 1048576} MB)"
              f" [{src['license_class']}] {it['title'][:60]!r}")
    print(f"[{src['id']}] collected {n}")


# --------------------------------------------------------------- ytdlp mode

def collect_ytdlp(src: dict, out_dir: Path, ledger, max_n: int,
                  seen: set[str]) -> None:
    for url in src["urls"][:max_n]:
        if url in seen:
            continue
        meta = json.loads(subprocess.run(
            ["yt-dlp", "--no-warnings", "-J", "--sleep-interval", "2", url],
            capture_output=True, text=True, check=True).stdout or "{}")
        lic = (meta.get("license") or "").lower()
        if "creative commons" not in lic:
            print(f"  skip {url}: license={lic or 'unknown'} (not CC)")
            continue
        dest = out_dir / f"{src['id']}_{hashlib.sha1(url.encode()).hexdigest()[:10]}.m4a"
        subprocess.run(["yt-dlp", "-f", "bestaudio", "--sleep-interval", "2",
                        "-o", str(dest), url], check=True)
        row = {"clip": dest.name, "source": src["id"], "url": url,
               "title": (meta.get("title") or "")[:120],
               "license_class": "cc-by", "basis": lic, "bytes": 0,
               "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()[:16]}
        ledger.write(json.dumps(row, ensure_ascii=False) + "\n"); ledger.flush()
        print(f"  got {dest.name} [cc-by] {row['title']!r}")


# ---------------------------------------------------------------- discovery

def discover(term: str) -> None:
    api = ("https://itunes.apple.com/search?media=podcast&limit=8&term="
           + urllib.parse.quote(term))
    for r in json.loads(_get(api)).get("results", []):
        print(f"{(r.get('collectionName') or '')[:44]:44s} "
              f"{r.get('feedUrl') or '(no feed)'}")


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/real",
                    help="dir for sources.json, collected.jsonl, audio/")
    ap.add_argument("--discover")
    ap.add_argument("--add", nargs="+", default=[],
                    help="id=.. kind=rss url=.. license=.. basis=.. [urls=..]")
    ap.add_argument("--list-sources", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--source", help="source id (default: all rss)")
    ap.add_argument("--max", type=int, default=2)
    ap.add_argument("--purge-source")
    args = ap.parse_args()

    import urllib.parse  # noqa: F401  (discover uses it)
    root = Path(args.root)
    (root / "audio").mkdir(parents=True, exist_ok=True)
    sources = _load_sources(root)

    if args.discover:
        discover(args.discover)
        return
    if args.add:
        src = {kv.split("=", 1)[0]: kv.split("=", 1)[1] for kv in args.add}
        if "license" in src:  # accept the shorter flag name
            src["license_class"] = src.pop("license")
        if "urls" in src:
            src["urls"] = src.pop("urls").split(",")
        sources = [s for s in sources if s["id"] != src["id"]] + [src]
        (root / "sources.json").write_text(json.dumps(sources, indent=1))
        print(f"added {src['id']} ({len(sources)} sources)")
        return
    if args.list_sources:
        for s in sources:
            print(f"{s['id']:20s} {s['kind']:6s} [{s.get('license_class','?'):14s}] "
                  f"{s.get('url') or s.get('urls')}")
        return
    if args.purge_source:
        keep, removed = [], 0
        lp = _ledger_path(root)
        for l in lp.read_text().splitlines():
            r = json.loads(l)
            if r["source"] == args.purge_source:
                (root / "audio" / r["clip"]).unlink(missing_ok=True)
                removed += 1
            else:
                keep.append(l)
        lp.write_text("".join(x + "\n" for x in keep))
        print(f"purged {removed} clips from {args.purge_source}")
        return
    if args.collect:
        seen = ledger_ids(root)
        with open(_ledger_path(root), "a") as ledger:
            for src in sources:
                if args.source and src["id"] != args.source:
                    continue
                if src["kind"] == "rss":
                    collect_rss(src, root / "audio", ledger, args.max, seen)
                elif src["kind"] == "ytdlp":
                    collect_ytdlp(src, root / "audio", ledger, args.max, seen)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
