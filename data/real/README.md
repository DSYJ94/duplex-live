# Layer-2 real-audio collector (Issue #3 / docs/02-data.md)

Fetches candidate conversational audio with per-clip provenance. Every clip
gets a `collected.jsonl` ledger row (source, url, license_class, sha256)
BEFORE it is usable; purgeable by source. License classes gate training use:

- `pd-usgov`, `opl-uk`, `cc-by` — clean (attribution where required)
- `nc-attribution` (e.g. C-SPAN policy) — research-only, same review
  OpenDialog's NC flag gets before any commercial-release model
- `unverified` — cannot train

Modes: `rss` (podcast enclosures — the sanctioned path; stdlib, runs
anywhere) and `ytdlp` (YouTube CC-BY subset only: the video's license
field is verified Creative Commons before download; rate-limited, no
cookies, no circumvention). Platform-ToS caveat for YouTube downloads is
recorded as a deliberate user decision (2026-09-18, docs/02-data.md).

```bash
python -m data.real.collect --list-sources
python -m data.real.collect --discover "UK Parliament"    # itunes API
python -m data.real.collect --add id=.. kind=rss url=.. license=opl-uk basis=".."
python -m data.real.collect --collect --source parliament-pmq --max 5
python -m data.real.collect --purge-source cspan-wt
```

Seeded sources: `cspan-wt` + `cspan-booknotes` (nc-attribution),
`parliament-pmq` (opl-uk — interruption-heavy PMQs audio).

Downstream: collected audio is raw; speaker separation + diarization +
80 ms forced alignment (DuplexChat / Nakata-style pipeline) gate its entry
into the training mixture — collection fills the shelf meanwhile. Audition
on the gallery comes after segmentation into scenario-like entries.
