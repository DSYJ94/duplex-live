"""Brain backends for the delegation boundary (Issue #5).

Interface: a backend owns ONE pre-warmed background session and streams
answer chunks. The boundary (boundary.py) enforces the async/never-block
contract, latency budget, and provenance — backends just talk to brains.

ADR-0002: closed-API brains (Kimi, hosted agents) are LEGAL here — the
delegation boundary is exactly the sanctioned place for them — but their
outputs never enter training data. The boundary's ledger + directory
separation enforce that; swap to a self-hosted open model by writing one
more backend.
"""

from __future__ import annotations

import abc
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path


class BrainBackend(abc.ABC):
    """One resident background session per conversation."""

    name = "abstract"

    @abc.abstractmethod
    def prewarm(self, context: str) -> float:
        """Start/resume the session with shared context. Returns warm-up
        seconds (prompt-cache friendly: send context exactly once)."""

    @abc.abstractmethod
    def ask_stream(self, question: str):
        """Yield answer chunks (str) as they arrive. First yield latency
        is the first-useful-result metric."""


class MockBackend(BrainBackend):
    """Deterministic timing for tests/CI — no network, no secrets."""

    name = "mock"

    def __init__(self, first_chunk_s: float = 0.8, chunk_s: float = 0.05,
                 n_chunks: int = 8):
        self.first_chunk_s, self.chunk_s, self.n_chunks = first_chunk_s, chunk_s, n_chunks
        self.prewarmed = None

    def prewarm(self, context: str) -> float:
        self.prewarmed = context
        return 0.0

    def ask_stream(self, question: str):
        time.sleep(self.first_chunk_s)
        for i in range(self.n_chunks):
            yield f"[mock {i}] part of the answer to: {question[:40]}… "
            time.sleep(self.chunk_s)


class KimiBackend(BrainBackend):
    """Moonshot Kimi via the OpenAI-compatible API, SSE streaming.

    Key via KIMI_API_KEY (or MOONSHOT_API_KEY). Base URL overridable for
    self-hosted OpenAI-compatible servers — that's the open-model swap
    path (vLLM/sglang serve the same schema).
    """

    name = "kimi"

    def __init__(self, model: str = "kimi-k2-0905-preview",
                 base_url: str = "https://api.moonshot.ai/v1"):
        self.model, self.base_url = model, base_url.rstrip("/")
        self.key = os.environ.get("KIMI_API_KEY") or os.environ.get("MOONSHOT_API_KEY")
        self.messages: list[dict] = []

    def prewarm(self, context: str) -> float:
        t0 = time.time()
        if not self.messages:
            self.messages = [{"role": "system", "content": context}]
        return time.time() - t0

    def ask_stream(self, question: str):
        self.messages.append({"role": "user", "content": question})
        body = json.dumps({"model": self.model, "stream": True,
                           "messages": self.messages}).encode()
        req = urllib.request.Request(
            self.base_url + "/chat/completions", data=body, headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.key}",
                "Accept": "text/event-stream"})
        full = []
        with urllib.request.urlopen(req, timeout=60) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    delta = json.loads(payload)["choices"][0]["delta"]
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                text = delta.get("content") or ""
                if text:
                    full.append(text)
                    yield text
        self.messages.append({"role": "assistant", "content": "".join(full)})


# ---------------------------------------------------------------- ACP

class AcpBackend(BrainBackend):
    """Any ACP-compatible coding agent (claude/qwen/codex/gemini --acp)
    as a child process over JSON-RPC 2.0 stdio (agentclientprotocol.com).

    The generic 'background brain' of docs/01: delegates real work to a
    coding agent and streams its agent_message_chunk updates back.
    """

    name = "acp"

    def __init__(self, command: list[str], cwd: str = "."):
        self.command, self.cwd = command, cwd
        self.session_id: str | None = None
        self.proc: subprocess.Popen | None = None
        self._id = 0

    def _send(self, method: str, params: dict) -> int:
        self._id += 1
        self.proc.stdin.write(json.dumps(
            {"jsonrpc": "2.0", "id": self._id, "method": method,
             "params": params}) + "\n")
        self.proc.stdin.flush()
        return self._id

    def _read(self, want_id: int, timeout_s: float = 120.0) -> dict:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("ACP agent exited")
            msg = json.loads(line)
            if msg.get("id") == want_id:
                if "error" in msg:
                    raise RuntimeError(f"ACP error: {msg['error']}")
                return msg.get("result", {})

    def prewarm(self, context: str) -> float:
        t0 = time.time()
        self.proc = subprocess.Popen(
            self.command, cwd=self.cwd, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            bufsize=1)
        rid = self._send("initialize", {"protocolVersion": 1,
                                        "clientCapabilities": {}})
        self._read(rid)
        rid = self._send("session/new",
                         {"cwd": str(Path(self.cwd).resolve()),
                          "mcpServers": []})
        res = self._read(rid)
        self.session_id = res.get("sessionId")
        if context:
            rid = self._send("session/prompt", {
                "sessionId": self.session_id,
                "prompt": [{"type": "text",
                            "text": f"[session context, no reply needed] {context}"}]})
            self._drain_prompt(rid)
        return time.time() - t0

    def _drain_prompt(self, rid: int):
        """Consume notifications until the prompt's result arrives;
        yields nothing (used by prewarm)."""
        for _ in self._prompt_iter(rid):
            pass

    def _prompt_iter(self, rid: int):
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("ACP agent exited mid-prompt")
            msg = json.loads(line)
            if msg.get("id") == rid and "result" in msg:
                yield None
                return
            if msg.get("method") == "session/update":
                upd = msg.get("params", {}).get("update", {})
                if upd.get("sessionUpdate") == "agent_message_chunk":
                    block = upd.get("content", {})
                    if block.get("type") == "text" and block.get("text"):
                        yield block["text"]

    def ask_stream(self, question: str):
        rid = self._send("session/prompt", {
            "sessionId": self.session_id,
            "prompt": [{"type": "text", "text": question}]})
        for chunk in self._prompt_iter(rid):
            if chunk:
                yield chunk
