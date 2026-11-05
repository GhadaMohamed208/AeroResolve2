"""Second data source: retrieval over the baggage policy PDF (tiny BM25, no heavy deps)."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_STOP = set("""a an and are as at be by can for from has have how i in is it its of on or that the
their this to was were what when which who will with you your my me if do does not no than then""".split())
_HEADING = re.compile(r"^(\d+)\.\s+([A-Z].*)$")


@dataclass
class Chunk:
    number: str
    title: str
    text: str


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w[:-1] if len(w) > 4 and w.endswith("s") else w for w in words if w not in _STOP]


def _extract_sections(pdf_path: Path) -> list[Chunk]:
    from pypdf import PdfReader
    lines: list[str] = []
    for page in PdfReader(str(pdf_path)).pages:
        lines.extend((page.extract_text() or "").splitlines())
    chunks: list[Chunk] = []
    number = title = ""
    body: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if m := _HEADING.match(line):
            if title:
                chunks.append(Chunk(number, title, " ".join(body)))
            number, title, body = m.group(1), m.group(2), []
        elif title:
            body.append(line)
    if title:
        chunks.append(Chunk(number, title, " ".join(body)))
    return chunks


class PolicyIndex:
    """BM25 index over policy sections."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._docs = [Counter(_tokens(c.title + " " + c.title + " " + c.text)) for c in chunks]
        self._len = [sum(d.values()) for d in self._docs]
        self._avg = (sum(self._len) / len(self._len)) if self._len else 1.0
        df: Counter = Counter()
        for d in self._docs:
            df.update(d.keys())
        n = len(self._docs)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 3) -> list[dict]:
        q = _tokens(query)
        scored = []
        for i, doc in enumerate(self._docs):
            score = 0.0
            for t in q:
                tf = doc.get(t, 0)
                if tf:
                    score += self._idf.get(t, 0) * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * self._len[i] / self._avg))
            if score > 0:
                scored.append((score, i))
        scored.sort(reverse=True)
        out = []
        for score, i in scored[:k]:
            c = self.chunks[i]
            out.append({"section": c.number, "title": c.title, "text": c.text, "score": round(score, 3)})
        return out


@lru_cache(maxsize=4)
def _load(path: str, mtime: float) -> PolicyIndex:
    return PolicyIndex(_extract_sections(Path(path)))


def load_policy_index(pdf_path: str | Path) -> PolicyIndex:
    p = Path(pdf_path)
    return _load(str(p), p.stat().st_mtime)
