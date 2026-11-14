"""Lightweight hybrid retriever (BM25 + dense vectors, fused with RRF) that runs on Vercel.

The full `retrieval.Retriever` needs chromadb + sentence-transformers (too big for a serverless
function). This one needs only numpy and google-genai:

    keyword side : pure-Python BM25 over chunking/chunks.json
    semantic side: chunk vectors precomputed ONCE by `python embedding/build_vectors.py`
                   (embedding/vectors.npy), the query is embedded with the same Gemini model
    fusion       : reciprocal-rank fusion, same as retrieval.Retriever
    routing      : the same rule-based query -> metadata-filter routing as retrieval.Retriever

If the vectors file is missing, GEMINI_API_KEY is not set, or the embedding call fails, it degrades
to BM25 (+ routing) and reports `last_mode == "keyword"`; otherwise `last_mode == "hybrid"`.
"""

import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    from .retrieval import (CANDIDATE_MULT, RRF_K, build_aliases, build_where, is_conversational_or_irrelevant,
                            matches, route_query, summarize_matches, tokenize)
except ImportError:
    from retrieval import (CANDIDATE_MULT, RRF_K, build_aliases, build_where, is_conversational_or_irrelevant,
                           matches, route_query, summarize_matches, tokenize)

ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = ROOT / "chunking" / "chunks.json"
VECTORS_PATH = ROOT / "embedding" / "vectors.npy"
VECTORS_META_PATH = ROOT / "embedding" / "vectors_meta.json"

EMBED_MODEL = os.environ.get("GEMINI_EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = int(os.environ.get("GEMINI_EMBED_DIM", "768"))
# Optional "no answer" gate on the best dense cosine score (0 = off). Gemini scores are not
# calibrated here; set LITE_MIN_DENSE after looking at a few unanswerable questions.
MIN_DENSE = float(os.environ.get("LITE_MIN_DENSE", "0") or 0)


def gemini_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


class GeminiEmbedder:
    """Embeds documents / queries with the Gemini embedding API (shared by build_vectors.py and runtime)."""

    def __init__(self, model: str = EMBED_MODEL, dim: int = EMBED_DIM, api_key: str | None = None):
        from google import genai
        key = api_key or gemini_key()
        if not key:
            raise RuntimeError("Set GEMINI_API_KEY to use semantic search")
        self.client = genai.Client(api_key=key)
        self.model, self.dim = model, dim

    def _embed(self, texts: list[str], task_type: str) -> np.ndarray:
        from google.genai import types
        res = self.client.models.embed_content(
            model=self.model, contents=texts,
            config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.dim))
        vecs = np.array([e.values for e in res.embeddings], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.clip(norms, 1e-12, None)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


class LiteRetriever:
    def __init__(self, chunks_path: Path = CHUNKS_PATH, vectors_path: Path = VECTORS_PATH,
                 meta_path: Path = VECTORS_META_PATH, embed_query: Callable[[str], np.ndarray] | None = None):
        with Path(chunks_path).open(encoding="utf-8") as f:
            chunks = json.load(f)
        self.ids = [c["chunk_id"] for c in chunks]
        self.docs = [c["text"] for c in chunks]
        self.metas = [c.get("metadata", {}) for c in chunks]
        self.fuels = sorted({m["fuel"] for m in self.metas if m.get("fuel") and re.fullmatch(r"[a-z][a-z \-]+", m["fuel"])},
                            key=len, reverse=True)
        self.has_numeric_meta = any("o2_pct" in m for m in self.metas)
        self.aliases = build_aliases(self.metas)
        self.min_dense = MIN_DENSE
        self.last_route: dict[str, Any] = {}
        self.last_mode = "keyword"

        # ---- BM25 (inverted index) ----
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._len = np.zeros(len(self.docs), dtype=np.float32)
        for i, d in enumerate(self.docs):
            toks = tokenize(d)
            self._len[i] = len(toks) or 1
            for t, tf in Counter(toks).items():
                self._postings.setdefault(t, []).append((i, tf))
        n = len(self.docs)
        self._avgdl = float(self._len.mean()) if n else 1.0
        self._idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self._postings.items()}

        # ---- dense vectors (optional) ----
        self.vectors: np.ndarray | None = None
        self.dense_error: str | None = None
        self._embed_query = embed_query
        self._load_vectors(Path(vectors_path), Path(meta_path))

    # ------------------------------------------------------------------ setup
    def _load_vectors(self, vectors_path: Path, meta_path: Path) -> None:
        if not (vectors_path.exists() and meta_path.exists()):
            self.dense_error = "no embedding/vectors.npy (run: python embedding/build_vectors.py)"
            return
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            raw = np.load(vectors_path).astype(np.float32)
            row = {cid: i for i, cid in enumerate(meta["ids"])}
            mat = np.zeros((len(self.ids), raw.shape[1]), dtype=np.float32)  # chunks without a vector score 0
            for i, cid in enumerate(self.ids):
                j = row.get(cid)
                if j is not None:
                    mat[i] = raw[j]
            self.vectors = mat
            self.embed_model, self.embed_dim = meta["model"], int(meta["dim"])
            if self._embed_query is None:
                if not gemini_key():
                    self.dense_error = "GEMINI_API_KEY not set (needed to embed the query)"
                    return
                emb = GeminiEmbedder(self.embed_model, self.embed_dim)
                self._embed_query = emb.embed_query
        except Exception as exc:  # corrupt file, missing google-genai, ...
            self.vectors, self.dense_error = None, f"{type(exc).__name__}: {exc}"

    @property
    def dense_ready(self) -> bool:
        return self.vectors is not None and self._embed_query is not None

    # ------------------------------------------------------------------ rankers
    def _keyword(self, query: str, n: int, allowed: set[int] | None) -> list[dict[str, Any]]:
        k1, b = 1.5, 0.75
        scores: dict[int, float] = {}
        for t in set(tokenize(query)):
            idf = self._idf.get(t)
            if idf is None:
                continue
            for i, tf in self._postings[t]:
                if allowed is not None and i not in allowed:
                    continue
                scores[i] = scores.get(i, 0.0) + idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * self._len[i] / self._avgdl))
        top = sorted(scores, key=scores.__getitem__, reverse=True)[:n]
        return [{"chunk_id": self.ids[i], "text": self.docs[i], "metadata": self.metas[i],
                 "bm25_score": float(scores[i])} for i in top]

    def _dense(self, query: str, n: int, allowed: set[int] | None) -> list[dict[str, Any]]:
        q = np.asarray(self._embed_query(query), dtype=np.float32)
        sims = self.vectors @ q
        idx = np.fromiter(allowed, dtype=np.int64) if allowed is not None else np.arange(len(sims))
        if idx.size == 0:
            return []
        order = idx[np.argsort(-sims[idx])[:n]]
        return [{"chunk_id": self.ids[i], "text": self.docs[i], "metadata": self.metas[i],
                 "dense_score": float(sims[i])} for i in order if sims[i] > 0]

    # ------------------------------------------------------------------ public API (same as Retriever.search)
    def search(self, query: str, top_k: int = 5, doc_type: str | None = None, psi_id: str | None = None,
               route: bool = True, min_score: float | None = None) -> list[dict[str, Any]]:
        self.last_mode = "keyword"
        if is_conversational_or_irrelevant(query):
            self.last_route = {"intent": "conversational", "no_answer": "conversational or greeting query"}
            return []
        clauses: list[dict] = []
        self.last_route = {}
        if route:
            r = route_query(query, self.fuels, self.aliases)
            clauses = [c for c in r["clauses"]
                       if not (doc_type and "document_type" in c) and not (psi_id and "psi_id" in c)]
            if not self.has_numeric_meta:
                clauses = [c for c in clauses if not set(c) & {"o2_pct", "fuel", "outcome"}]
            self.last_route = dict(r["notes"])
        if doc_type:
            clauses.append({"document_type": doc_type})
        if psi_id:
            clauses.append({"psi_id": psi_id})
        where = build_where(clauses)
        allowed = {i for i, m in enumerate(self.metas) if matches(m, where)} if where else None

        if any(set(c) & {"o2_pct", "fuel", "outcome"} for c in clauses):
            matched = [self.metas[i] for i in (allowed or ())]
            self.last_route["total_matches"] = len(matched)
            self.last_route["match_summary"] = summarize_matches(matched)

        n = top_k * CANDIDATE_MULT
        rankings = [self._keyword(query, n, allowed)]
        if self.dense_ready:
            try:
                dense = self._dense(query, n, allowed)
                gate = min_score if min_score else self.min_dense
                if gate and dense and dense[0]["dense_score"] < gate:
                    self.last_route["no_answer"] = f"best dense score {dense[0]['dense_score']:.3f} < {gate}"
                    return []
                rankings.append(dense)
                self.last_mode = "hybrid"
            except Exception as exc:  # rate limit / network: keep answering with keywords
                self.last_route["dense_error"] = f"{type(exc).__name__}: {exc}"[:200]

        merged: dict[str, dict[str, Any]] = {}
        for ranked in rankings:
            for rank, hit in enumerate(ranked, start=1):
                entry = merged.setdefault(hit["chunk_id"], {**hit, "rrf_score": 0.0})
                entry.update({k: v for k, v in hit.items() if k.endswith("_score")})
                entry["rrf_score"] += 1.0 / (RRF_K + rank)
        return sorted(merged.values(), key=lambda h: h["rrf_score"], reverse=True)[:top_k]
