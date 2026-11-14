"""Precompute chunk embeddings for the serverless hybrid retriever (embedding/lite_retrieval.py).

Run ONCE locally (needs GEMINI_API_KEY) and commit the two output files so Vercel can use them:

    pip install -r requirements.txt
    python embedding/build_vectors.py

Writes embedding/vectors.npy (float16, one row per chunk) and embedding/vectors_meta.json
(model, dim, chunk ids). Safe to interrupt and re-run: progress is kept in embedding/vectors_partial.npz.
Re-run it whenever chunking/chunks.json changes. The query side must use the same model/dim
(recorded in vectors_meta.json, so nothing to configure).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lite_retrieval import CHUNKS_PATH, EMBED_DIM, EMBED_MODEL, VECTORS_META_PATH, VECTORS_PATH, GeminiEmbedder  # noqa: E402

PARTIAL = VECTORS_PATH.with_name("vectors_partial.npz")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=EMBED_MODEL)
    ap.add_argument("--dim", type=int, default=EMBED_DIM)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--max-chars", type=int, default=6000, help="truncate very long chunks")
    args = ap.parse_args()

    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    ids = [c["chunk_id"] for c in chunks]
    done: dict[str, np.ndarray] = {}
    if PARTIAL.exists():
        p = np.load(PARTIAL, allow_pickle=False)
        if str(p["model"]) == args.model and int(p["dim"]) == args.dim:
            done = dict(zip(p["ids"].tolist(), p["vecs"]))
            print(f"Resuming: {len(done)}/{len(ids)} already embedded")

    emb = GeminiEmbedder(args.model, args.dim)
    todo = [c for c in chunks if c["chunk_id"] not in done]
    for start in range(0, len(todo), args.batch_size):
        batch = todo[start:start + args.batch_size]
        texts = [c["text"][:args.max_chars] for c in batch]
        for attempt in range(8):
            try:
                vecs = emb.embed_documents(texts)
                break
            except Exception as exc:  # 429 / transient: back off
                wait = min(60, 5 * 2 ** attempt)
                print(f"  retry in {wait}s ({type(exc).__name__}: {str(exc)[:120]})")
                time.sleep(wait)
        else:
            sys.exit("Giving up after 8 retries; re-run to resume.")
        for c, v in zip(batch, vecs):
            done[c["chunk_id"]] = v.astype(np.float16)
        np.savez(PARTIAL, ids=np.array(list(done)), vecs=np.stack(list(done.values())),
                 model=args.model, dim=args.dim)
        print(f"Embedded {len(done)}/{len(ids)}")

    np.save(VECTORS_PATH, np.stack([done[i] for i in ids]).astype(np.float16))
    VECTORS_META_PATH.write_text(json.dumps({"model": args.model, "dim": args.dim, "ids": ids}), encoding="utf-8")
    PARTIAL.unlink(missing_ok=True)
    print(f"Done: {VECTORS_PATH} ({VECTORS_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
