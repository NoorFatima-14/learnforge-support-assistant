import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from schema import Chunk, RetrievedChunk

INDEX_DIR = Path(__file__).parent.parent / "index"
MODEL_NAME = "all-MiniLM-L6-v2"

_model = None  # loaded lazily and cached at module level


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


class Retriever:
    def __init__(self, chunks=None):
        self.chunks = chunks or []
        self.embeddings = None  # np.ndarray, shape (n_chunks, dim), L2-normalized

    def build(self, chunks):
        self.chunks = chunks
        model = _get_model()
        texts = [c.text for c in self.chunks]
        self.embeddings = model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )
        return self

    def save(self, index_dir: Path = INDEX_DIR):
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / "embeddings.npy", self.embeddings)
        with open(index_dir / "chunks.json", "w") as f:
            json.dump([c.to_dict() for c in self.chunks], f, indent=2)

    @classmethod
    def load(cls, index_dir: Path = INDEX_DIR):
        embeddings = np.load(index_dir / "embeddings.npy")
        with open(index_dir / "chunks.json") as f:
            chunks = [Chunk.from_dict(d) for d in json.load(f)]
        r = cls(chunks)
        r.embeddings = embeddings
        return r

    def search(self, query: str, top_k: int = 4):
        if self.embeddings is None:
            raise RuntimeError("Retriever index not built/loaded yet.")
        model = _get_model()
        q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        # both sides are L2-normalized, so a dot product IS cosine similarity
        sims = self.embeddings @ q_emb
        ranked = np.argsort(-sims)[:top_k]
        return [RetrievedChunk(chunk=self.chunks[i], score=float(sims[i])) for i in ranked]


def build_and_save_index():
    from ingest import load_all_chunks
    chunks = load_all_chunks()
    r = Retriever().build(chunks)
    r.save()
    print(f"Built index with {len(chunks)} chunks -> {INDEX_DIR}")
    return r


if __name__ == "__main__":
    r = build_and_save_index()
    for q in ["can I get my money back", "video won't play on my phone", "someone charged my card twice"]:
        print(f"\nQuery: {q}")
        for rc in r.search(q, top_k=3):
            print(f"  {rc.score:.3f}  {rc.chunk.id}  {rc.chunk.title}")
