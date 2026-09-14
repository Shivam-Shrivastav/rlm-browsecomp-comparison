"""Document loading, chunking, and two retrievers: BM25 and dense (OpenAI-
compatible embeddings, with a zero-dependency TF-IDF fallback when no key).

`load_text` accepts:
  - a single .pdf file           -> extracted page text
  - a single .txt/.md file       -> raw text
  - a directory                  -> every .txt/.md/.pdf under it, recursively,
                                    concatenated. This is how you load the
                                    BrowseComp-Plus 100K corpus (plain .txt
                                    files) to reach the >10M-token regime.

All retrievers expose the same interface:
    retrieve(query: str, k: int) -> list[str]   # the k retrieved chunk texts
"""
from __future__ import annotations

import os
from dataclasses import dataclass


# --------------------------------------------------------------------------- #
# file / dir -> text -> chunks
# --------------------------------------------------------------------------- #
def load_pdf_text(path: str) -> str:
    """Extract concatenated text from a PDF. Tries pdfplumber, then pypdf."""
    try:
        import pdfplumber

        pages = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                pages.append(page.extract_text() or "")
        text = "\n\n".join(p for p in pages if p.strip())
        if text.strip():
            return text
    except Exception as e:  # pragma: no cover
        print(f"[retrievers] pdfplumber failed ({e}); trying pypdf…")

    import pypdf  # installed as a pdfplumber dependency

    reader = pypdf.PdfReader(path)
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def load_text(path: str) -> str:
    """Load text from a .pdf, a .txt/.md file, or a directory of such files."""
    if os.path.isdir(path):
        parts, n = [], 0
        for root, _dirs, files in os.walk(path):
            for fn in sorted(files):
                if fn.lower().endswith((".txt", ".md", ".pdf")):
                    fp = os.path.join(root, fn)
                    if fn.lower().endswith(".pdf"):
                        parts.append(load_pdf_text(fp))
                    else:
                        with open(fp, encoding="utf-8", errors="ignore") as f:
                            parts.append(f.read())
                    n += 1
        print(f"[retrievers] loaded {n} files from directory {path}")
        return "\n\n".join(p for p in parts if p and p.strip())
    if path.lower().endswith(".pdf"):
        return load_pdf_text(path)
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()


def chunk_text(text: str, size: int = 1500, overlap: int = 300) -> list[str]:
    """Sliding-window char chunks with overlap. 1500 chars ~= 350-400 tokens."""
    if not text:
        return []
    step = max(1, size - overlap)
    return [text[i : i + size] for i in range(0, len(text), step)]


# --------------------------------------------------------------------------- #
# BM25
# --------------------------------------------------------------------------- #
@dataclass
class BM25Retriever:
    chunks: list[str]

    @classmethod
    def fit(cls, chunks: list[str]) -> "BM25Retriever":
        # tokenized lazily; rank_bm25 is a pure-Python dependency
        from rank_bm25 import BM25Okapi

        self = cls(chunks=chunks)
        self._tok = [c.lower().split() for c in chunks]
        self._bm = BM25Okapi(self._tok)
        return self

    def retrieve(self, query: str, k: int = 8) -> list[str]:
        scores = self._bm.get_scores(query.lower().split())
        idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.chunks[i] for i in idx if scores[i] > 0]


# --------------------------------------------------------------------------- #
# Dense: OpenAI embeddings, with TF-IDF fallback
# --------------------------------------------------------------------------- #
@dataclass
class DenseRetriever:
    chunks: list[str]
    kind: str  # "openai" or "tfidf"

    @classmethod
    def fit(cls, chunks: list[str], openai_key: str | None, embed_model: str,
            base_url: str | None = None) -> "DenseRetriever":
        if openai_key:
            try:
                return cls._fit_openai(chunks, openai_key, embed_model, base_url)
            except Exception as e:
                print(f"[retrievers] embeddings failed ({e}); using TF-IDF fallback.")
        return cls._fit_tfidf(chunks)

    @classmethod
    def _fit_openai(cls, chunks, key, model, base_url) -> "DenseRetriever":
        import numpy as np
        from openai import OpenAI

        client = OpenAI(api_key=key, base_url=base_url or None)
        # batch embeddings (OpenAI allows up to 2048 inputs; keep batches small)
        vecs = []
        for i in range(0, len(chunks), 256):
            batch = chunks[i : i + 256]
            r = client.embeddings.create(input=batch, model=model)
            vecs.extend([d.embedding for d in r.data])
        self = cls(chunks=chunks, kind="openai")
        self._mat = np.array(vecs)
        self._client = client
        self._emodel = model
        return self

    @classmethod
    def _fit_tfidf(cls, chunks) -> "DenseRetriever":
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vec = TfidfVectorizer(stop_words="english")
        self = cls(chunks=chunks, kind="tfidf")
        self._vec = vec
        self._mat = vec.fit_transform(chunks)  # sparse
        self._cos = cosine_similarity
        return self

    def retrieve(self, query: str, k: int = 8) -> list[str]:
        if self.kind == "openai":
            import numpy as np

            qv = self._client.embeddings.create(input=[query], model=self._emodel).data[0].embedding
            sims = self._mat @ np.array(qv)
        else:
            qv = self._vec.transform([query])
            sims = (self._mat @ qv.T).toarray().ravel()
        idx = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]
        return [self.chunks[i] for i in idx if sims[i] > 0]