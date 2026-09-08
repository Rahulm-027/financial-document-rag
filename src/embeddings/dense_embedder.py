"""
dense_embedder.py
=================
Embeds chunks using BAAI/bge-base-en-v1.5 and stores them in a FAISS index.

BGE models are trained with a query instruction prefix — we apply it
at query time but NOT for document embeddings (standard BGE practice).

Query prefix: "Represent this sentence for searching relevant passages: "

Storage layout:
  data/indexes/faiss.index        — FAISS FlatIP index
  data/indexes/faiss_metadata.jsonl — one JSON line per chunk (same order as index)
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)

BGE_QUERY_INSTRUCTION = (
    "Represent this sentence for searching relevant passages: "
)


class DenseEmbedder:
    """
    Manages the dense embedding layer:
      - encode(texts) → normalised float32 ndarray
      - build_index(chunks) → writes FAISS + metadata to disk
      - load_index() → loads from disk
      - search(query, k) → list of (chunk_dict, score) pairs
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: str = "cpu",
        batch_size: int = 32,
        index_path: str = "data/indexes/faiss.index",
        metadata_path: str = "data/indexes/faiss_metadata.jsonl",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)

        self._model: Optional[SentenceTransformer] = None
        self._index: Optional[faiss.Index] = None
        self._metadata: list[dict] = []

    # ------------------------------------------------------------------
    # Model loading (lazy)
    # ------------------------------------------------------------------

    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(
        self,
        texts: list[str],
        is_query: bool = False,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode a list of texts to L2-normalised float32 embeddings.

        Args
        ----
        texts       : list of strings
        is_query    : if True, prepend the BGE query instruction
        show_progress : tqdm progress bar
        """
        model = self._load_model()

        if is_query:
            texts = [BGE_QUERY_INSTRUCTION + t for t in texts]

        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,   # cosine via inner product
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def build_index(self, chunks: list[dict]) -> None:
        """
        Build a FAISS FlatIP index from chunk dicts.
        Saves index + metadata to disk.

        Args
        ----
        chunks : list of Chunk.to_dict() objects — must have 'content' key
        """
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)

        texts = [c["content"] for c in chunks]

        logger.info(f"Encoding {len(texts)} chunks…")
        embeddings = self.encode(texts, is_query=False, show_progress=True)

        dim = embeddings.shape[1]
        logger.info(f"Building FAISS FlatIP index (dim={dim})…")
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        faiss.write_index(index, str(self.index_path))
        logger.info(f"FAISS index saved → {self.index_path}")

        with open(self.metadata_path, "w") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk) + "\n")
        logger.info(f"Metadata saved → {self.metadata_path} ({len(chunks)} records)")

        self._index = index
        self._metadata = chunks

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_index(self) -> None:
        """Load FAISS index and metadata from disk."""
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {self.index_path}. "
                "Run build_index first."
            )
        self._index = faiss.read_index(str(self.index_path))
        self._metadata = []
        with open(self.metadata_path) as f:
            for line in f:
                self._metadata.append(json.loads(line.strip()))
        logger.info(
            f"Loaded FAISS index with {self._index.ntotal} vectors "
            f"and {len(self._metadata)} metadata records."
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 20,
    ) -> list[tuple[dict, float]]:
        """
        Search the dense index for the top-k most relevant chunks.

        Returns
        -------
        list of (chunk_dict, score) pairs, sorted by score descending
        """
        if self._index is None:
            self.load_index()

        q_emb = self.encode([query], is_query=True)  # shape (1, dim)
        scores, indices = self._index.search(q_emb, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:        # FAISS returns -1 for empty slots
                continue
            chunk = self._metadata[idx].copy()
            results.append((chunk, float(score)))

        return results  # already sorted by score (inner product = cosine)
