
"""Persistent ChromaDB storage for precomputed scientific text embeddings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORE_PATH = PROJECT_ROOT / "data" / "vector_store" / "chroma"
DEFAULT_COLLECTION_NAME = "scientific_papers"


class ChromaStore:
    """Store and search chunk embeddings without generating embeddings here.

    The default location is alongside the project's saved embedding artifacts.
    ``search`` accepts an embedding produced by the project's embedding model;
    ``search_text`` is a convenience wrapper around a supplied model's ``encode``.
    """

    def __init__(
        self,
        persist_directory: str | Path = DEFAULT_STORE_PATH,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        distance_metric: str = "cosine",
    ) -> None:
        if not collection_name or len(collection_name) > 63:
            raise ValueError("collection_name must contain between 1 and 63 characters")
        if distance_metric not in {"cosine", "l2", "ip"}:
            raise ValueError("distance_metric must be one of: cosine, l2, ip")

        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - exercised when dependency missing
            raise ImportError(
                "ChromaDB is required. Install project dependencies with `pip install -r requirements.txt`."
            ) from exc

        self.persist_directory = Path(persist_directory).expanduser().resolve()
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": distance_metric},
        )
        self.distance_metric = distance_metric

    @staticmethod
    def _metadata_value(value: Any) -> str | int | float | bool | None:
        """Convert metadata into Chroma's supported scalar values."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, np.generic):
            return value.item()
        # Chroma metadata values are scalar; preserve richer metadata losslessly.
        if isinstance(value, (list, tuple, dict)):
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        return str(value)

    @staticmethod
    def _embedding_matrix(embeddings: Any) -> np.ndarray:
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("embeddings must be a non-empty 2D array")
        if not np.isfinite(matrix).all():
            raise ValueError("embeddings must contain only finite values")
        return matrix

    def add_embeddings(
        self,
        embeddings: Any,
        metadata: Sequence[Mapping[str, Any]],
        *,
        ids: Sequence[str] | None = None,
        documents: Sequence[str] | None = None,
        batch_size: int = 500,
    ) -> int:
        """Upsert precomputed vectors and chunk metadata; return row count.

        By default IDs are made from each chunk's paper_id and chunk_id, so
        rerunning an indexing pass updates existing chunks instead of duplicating
        them. ``text`` is stored as the Chroma document; remaining fields are
        stored as metadata.
        """
        vectors = self._embedding_matrix(embeddings)
        records = list(metadata)
        count = len(vectors)
        if len(records) != count:
            raise ValueError("metadata and embeddings must have the same number of rows")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        if ids is None:
            generated_ids = []
            for index, record in enumerate(records):
                paper_id = record.get("paper_id")
                chunk_id = record.get("chunk_id")
                if paper_id is None or chunk_id is None:
                    generated_ids.append(f"chunk-{index}")
                else:
                    generated_ids.append(f"{paper_id}::{chunk_id}")
            ids = generated_ids
        else:
            ids = [str(value) for value in ids]
        if len(ids) != count or len(set(ids)) != count:
            raise ValueError("ids must be unique and have the same length as embeddings")

        if documents is None:
            docs = [str(record.get("text", "")) for record in records]
        else:
            docs = [str(document) for document in documents]
        if len(docs) != count:
            raise ValueError("documents and embeddings must have the same number of rows")
        if any(not doc.strip() for doc in docs):
            raise ValueError("every embedding must have a non-empty document")

        metadatas = []
        for record in records:
            clean = {
                str(key): converted
                for key, value in record.items()
                if key != "text"
                if (converted := self._metadata_value(value)) is not None
            }
            metadatas.append(clean or None)

        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            self.collection.upsert(
                ids=list(ids[start:stop]),
                embeddings=vectors[start:stop].tolist(),
                documents=docs[start:stop],
                metadatas=metadatas[start:stop],
            )
        return count

    def add_embedded_chunks(
        self, embedded_chunks: Iterable[Mapping[str, Any]], *, batch_size: int = 500
    ) -> int:
        """Index the ``{text, embedding, paper_id, chunk_id, ...}`` records
        returned by ``src.retrieval.embedding_generator.embed_all_chunks``.
        """
        chunks = list(embedded_chunks)
        if not chunks:
            return 0
        if any("embedding" not in chunk or "text" not in chunk for chunk in chunks):
            raise ValueError("each embedded chunk must contain 'embedding' and 'text'")
        vectors = [chunk["embedding"] for chunk in chunks]
        metadata = [{key: value for key, value in chunk.items() if key != "embedding"} for chunk in chunks]
        return self.add_embeddings(vectors, metadata, batch_size=batch_size)

    def index_saved_embeddings(
        self,
        embeddings_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
        *,
        batch_size: int = 500,
    ) -> int:
        """Load the NumPy and JSON files written by ``save_embeddings``."""
        artifact_dir = PROJECT_ROOT / "data" / "vector_store"
        embeddings_path = Path(embeddings_path or artifact_dir / "embeddings.npy")
        metadata_path = Path(metadata_path or artifact_dir / "embedding_metadata.json")
        vectors = np.load(embeddings_path, allow_pickle=False)
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        return self.add_embeddings(vectors, metadata, batch_size=batch_size)

    def search(
        self,
        query_embedding: Any,
        *,
        n_results: int = 5,
        where: Mapping[str, Any] | None = None,
        where_document: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return nearest chunks for a query vector, ordered by distance."""
        query = np.asarray(query_embedding, dtype=np.float32)
        if query.ndim == 2 and query.shape[0] == 1:
            query = query[0]
        if query.ndim != 1 or query.size == 0 or not np.isfinite(query).all():
            raise ValueError("query_embedding must be a non-empty finite 1D vector")
        if n_results <= 0:
            raise ValueError("n_results must be greater than zero")
        if self.collection.count() == 0:
            return []

        result = self.collection.query(
            query_embeddings=[query.tolist()],
            n_results=min(n_results, self.collection.count()),
            where=dict(where) if where is not None else None,
            where_document=dict(where_document) if where_document is not None else None,
            include=["documents", "metadatas", "distances"],
        )
        matches = []
        for index, record_id in enumerate(result["ids"][0]):
            distance = result["distances"][0][index]
            match = {
                "id": record_id,
                "document": result["documents"][0][index],
                "metadata": result["metadatas"][0][index] or {},
                "distance": float(distance),
            }
            if self.distance_metric == "cosine":
                match["similarity"] = 1.0 - float(distance)
            matches.append(match)
        return matches

    def search_text(
        self,
        query: str,
        model: Any,
        *,
        n_results: int = 5,
        where: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Encode a scientific text query with the project's embedding model."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        embedding = model.encode(query, convert_to_numpy=True)
        return self.search(embedding, n_results=n_results, where=where)

    def count(self) -> int:
        """Return the number of indexed chunks."""
        return self.collection.count()
