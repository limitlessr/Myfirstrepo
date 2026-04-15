"""
ChromaDB vector store — persistent, enterprise-grade document store.
Supports text chunks from PDFs and image analysis summaries.
"""

import logging
import uuid
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

import config

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """Wrapper around ChromaDB with enterprise helpers."""

    def __init__(self, persist_dir: str = config.CHROMA_PERSIST_DIR,
                 collection_name: str = config.COLLECTION_NAME):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB collection '%s' ready at %s", collection_name, persist_dir)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_documents(
        self,
        texts: list[str],
        metadatas: list[dict],
        ids: list[str] | None = None,
    ) -> list[str]:
        """Embed and insert text chunks. Returns the list of stored IDs."""
        if not texts:
            return []
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        self._collection.add(documents=texts, metadatas=metadatas, ids=ids)
        logger.info("Stored %d chunks into collection '%s'", len(texts), self.collection_name)
        return ids

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        n_results: int = config.TOP_K_RESULTS,
        where: dict | None = None,
    ) -> list[dict[str, Any]]:
        """
        Semantic search. Returns a list of result dicts with keys:
        id, document, metadata, distance.
        """
        kwargs: dict[str, Any] = {"query_texts": [query_text], "n_results": n_results}
        if where:
            kwargs["where"] = where

        raw = self._collection.query(**kwargs)
        results = []
        for i, doc in enumerate(raw["documents"][0]):
            results.append({
                "id": raw["ids"][0][i],
                "document": doc,
                "metadata": raw["metadatas"][0][i] if raw["metadatas"] else {},
                "distance": raw["distances"][0][i] if raw.get("distances") else None,
            })
        return results

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    def delete_by_source(self, source: str) -> int:
        """Delete all chunks whose metadata.source matches *source*."""
        results = self._collection.get(where={"source": source})
        ids = results.get("ids", [])
        if ids:
            self._collection.delete(ids=ids)
        logger.info("Deleted %d chunks for source '%s'", len(ids), source)
        return len(ids)

    def list_sources(self) -> list[dict[str, Any]]:
        """Return a deduplicated list of indexed sources with chunk counts."""
        all_items = self._collection.get(include=["metadatas"])
        sources: dict[str, dict] = {}
        for meta in all_items.get("metadatas", []):
            src = meta.get("source", "unknown")
            if src not in sources:
                sources[src] = {
                    "source": src,
                    "doc_type": meta.get("doc_type", "unknown"),
                    "chunk_count": 0,
                }
            sources[src]["chunk_count"] += 1
        return list(sources.values())

    def stats(self) -> dict[str, Any]:
        """Return collection statistics."""
        count = self._collection.count()
        sources = self.list_sources()
        return {
            "total_chunks": count,
            "total_sources": len(sources),
            "collection": self.collection_name,
            "persist_dir": self.persist_dir,
        }

    def reset(self) -> None:
        """Drop and recreate the collection (used in tests / demos)."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning("Collection '%s' reset.", self.collection_name)

    # ------------------------------------------------------------------
    # Multi-collection management
    # ------------------------------------------------------------------

    def list_collections(self) -> list[dict[str, Any]]:
        """Return all collections with name, chunk count, and description."""
        result = []
        for col in self._client.list_collections():
            try:
                count = self._client.get_collection(
                    col.name, embedding_function=self._embedding_fn
                ).count()
            except Exception:
                count = 0
            result.append({
                "name":        col.name,
                "chunks":      count,
                "active":      col.name == self.collection_name,
                "description": (col.metadata or {}).get("description", ""),
            })
        return sorted(result, key=lambda x: x["name"])

    def create_new_collection(self, name: str, description: str = "") -> bool:
        """Create a new named collection. Returns True if created, False if already exists."""
        existing = [c.name for c in self._client.list_collections()]
        if name in existing:
            return False
        self._client.create_collection(
            name=name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine", "description": description},
        )
        logger.info("Created new collection '%s'", name)
        return True

    def delete_collection(self, name: str) -> bool:
        """Delete a collection by name. Cannot delete the active one."""
        if name == self.collection_name:
            logger.warning("Cannot delete the active collection '%s'", name)
            return False
        try:
            self._client.delete_collection(name)
            logger.info("Deleted collection '%s'", name)
            return True
        except Exception as e:
            logger.error("Failed to delete collection '%s': %s", name, e)
            return False

    def switch_collection(self, name: str) -> None:
        """Switch the active collection to *name* (must already exist)."""
        self.collection_name = name
        self._collection = self._client.get_or_create_collection(
            name=name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Switched active collection to '%s'", name)
