"""
DocumentIngestionAgent — processes large PDFs for enterprise use.

Handles 300+ page documents by:
  1. Extracting text page-by-page with pypdf.
  2. Splitting into overlapping chunks.
  3. Embedding and storing every chunk in ChromaDB.
  4. Returning per-page stats for the demo dashboard.
"""

import io
import uuid
from typing import Any

from pypdf import PdfReader

from .base_agent import BaseAgent, AgentResult
from vector_store import ChromaVectorStore
import config


_SYSTEM = """You are a document intelligence specialist. When given a page from
a business document, extract the key facts, entities, and semantic meaning in a
compact form suitable for enterprise search. Focus on actionable information."""


class DocumentIngestionAgent(BaseAgent):
    name = "DocumentIngestionAgent"
    description = "Ingests PDFs (including 300+ page documents) into the vector store with smart chunking"
    emoji = "📄"

    def __init__(self, vector_store: ChromaVectorStore):
        super().__init__()
        self._store = vector_store

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def process(self, *, pdf_bytes: bytes, filename: str = "document.pdf",
                extra_meta: dict | None = None, **_) -> AgentResult:
        """
        Ingest a PDF document into the vector store.

        Args:
            pdf_bytes:   Raw bytes of the PDF file.
            filename:    Original filename (used as the source identifier).
            extra_meta:  Additional metadata to attach to every chunk
                         (e.g. {"department": "underwriting", "policy_id": "P-1234"}).
        """
        self._log(f"Received PDF '{filename}' ({len(pdf_bytes):,} bytes)")

        reader = PdfReader(io.BytesIO(pdf_bytes))
        total_pages = len(reader.pages)
        self._log(f"Detected {total_pages} pages")

        doc_id = str(uuid.uuid4())
        base_meta = {
            "source": filename,
            "doc_type": "pdf",
            "doc_id": doc_id,
            "total_pages": str(total_pages),
            **(extra_meta or {}),
        }

        texts: list[str] = []
        metadatas: list[dict] = []
        ids: list[str] = []
        page_stats: list[dict] = []

        for page_num, page in enumerate(reader.pages, start=1):
            raw_text = page.extract_text() or ""
            if not raw_text.strip():
                page_stats.append({"page": page_num, "chunks": 0, "chars": 0})
                continue

            chunks = self._chunk_text(raw_text, filename, page_num)
            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_p{page_num}_c{chunk_idx}"
                texts.append(chunk)
                metadatas.append({
                    **base_meta,
                    "page_number": str(page_num),
                    "chunk_index": str(chunk_idx),
                })
                ids.append(chunk_id)

            page_stats.append({
                "page": page_num,
                "chunks": len(chunks),
                "chars": len(raw_text),
            })

            if page_num % 50 == 0:
                self._log(f"  → processed {page_num}/{total_pages} pages…")

        if texts:
            # Batch insert — ChromaDB handles large batches well
            batch_size = 500
            for i in range(0, len(texts), batch_size):
                self._store.add_documents(
                    texts=texts[i:i + batch_size],
                    metadatas=metadatas[i:i + batch_size],
                    ids=ids[i:i + batch_size],
                )
                self._log(f"  → inserted batch {i // batch_size + 1} "
                          f"({min(i + batch_size, len(texts))}/{len(texts)} chunks)")

        total_chunks = len(texts)
        self._log(f"Ingestion complete: {total_chunks} chunks from {total_pages} pages")

        return AgentResult(
            agent=self.name,
            success=True,
            data={
                "doc_id": doc_id,
                "filename": filename,
                "total_pages": total_pages,
                "total_chunks": total_chunks,
                "page_stats": page_stats,
            },
            message=f"Ingested '{filename}': {total_pages} pages → {total_chunks} searchable chunks",
            metadata={"doc_id": doc_id, "total_chunks": total_chunks},
        )

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def _chunk_text(self, text: str, source: str, page_num: int) -> list[str]:
        """Split text into overlapping chunks of ~CHUNK_SIZE characters."""
        size = config.CHUNK_SIZE
        overlap = config.CHUNK_OVERLAP
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start = end - overlap  # slide with overlap
        return chunks or [text.strip()]
