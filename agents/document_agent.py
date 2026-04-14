"""
DocumentIngestionAgent — processes large PDFs for enterprise use.

Supports 5 chunking strategies:
  - fixed      : Fixed character size with overlap (default, fast)
  - sentence   : Groups complete sentences up to the size limit
  - paragraph  : Splits on paragraph breaks (double newlines)
  - page       : One chunk per page — best when each page is self-contained
  - recursive  : Tries paragraph → sentence → word → character splits (best quality)
"""

import io
import re
import uuid

from pypdf import PdfReader

from .base_agent import BaseAgent, AgentResult
from vector_store import ChromaVectorStore
import config


# ── Strategy descriptions (shown in the UI) ────────────────────────────────

CHUNKING_STRATEGIES = {
    "recursive": {
        "label": "Recursive (Recommended)",
        "description": (
            "Tries to split on paragraphs first, then sentences, then words, "
            "then characters as a last resort. Produces the most natural, "
            "context-preserving chunks. Best for most documents."
        ),
        "supports_overlap": True,
    },
    "fixed": {
        "label": "Fixed Size",
        "description": (
            "Cuts text into chunks of exactly N characters with an overlap. "
            "Simple and predictable. Can split sentences mid-way. "
            "Good for uniform documents or when speed matters."
        ),
        "supports_overlap": True,
    },
    "sentence": {
        "label": "Sentence-based",
        "description": (
            "Groups complete sentences together until the chunk size is reached. "
            "Never cuts a sentence in half. Best for narrative text, "
            "reports, and legal clauses where sentence integrity matters."
        ),
        "supports_overlap": True,
    },
    "paragraph": {
        "label": "Paragraph-based",
        "description": (
            "Splits on blank lines (paragraph breaks). Keeps each paragraph "
            "intact. Best for structured documents like policies, contracts, "
            "or manuals where paragraphs are meaningful units."
        ),
        "supports_overlap": False,
    },
    "page": {
        "label": "Page-based",
        "description": (
            "Treats each PDF page as one chunk. If a page is too long it splits "
            "further using fixed-size chunking. Best when each page is "
            "self-contained (slides, forms, invoices)."
        ),
        "supports_overlap": False,
    },
}


class DocumentIngestionAgent(BaseAgent):
    name = "DocumentIngestionAgent"
    description = "Ingests PDFs into the vector store with configurable chunking strategies"
    emoji = "📄"

    def __init__(self, vector_store: ChromaVectorStore):
        super().__init__()
        self._store = vector_store

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def process(
        self,
        *,
        pdf_bytes: bytes,
        filename: str = "document.pdf",
        extra_meta: dict | None = None,
        strategy: str = "recursive",
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        min_chunk_length: int = 50,
        **_,
    ) -> AgentResult:
        """
        Ingest a PDF document into the vector store.

        Args:
            pdf_bytes:        Raw bytes of the PDF file.
            filename:         Original filename (source identifier).
            extra_meta:       Extra metadata tags for every chunk.
            strategy:         Chunking strategy — one of:
                              "recursive" | "fixed" | "sentence" |
                              "paragraph" | "page"
            chunk_size:       Max characters per chunk (default from config).
            chunk_overlap:    Overlap characters between chunks (default from config).
            min_chunk_length: Discard chunks shorter than this (removes noise).
        """
        size    = chunk_size    if chunk_size    is not None else config.CHUNK_SIZE
        overlap = chunk_overlap if chunk_overlap is not None else config.CHUNK_OVERLAP
        strategy = strategy if strategy in CHUNKING_STRATEGIES else "recursive"

        self._log(
            f"Received '{filename}' ({len(pdf_bytes):,} bytes) | "
            f"strategy={strategy} size={size} overlap={overlap} min_len={min_chunk_length}"
        )

        reader = PdfReader(io.BytesIO(pdf_bytes))
        total_pages = len(reader.pages)
        self._log(f"Detected {total_pages} pages")

        doc_id = str(uuid.uuid4())
        base_meta = {
            "source": filename,
            "doc_type": "pdf",
            "doc_id": doc_id,
            "total_pages": str(total_pages),
            "chunk_strategy": strategy,
            "chunk_size": str(size),
            "chunk_overlap": str(overlap),
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

            chunks = self._apply_strategy(raw_text, strategy, size, overlap, min_chunk_length)

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
                "avg_chunk_len": int(sum(len(c) for c in chunks) / len(chunks)) if chunks else 0,
            })

            if page_num % 50 == 0:
                self._log(f"  → processed {page_num}/{total_pages} pages…")

        if texts:
            batch_size = 500
            for i in range(0, len(texts), batch_size):
                self._store.add_documents(
                    texts=texts[i:i + batch_size],
                    metadatas=metadatas[i:i + batch_size],
                    ids=ids[i:i + batch_size],
                )
                self._log(
                    f"  → inserted batch {i // batch_size + 1} "
                    f"({min(i + batch_size, len(texts))}/{len(texts)} chunks)"
                )

        total_chunks = len(texts)
        avg_len = int(sum(len(t) for t in texts) / total_chunks) if total_chunks else 0
        self._log(f"Ingestion complete: {total_chunks} chunks, avg {avg_len} chars each")

        return AgentResult(
            agent=self.name,
            success=True,
            data={
                "doc_id": doc_id,
                "filename": filename,
                "total_pages": total_pages,
                "total_chunks": total_chunks,
                "avg_chunk_length": avg_len,
                "strategy": strategy,
                "chunk_size": size,
                "chunk_overlap": overlap,
                "page_stats": page_stats,
            },
            message=(
                f"Ingested '{filename}': {total_pages} pages → "
                f"{total_chunks} chunks [{strategy}, size={size}, overlap={overlap}]"
            ),
            metadata={"doc_id": doc_id, "total_chunks": total_chunks, "strategy": strategy},
        )

    # ------------------------------------------------------------------
    # Strategy dispatcher
    # ------------------------------------------------------------------

    def _apply_strategy(
        self,
        text: str,
        strategy: str,
        size: int,
        overlap: int,
        min_len: int,
    ) -> list[str]:
        if strategy == "fixed":
            chunks = self._chunk_fixed(text, size, overlap)
        elif strategy == "sentence":
            chunks = self._chunk_sentence(text, size, overlap)
        elif strategy == "paragraph":
            chunks = self._chunk_paragraph(text, size)
        elif strategy == "page":
            chunks = self._chunk_page(text, size)
        else:  # recursive (default)
            chunks = self._chunk_recursive(text, size, overlap)

        # Filter out noise — very short fragments
        return [c for c in chunks if len(c) >= min_len] or [text.strip()]

    # ------------------------------------------------------------------
    # Strategy 1 — Fixed Size
    # ------------------------------------------------------------------

    def _chunk_fixed(self, text: str, size: int, overlap: int) -> list[str]:
        """Split by exact character count with a sliding overlap window."""
        chunks: list[str] = []
        start = 0
        while start < len(text):
            chunk = text[start: start + size].strip()
            if chunk:
                chunks.append(chunk)
            if start + size >= len(text):
                break
            start += size - overlap
        return chunks or [text.strip()]

    # ------------------------------------------------------------------
    # Strategy 2 — Sentence-based
    # ------------------------------------------------------------------

    def _chunk_sentence(self, text: str, size: int, overlap: int) -> list[str]:
        """Group complete sentences into chunks without splitting them."""
        # Split on sentence-ending punctuation followed by whitespace
        sentence_pattern = re.compile(r'(?<=[.!?])\s+')
        sentences = [s.strip() for s in sentence_pattern.split(text) if s.strip()]

        chunks: list[str] = []
        current = ""

        for sentence in sentences:
            candidate = (current + " " + sentence).strip() if current else sentence
            if len(candidate) <= size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # Carry-over: last `overlap` characters for context continuity
                if overlap > 0 and current:
                    tail = current[-overlap:].strip()
                    current = (tail + " " + sentence).strip()
                else:
                    current = sentence

        if current.strip():
            chunks.append(current.strip())

        return chunks or [text.strip()]

    # ------------------------------------------------------------------
    # Strategy 3 — Paragraph-based
    # ------------------------------------------------------------------

    def _chunk_paragraph(self, text: str, size: int) -> list[str]:
        """
        Split on blank lines. Merge small consecutive paragraphs up to
        the size limit; split oversized single paragraphs with fixed chunking.
        """
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        chunks: list[str] = []
        current = ""

        for para in paragraphs:
            candidate = (current + "\n\n" + para).strip() if current else para
            if len(candidate) <= size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # Single paragraph larger than size — fall back to fixed
                if len(para) > size:
                    chunks.extend(self._chunk_fixed(para, size, 0))
                    current = ""
                else:
                    current = para

        if current.strip():
            chunks.append(current.strip())

        return chunks or [text.strip()]

    # ------------------------------------------------------------------
    # Strategy 4 — Page-based
    # ------------------------------------------------------------------

    def _chunk_page(self, text: str, size: int) -> list[str]:
        """
        Treat the whole page as one chunk. If it exceeds the size limit,
        fall back to fixed-size splitting with no overlap.
        """
        text = text.strip()
        if not text:
            return []
        if len(text) <= size:
            return [text]
        return self._chunk_fixed(text, size, overlap=0)

    # ------------------------------------------------------------------
    # Strategy 5 — Recursive (best quality)
    # ------------------------------------------------------------------

    def _chunk_recursive(self, text: str, size: int, overlap: int) -> list[str]:
        """
        Attempt to split using progressively finer separators:
        paragraph breaks → newlines → sentence ends → spaces → characters.
        Produces the most natural chunks.
        """
        separators = ["\n\n", "\n", r"(?<=[.!?])\s+", " ", ""]

        def _split(text: str, sep_list: list[str]) -> list[str]:
            if len(text) <= size:
                return [text.strip()] if text.strip() else []
            if not sep_list:
                # Character-level last resort
                return self._chunk_fixed(text, size, overlap)

            sep = sep_list[0]
            rest = sep_list[1:]

            # Split text by this separator
            parts = re.split(sep, text) if sep else list(text)
            parts = [p for p in parts if p and p.strip()]

            if len(parts) <= 1:
                # Separator not found — try next
                return _split(text, rest)

            chunks: list[str] = []
            current = ""

            for part in parts:
                joiner = "\n\n" if sep == "\n\n" else (" " if sep == " " else " ")
                candidate = (current + joiner + part).strip() if current else part.strip()

                if len(candidate) <= size:
                    current = candidate
                else:
                    if current:
                        chunks.append(current.strip())
                        # Overlap: carry last N characters forward
                        if overlap > 0:
                            tail = current[-overlap:].strip()
                            current = (tail + joiner + part).strip()
                        else:
                            current = part.strip()
                    else:
                        # Single part larger than size — recurse
                        chunks.extend(_split(part.strip(), rest))
                        current = ""

            if current.strip():
                chunks.append(current.strip())

            return chunks or [text.strip()]

        return _split(text, separators)

    # ------------------------------------------------------------------
    # Preview helper (called by Streamlit UI without hitting vector store)
    # ------------------------------------------------------------------

    def preview_chunks(
        self,
        text: str,
        strategy: str,
        chunk_size: int,
        chunk_overlap: int,
        min_chunk_length: int = 50,
    ) -> list[dict]:
        """Return chunk previews for a sample text (no DB writes)."""
        chunks = self._apply_strategy(text, strategy, chunk_size, chunk_overlap, min_chunk_length)
        return [
            {
                "chunk_#": i + 1,
                "length": len(c),
                "preview": c[:120] + ("…" if len(c) > 120 else ""),
            }
            for i, c in enumerate(chunks)
        ]
