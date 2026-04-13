"""
RAGQueryAgent — Retrieval-Augmented Generation for enterprise Q&A.

1. Embeds the user query via ChromaDB.
2. Retrieves the top-k most relevant chunks.
3. Passes them to Claude to synthesise a grounded, cited answer.
"""

from .base_agent import BaseAgent, AgentResult
from vector_store import ChromaVectorStore
import config


_SYSTEM = """You are an expert enterprise knowledge assistant with access to a
vector store of company documents and images.

Rules:
- Answer ONLY from the provided context. If the answer is not in the context,
  say so clearly — do not hallucinate.
- Cite your sources: mention the document name and page number when available.
- Be concise but complete. Use bullet points for multi-part answers.
- If multiple documents are relevant, synthesise across them coherently."""


class RAGQueryAgent(BaseAgent):
    name = "RAGQueryAgent"
    description = "Retrieves relevant context from the vector store and generates grounded answers with Claude"
    emoji = "🔍"

    def __init__(self, vector_store: ChromaVectorStore):
        super().__init__()
        self._store = vector_store

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def process(self, *, query: str, top_k: int = config.TOP_K_RESULTS,
                doc_type: str | None = None, source_filter: str | None = None,
                **_) -> AgentResult:
        """
        Answer a query using RAG.

        Args:
            query:         Natural-language question.
            top_k:         Number of chunks to retrieve.
            doc_type:      Optional filter by doc_type ("pdf" | "image").
            source_filter: Optional filter to a specific filename.
        """
        self._log(f"Query: '{query[:80]}…'" if len(query) > 80 else f"Query: '{query}'")

        # Build ChromaDB where-filter
        where: dict | None = None
        if doc_type and source_filter:
            where = {"$and": [{"doc_type": doc_type}, {"source": source_filter}]}
        elif doc_type:
            where = {"doc_type": doc_type}
        elif source_filter:
            where = {"source": source_filter}

        # Retrieve
        self._log(f"Retrieving top-{top_k} chunks from vector store…")
        chunks = self._store.query(query, n_results=top_k, where=where)
        self._log(f"Retrieved {len(chunks)} chunks")

        if not chunks:
            return AgentResult(
                agent=self.name,
                success=True,
                data={"answer": "No relevant documents found in the knowledge base.", "sources": []},
                message="No matching content in vector store",
            )

        # Build context block
        context_parts: list[str] = []
        sources: list[dict] = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk["metadata"]
            src = meta.get("source", "unknown")
            page = meta.get("page_number", "")
            page_str = f" (page {page})" if page else ""
            context_parts.append(f"[Source {i}: {src}{page_str}]\n{chunk['document']}")
            sources.append({
                "rank": i,
                "source": src,
                "page": page,
                "doc_type": meta.get("doc_type", "unknown"),
                "distance": chunk.get("distance"),
            })

        context = "\n\n---\n\n".join(context_parts)
        user_message = f"Context from the knowledge base:\n\n{context}\n\n---\n\nQuestion: {query}"

        # Generate answer
        self._log("Generating answer with Claude…")
        answer = self._call_claude(_SYSTEM, user_message, max_tokens=2048)
        self._log("Answer generated")

        return AgentResult(
            agent=self.name,
            success=True,
            data={"answer": answer, "sources": sources, "query": query},
            message=f"Answered using {len(chunks)} context chunks",
            metadata={"chunk_count": len(chunks), "sources": [s["source"] for s in sources]},
        )
