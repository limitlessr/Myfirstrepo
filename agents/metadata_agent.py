"""
MetadataAgent — tracks document provenance, status, and governance.

Responsibilities:
  • List all indexed sources with statistics.
  • Generate document inventory reports.
  • Handle document deletion / retirement.
  • Answer "what do we know about X?" by inspecting metadata.
"""

from .base_agent import BaseAgent, AgentResult
from vector_store import ChromaVectorStore


_SYSTEM = """You are an enterprise document governance specialist.
Given a list of indexed documents with metadata, generate a clear, structured
inventory report that an executive or compliance officer would find useful.
Include: total documents, types, sizes, and any notable patterns."""


class MetadataAgent(BaseAgent):
    name = "MetadataAgent"
    description = "Manages document provenance, inventory reports, and governance metadata"
    emoji = "🗂️"

    def __init__(self, vector_store: ChromaVectorStore):
        super().__init__()
        self._store = vector_store

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def process(self, *, action: str = "report", source: str | None = None,
                **_) -> AgentResult:
        """
        Args:
            action:  One of "report" | "list" | "delete" | "stats".
            source:  Filename to delete (only used with action="delete").
        """
        if action == "delete":
            return self._delete(source)
        if action == "list":
            return self._list()
        if action == "stats":
            return self._stats()
        return self._report()   # default: full report

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _list(self) -> AgentResult:
        self._log("Listing indexed sources…")
        sources = self._store.list_sources()
        self._log(f"Found {len(sources)} sources")
        return AgentResult(
            agent=self.name, success=True,
            data={"sources": sources, "count": len(sources)},
            message=f"{len(sources)} sources indexed",
        )

    def _stats(self) -> AgentResult:
        self._log("Fetching vector store statistics…")
        stats = self._store.stats()
        self._log(f"Stats: {stats}")
        return AgentResult(
            agent=self.name, success=True, data=stats,
            message=f"Vector store: {stats['total_chunks']} chunks across "
                    f"{stats['total_sources']} sources",
        )

    def _delete(self, source: str | None) -> AgentResult:
        if not source:
            return AgentResult(agent=self.name, success=False,
                               data=None, message="No source specified for deletion")
        self._log(f"Deleting source '{source}'…")
        deleted = self._store.delete_by_source(source)
        return AgentResult(
            agent=self.name, success=True,
            data={"deleted_chunks": deleted, "source": source},
            message=f"Removed {deleted} chunks for '{source}'",
        )

    def _report(self) -> AgentResult:
        self._log("Generating inventory report…")
        sources = self._store.list_sources()
        stats = self._store.stats()

        if not sources:
            return AgentResult(
                agent=self.name, success=True,
                data={"report": "No documents indexed yet.", "stats": stats},
                message="Empty knowledge base",
            )

        inventory_text = "\n".join(
            f"- {s['source']} [{s['doc_type']}]: {s['chunk_count']} chunks"
            for s in sources
        )
        prompt = (
            f"Enterprise Knowledge Base Inventory\n\n"
            f"Total chunks: {stats['total_chunks']}\n"
            f"Total sources: {stats['total_sources']}\n\n"
            f"Document list:\n{inventory_text}\n\n"
            f"Please generate a professional inventory report."
        )
        report = self._call_claude(_SYSTEM, prompt, max_tokens=512)
        self._log("Report generated")
        return AgentResult(
            agent=self.name, success=True,
            data={"report": report, "stats": stats, "sources": sources},
            message=f"Inventory report generated for {len(sources)} sources",
        )
