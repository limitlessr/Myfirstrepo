"""
OrchestratorAgent — the enterprise AI assistant's brain.

Uses Claude to understand incoming requests and route them to the correct
specialist agent(s). Supports multi-step pipelines (e.g. ingest → query).
"""

import json
from typing import Any

from .base_agent import BaseAgent, AgentResult
from .image_agent import ImageAnalysisAgent
from .document_agent import DocumentIngestionAgent
from .query_agent import RAGQueryAgent
from .metadata_agent import MetadataAgent
from vector_store import ChromaVectorStore


_ROUTING_SYSTEM = """You are the routing engine for an enterprise AI platform.
Given a user request, decide which agent(s) should handle it.

Available agents:
- "image"    : Analyse an image (cat detection, object recognition, scene description).
               Required input: image_bytes.
- "document" : Ingest a PDF document into the knowledge base.
               Required input: pdf_bytes.
- "query"    : Answer a question using the knowledge base (RAG).
               Required input: query text.
- "metadata" : Get stats, list indexed sources, or delete a document.
               Required input: action ("report"/"list"/"stats"/"delete").
- "pipeline" : Ingest a document THEN immediately answer a question about it.
               Required input: pdf_bytes + query.

Respond with a JSON object only — no prose:
{
  "agent": "<agent_name>",
  "reasoning": "<one sentence>",
  "params": { /* any extra params to pass through */ }
}"""


class OrchestratorAgent(BaseAgent):
    name = "OrchestratorAgent"
    description = "Routes requests to the right specialist agent using Claude reasoning"
    emoji = "🎯"

    def __init__(self, vector_store: ChromaVectorStore):
        super().__init__()
        self._store = vector_store
        self._agents: dict[str, BaseAgent] = {
            "image":    ImageAnalysisAgent(vector_store),
            "document": DocumentIngestionAgent(vector_store),
            "query":    RAGQueryAgent(vector_store),
            "metadata": MetadataAgent(vector_store),
        }

    # ------------------------------------------------------------------
    # Public convenience methods (used by the API / demo)
    # ------------------------------------------------------------------

    def analyse_image(self, image_bytes: bytes, filename: str,
                      question: str | None = None) -> AgentResult:
        self._log(f"Routing image analysis for '{filename}'")
        return self._agents["image"].run(
            image_bytes=image_bytes, filename=filename, question=question,
            description=f"analyse {filename}",
        )

    def ingest_document(self, pdf_bytes: bytes, filename: str,
                        extra_meta: dict | None = None) -> AgentResult:
        self._log(f"Routing document ingestion for '{filename}'")
        return self._agents["document"].run(
            pdf_bytes=pdf_bytes, filename=filename, extra_meta=extra_meta,
            description=f"ingest {filename}",
        )

    def query(self, text: str, top_k: int = 5,
              doc_type: str | None = None) -> AgentResult:
        self._log(f"Routing query: '{text[:60]}'")
        return self._agents["query"].run(
            query=text, top_k=top_k, doc_type=doc_type,
            description="RAG query",
        )

    def get_metadata(self, action: str = "report",
                     source: str | None = None) -> AgentResult:
        self._log(f"Routing metadata action: '{action}'")
        return self._agents["metadata"].run(
            action=action, source=source,
            description=f"metadata {action}",
        )

    # ------------------------------------------------------------------
    # Intelligent chat routing (natural language → agent)
    # ------------------------------------------------------------------

    def chat(self, message: str, attachments: dict[str, Any] | None = None) -> AgentResult:
        """
        Handle a free-form chat message and dispatch to the right agent.

        attachments can contain:
          {"image_bytes": bytes, "filename": str}  or
          {"pdf_bytes": bytes, "filename": str}
        """
        attachments = attachments or {}
        self._log(f"Chat message received: '{message[:80]}'")

        # Build routing prompt
        attach_desc = ""
        if "image_bytes" in attachments:
            attach_desc = f"\nAttachment: image file '{attachments.get('filename','image')}'"
        elif "pdf_bytes" in attachments:
            attach_desc = (f"\nAttachment: PDF file '{attachments.get('filename','doc.pdf')}' "
                           f"({len(attachments['pdf_bytes']):,} bytes)")

        routing_prompt = f"User message: {message}{attach_desc}"
        raw_json = self._call_claude(_ROUTING_SYSTEM, routing_prompt, max_tokens=256)

        # Parse routing decision
        try:
            decision = json.loads(raw_json.strip().strip("```json").strip("```"))
        except json.JSONDecodeError:
            # Fallback: if we have pdf try document, if query text do query
            if "pdf_bytes" in attachments:
                decision = {"agent": "document", "reasoning": "PDF attachment detected", "params": {}}
            elif "image_bytes" in attachments:
                decision = {"agent": "image", "reasoning": "Image attachment detected", "params": {}}
            else:
                decision = {"agent": "query", "reasoning": "Default to RAG query", "params": {}}

        chosen = decision.get("agent", "query")
        reasoning = decision.get("reasoning", "")
        self._log(f"Routing decision: {chosen} — {reasoning}")

        # Dispatch
        if chosen == "image" and "image_bytes" in attachments:
            result = self._agents["image"].run(
                image_bytes=attachments["image_bytes"],
                filename=attachments.get("filename", "image.jpg"),
                question=message,
                description="image analysis via chat",
            )

        elif chosen == "document" and "pdf_bytes" in attachments:
            result = self._agents["document"].run(
                pdf_bytes=attachments["pdf_bytes"],
                filename=attachments.get("filename", "document.pdf"),
                description="document ingestion via chat",
            )

        elif chosen == "pipeline" and "pdf_bytes" in attachments:
            # Ingest first, then query
            ingest_result = self._agents["document"].run(
                pdf_bytes=attachments["pdf_bytes"],
                filename=attachments.get("filename", "document.pdf"),
                description="pipeline: ingest step",
            )
            if ingest_result.success:
                result = self._agents["query"].run(
                    query=message,
                    source_filter=attachments.get("filename"),
                    description="pipeline: query step",
                )
                result.metadata["ingest"] = ingest_result.data
            else:
                result = ingest_result

        elif chosen == "metadata":
            action = decision.get("params", {}).get("action", "report")
            result = self._agents["metadata"].run(
                action=action, description=f"metadata {action}",
            )

        else:
            # Default: RAG query
            result = self._agents["query"].run(
                query=message,
                description="RAG query via chat",
            )

        result.metadata["routing"] = {"agent": chosen, "reasoning": reasoning}
        return result

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def all_agent_statuses(self) -> list[dict]:
        statuses = [self.status()]
        for agent in self._agents.values():
            statuses.append(agent.status())
        return statuses

    def process(self, **kwargs) -> AgentResult:
        """Generic entry-point (delegates to chat)."""
        return self.chat(
            message=kwargs.get("message", ""),
            attachments=kwargs.get("attachments"),
        )
