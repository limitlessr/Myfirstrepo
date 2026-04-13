"""
FastAPI backend — the AI assistant HTTP interface.

Endpoints:
  POST /upload/image     — analyse an image
  POST /upload/document  — ingest a PDF
  POST /query            — RAG query
  POST /chat             — conversational interface (with optional file)
  GET  /documents        — list indexed sources
  GET  /agents/status    — status of all agents
  GET  /stats            — vector store statistics
  DELETE /documents/{source} — remove a document
"""

import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from vector_store import ChromaVectorStore
from agents import OrchestratorAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-22s  %(levelname)s  %(message)s",
)
logger = logging.getLogger("assistant.api")

# ── Global singletons (created at startup) ─────────────────────────────────
_store: ChromaVectorStore | None = None
_orchestrator: OrchestratorAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _store, _orchestrator
    logger.info("Initialising vector store and agents…")
    _store = ChromaVectorStore()
    _orchestrator = OrchestratorAgent(_store)
    logger.info("All agents ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Enterprise Vector Store — AI Assistant",
    description=(
        "Multi-agent AI backend powered by Claude. "
        "Supports image analysis, large-document ingestion, and RAG queries."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ────────────────────────────────────────────────────────────────

def get_orchestrator() -> OrchestratorAgent:
    if _orchestrator is None:
        raise HTTPException(503, "Agents not yet initialised")
    return _orchestrator


# ── Pydantic schemas ───────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    doc_type: str | None = None
    source_filter: str | None = None


class ChatRequest(BaseModel):
    message: str


class AgentResultResponse(BaseModel):
    agent: str
    success: bool
    message: str
    data: dict | list | str | None = None
    duration_ms: float = 0.0


# ── Routes ─────────────────────────────────────────────────────────────────

@app.post("/upload/image", summary="Analyse an image (cat detection, scene description, etc.)")
async def upload_image(
    file: Annotated[UploadFile, File(description="Image file (JPEG/PNG/WEBP/GIF)")],
    question: Annotated[str | None, Form()] = None,
) -> AgentResultResponse:
    orch = get_orchestrator()
    image_bytes = await file.read()
    result = orch.analyse_image(image_bytes, file.filename or "image.jpg", question=question)
    return AgentResultResponse(
        agent=result.agent,
        success=result.success,
        message=result.message,
        data=result.data,
        duration_ms=result.duration_ms,
    )


@app.post("/upload/document", summary="Ingest a PDF document into the vector store")
async def upload_document(
    file: Annotated[UploadFile, File(description="PDF document")],
    department: Annotated[str | None, Form()] = None,
    policy_id: Annotated[str | None, Form()] = None,
) -> AgentResultResponse:
    orch = get_orchestrator()
    pdf_bytes = await file.read()
    extra_meta: dict = {}
    if department:
        extra_meta["department"] = department
    if policy_id:
        extra_meta["policy_id"] = policy_id
    result = orch.ingest_document(pdf_bytes, file.filename or "document.pdf", extra_meta or None)
    return AgentResultResponse(
        agent=result.agent,
        success=result.success,
        message=result.message,
        data=result.data,
        duration_ms=result.duration_ms,
    )


@app.post("/query", summary="Semantic search + AI-generated answer (RAG)")
async def query_knowledge_base(req: QueryRequest) -> AgentResultResponse:
    orch = get_orchestrator()
    result = orch.query(req.query, top_k=req.top_k, doc_type=req.doc_type)
    return AgentResultResponse(
        agent=result.agent,
        success=result.success,
        message=result.message,
        data=result.data,
        duration_ms=result.duration_ms,
    )


@app.post("/chat", summary="Conversational AI assistant (auto-routes to correct agent)")
async def chat(
    message: Annotated[str, Form()],
    file: Annotated[UploadFile | None, File()] = None,
) -> AgentResultResponse:
    orch = get_orchestrator()
    attachments: dict = {}
    if file:
        content = await file.read()
        fname = file.filename or "attachment"
        if fname.lower().endswith(".pdf"):
            attachments = {"pdf_bytes": content, "filename": fname}
        else:
            attachments = {"image_bytes": content, "filename": fname}
    result = orch.chat(message, attachments or None)
    return AgentResultResponse(
        agent=result.agent,
        success=result.success,
        message=result.message,
        data=result.data,
        duration_ms=result.duration_ms,
    )


@app.get("/documents", summary="List all indexed sources")
async def list_documents():
    orch = get_orchestrator()
    result = orch.get_metadata(action="list")
    return result.data


@app.get("/stats", summary="Vector store statistics")
async def get_stats():
    orch = get_orchestrator()
    result = orch.get_metadata(action="stats")
    return result.data


@app.get("/agents/status", summary="Status of all running agents")
async def agents_status():
    orch = get_orchestrator()
    return orch.all_agent_statuses()


@app.delete("/documents/{source:path}", summary="Remove a document from the vector store")
async def delete_document(source: str):
    orch = get_orchestrator()
    result = orch.get_metadata(action="delete", source=source)
    if not result.success:
        raise HTTPException(400, result.message)
    return {"deleted": True, "message": result.message}


@app.get("/", summary="Health check")
async def root():
    return {
        "status": "ok",
        "service": "Enterprise Vector Store — AI Assistant",
        "docs": "/docs",
    }
