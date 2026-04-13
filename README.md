# Enterprise Vector Store — Multi-Agent AI Platform

Powered by **Claude (Anthropic)** + **ChromaDB**. Five specialized agents work together via an intelligent orchestrator to handle image analysis, large-document ingestion, and enterprise RAG queries.

---

## Architecture

```
                        ┌─────────────────────────────────┐
                        │        OrchestratorAgent  🎯     │
                        │   (Claude-powered routing)       │
                        └──┬──────┬──────┬──────┬─────────┘
                           │      │      │      │
               ┌───────────┘   ┌──┘   ┌──┘   ┌──┘
               ▼               ▼      ▼      ▼
        ┌──────────┐  ┌─────────┐  ┌──────┐  ┌──────────┐
        │ Image    │  │Document │  │ RAG  │  │Metadata  │
        │ Agent 🖼️ │  │Agent 📄 │  │Query │  │Agent 🗂️  │
        │(Vision)  │  │(Ingest) │  │🔍    │  │(Govern.) │
        └────┬─────┘  └────┬────┘  └──┬───┘  └────┬─────┘
             │              │          │             │
             └──────────────┴──────────┴─────────────┘
                                    │
                         ┌──────────▼──────────┐
                         │   ChromaDB           │
                         │   Vector Store       │
                         │  (cosine similarity) │
                         └─────────────────────┘
```

### Agents

| Agent | Role |
|---|---|
| 🎯 **OrchestratorAgent** | Uses Claude to route any request to the right agent(s) |
| 🖼️ **ImageAnalysisAgent** | Claude Vision — detects cats, describes scenes, extracts text |
| 📄 **DocumentIngestionAgent** | Chunks & indexes PDFs — tested with 300+ page documents |
| 🔍 **RAGQueryAgent** | Semantic retrieval + Claude synthesis with source citations |
| 🗂️ **MetadataAgent** | Document governance, inventory reports, deletion |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your API key

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 3. Run the CLI demo (no UI needed)

```bash
python demo/demo_runner.py
```

### 4. Launch the Streamlit web UI

```bash
streamlit run demo/streamlit_app.py
```

### 5. Start the REST API

```bash
uvicorn main:app --reload
# Docs at http://localhost:8000/docs
```

---

## Demo Scenarios

### Image Analysis — Cat Detection

```python
from vector_store import ChromaVectorStore
from agents import OrchestratorAgent

store = ChromaVectorStore()
orch  = OrchestratorAgent(store)

with open("cat.jpg", "rb") as f:
    result = orch.analyse_image(f.read(), "cat.jpg", question="Is this a cat?")

print(result.data["is_cat"])      # True
print(result.data["analysis"])    # Full Claude vision description
```

### Underwriter — 300-page Policy Document

```python
with open("policy_300pages.pdf", "rb") as f:
    result = orch.ingest_document(
        f.read(),
        filename="policy_300pages.pdf",
        extra_meta={"department": "underwriting", "policy_id": "P-2024-001"},
    )

print(result.data["total_pages"])   # 300
print(result.data["total_chunks"])  # ~900 searchable chunks

# Now query it
answer = orch.query("What is the flood risk classification?")
print(answer.data["answer"])        # Grounded answer with page citations
```

### Conversational Interface

```python
# Auto-routes to the right agent
result = orch.chat("What does our policy say about reinsurance?")
result = orch.chat("Is this a cat?", attachments={"image_bytes": img, "filename": "photo.jpg"})
result = orch.chat("Ingest this policy", attachments={"pdf_bytes": pdf, "filename": "p.pdf"})
```

---

## REST API

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload/image` | Upload & analyse an image |
| `POST` | `/upload/document` | Ingest a PDF |
| `POST` | `/query` | RAG query |
| `POST` | `/chat` | Conversational (multipart, optional file) |
| `GET` | `/documents` | List indexed sources |
| `GET` | `/stats` | Vector store statistics |
| `GET` | `/agents/status` | All agent statuses |
| `DELETE` | `/documents/{source}` | Remove a document |

Interactive docs: `http://localhost:8000/docs`

---

## Project Layout

```
enterprise-vector-store/
├── config.py                  # Environment config
├── main.py                    # uvicorn entry point
├── requirements.txt
├── .env.example
├── vector_store/
│   └── chroma_store.py        # ChromaDB wrapper
├── agents/
│   ├── base_agent.py          # Abstract base + AgentResult
│   ├── image_agent.py         # Claude Vision
│   ├── document_agent.py      # PDF chunking + ingestion
│   ├── query_agent.py         # RAG Q&A
│   ├── metadata_agent.py      # Governance
│   └── orchestrator_agent.py  # Routing brain
├── api/
│   └── assistant.py           # FastAPI app
└── demo/
    ├── streamlit_app.py        # Web UI
    └── demo_runner.py          # CLI demo with rich output
```
