"""
Enterprise Vector Store — Streamlit Demo UI

Launch with:
    streamlit run demo/streamlit_app.py
"""

import os
import sys
import time
import logging
import io
import pypdf

import streamlit as st

# Make repo root importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO)

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Enterprise Vector Store",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── API Key gate ───────────────────────────────────────────────────────────
# Allow key to come from env, .env file, or the sidebar input.
_env_key = os.environ.get("ANTHROPIC_API_KEY", "")

with st.sidebar:
    st.title("🏢 Enterprise Vector Store")
    st.caption("Powered by Claude + ChromaDB")
    st.divider()

    if _env_key:
        api_key = _env_key
        st.success("API key loaded from environment.")
    else:
        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            placeholder="sk-ant-...",
            help="Get yours at https://console.anthropic.com",
        )

if not api_key:
    st.info("Enter your **Anthropic API key** in the sidebar to start.")
    st.stop()

# Inject key so config.py and the Anthropic client pick it up
os.environ["ANTHROPIC_API_KEY"] = api_key

from vector_store import ChromaVectorStore
from agents import OrchestratorAgent

# ── Shared state ───────────────────────────────────────────────────────────

@st.cache_resource
def load_system(key: str):  # key arg busts cache when key changes
    store = ChromaVectorStore()
    orch  = OrchestratorAgent(store)
    return store, orch


store, orchestrator = load_system(api_key)

# ── Sidebar — Agent Panel ──────────────────────────────────────────────────

with st.sidebar:
    st.subheader("Active Agents")
    agent_defs = [
        ("🎯", "OrchestratorAgent", "Routes & coordinates"),
        ("🖼️", "ImageAnalysisAgent", "Vision & cat detection"),
        ("📄", "DocumentIngestionAgent", "PDF → vector store"),
        ("🔍", "RAGQueryAgent", "Retrieval + synthesis"),
        ("🗂️", "MetadataAgent", "Governance & inventory"),
    ]
    for emoji, name, desc in agent_defs:
        st.success(f"{emoji} **{name}**  \n{desc}")

    st.divider()
    stats_result = orchestrator.get_metadata(action="stats")
    stats = stats_result.data or {}
    col1, col2 = st.columns(2)
    col1.metric("Total Chunks", stats.get("total_chunks", 0))
    col2.metric("Sources", stats.get("total_sources", 0))

    if st.button("🔄 Refresh Stats"):
        st.rerun()

# ── Main tabs ──────────────────────────────────────────────────────────────

tab_chat, tab_image, tab_doc, tab_store = st.tabs(
    ["💬 AI Assistant", "🖼️ Image Analysis", "📄 Document Upload", "🗄️ Knowledge Base"]
)

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — Chat / AI Assistant
# ══════════════════════════════════════════════════════════════════════════

with tab_chat:
    st.header("💬 Enterprise AI Assistant")
    st.caption("Ask anything. Attach an image or PDF. The Orchestrator routes your request to the right agent.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display conversation
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("meta"):
                with st.expander("Agent details"):
                    st.json(msg["meta"])

    # Input area
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Your message", placeholder="e.g. What does this document say about claims?")
        attached_file = st.file_uploader("Optional attachment (image or PDF)", key="chat_file")
        submitted = st.form_submit_button("Send ➤")

    if submitted and user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        attachments = {}
        if attached_file:
            fname = attached_file.name
            content = attached_file.read()
            if fname.lower().endswith(".pdf"):
                attachments = {"pdf_bytes": content, "filename": fname}
            else:
                attachments = {"image_bytes": content, "filename": fname}

        with st.spinner("Agents working…"):
            result = orchestrator.chat(user_input, attachments or None)

        # Format answer
        if result.success and result.data:
            data = result.data
            if isinstance(data, dict) and "answer" in data:
                reply = data["answer"]
            elif isinstance(data, dict) and "analysis" in data:
                reply = data["analysis"]
                if data.get("is_cat"):
                    reply = "🐱 **Cat detected!**\n\n" + reply
            elif isinstance(data, dict) and "report" in data:
                reply = data["report"]
            elif isinstance(data, dict) and "total_chunks" in data:
                reply = f"Ingested successfully! **{data.get('total_chunks',0)} chunks** from {data.get('total_pages',0)} pages."
            else:
                reply = result.message
        else:
            reply = f"Error: {result.message}"

        meta = {
            "agent": result.agent,
            "duration_ms": f"{result.duration_ms:.0f}ms",
            "routing": result.metadata.get("routing", {}),
        }
        st.session_state.chat_history.append({"role": "assistant", "content": reply, "meta": meta})
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — Image Analysis
# ══════════════════════════════════════════════════════════════════════════

with tab_image:
    st.header("🖼️ Image Analysis Agent")
    st.caption("Upload any image. The agent uses Claude Vision to describe it and check if there's a cat.")

    col_upload, col_result = st.columns([1, 1])

    with col_upload:
        img_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp", "gif"], key="img_tab")
        question = st.text_input("Optional question", placeholder="Is there a cat? What breed?")
        store_flag = st.checkbox("Store analysis in knowledge base", value=True)

        if img_file:
            st.image(img_file, caption=img_file.name, use_column_width=True)

        analyse_btn = st.button("🔍 Analyse Image", type="primary", disabled=img_file is None)

    with col_result:
        if analyse_btn and img_file:
            img_bytes = img_file.read()
            with st.spinner("ImageAnalysisAgent working…"):
                t0 = time.time()
                result = orchestrator.analyse_image(
                    img_bytes, img_file.name,
                    question=question or None,
                )
                elapsed = (time.time() - t0) * 1000

            if result.success:
                data = result.data
                # Cat verdict banner
                if data.get("is_cat"):
                    st.success("🐱 YES — A CAT IS PRESENT!")
                else:
                    st.info("🚫 No cat detected.")

                st.subheader("Analysis")
                st.write(data.get("analysis", ""))

                with st.expander("Metadata"):
                    st.json({
                        "filename": data.get("filename"),
                        "mime_type": data.get("mime_type"),
                        "image_size_bytes": data.get("image_size_bytes"),
                        "doc_id": data.get("doc_id"),
                        "duration_ms": f"{elapsed:.0f}ms",
                    })
            else:
                st.error(f"Analysis failed: {result.message}")

# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — Document Upload
# ══════════════════════════════════════════════════════════════════════════

with tab_doc:
    st.header("📄 Document Ingestion Agent")
    st.caption("Upload PDF documents — even 300+ pages. Choose your chunking strategy and parameters below.")

    from agents.document_agent import CHUNKING_STRATEGIES, DocumentIngestionAgent as _DocAgent

    # ── PDF upload ─────────────────────────────────────────────────────
    with st.expander("ℹ️ What kind of PDF will work?"):
        st.markdown("""
| PDF Type | Works? | Notes |
|---|---|---|
| Exported from Word / Google Docs | ✅ Yes | Best results |
| Typed digital document | ✅ Yes | Works great |
| Scanned / photographed pages | ❌ No | No text to extract |
| Password protected | ❌ No | Can't be opened |
| Mostly images with little text | ⚠️ Partial | Only text portions indexed |
        """)

    pdf_file = st.file_uploader("Upload a PDF (up to 500 MB)", type=["pdf"], key="pdf_tab")

    if pdf_file:
        file_size_mb = len(pdf_file.getvalue()) / (1024 * 1024)
        st.caption(f"File: **{pdf_file.name}**  |  Size: {file_size_mb:.1f} MB")
        if file_size_mb > 50:
            st.warning(f"Large file ({file_size_mb:.0f} MB) — ingestion may take 2–5 minutes.")

    c1, c2 = st.columns(2)
    dept   = c1.text_input("Department (optional)", placeholder="Underwriting")
    pol_id = c2.text_input("Policy ID (optional)",  placeholder="P-2024-001")

    st.divider()

    # ── Chunking settings panel ────────────────────────────────────────
    st.subheader("⚙️ Chunking Settings")
    st.caption("These control how the document is split before being stored. Different strategies suit different document types.")

    # Strategy picker
    strategy_keys   = list(CHUNKING_STRATEGIES.keys())
    strategy_labels = [CHUNKING_STRATEGIES[k]["label"] for k in strategy_keys]
    chosen_label    = st.radio(
        "Chunking Strategy",
        strategy_labels,
        index=0,
        horizontal=True,
    )
    chosen_strategy = strategy_keys[strategy_labels.index(chosen_label)]
    strat_info      = CHUNKING_STRATEGIES[chosen_strategy]

    st.info(f"**{strat_info['label']}** — {strat_info['description']}")

    # Parameters
    supports_overlap = strat_info["supports_overlap"]
    p1, p2, p3 = st.columns(3)

    chunk_size = p1.slider(
        "Chunk size (characters)",
        min_value=200,
        max_value=3000,
        value=1000,
        step=100,
        help="Maximum number of characters in one chunk. Smaller = more precise retrieval. Larger = more context per chunk.",
    )

    chunk_overlap = p2.slider(
        "Overlap (characters)",
        min_value=0,
        max_value=500,
        value=200 if supports_overlap else 0,
        step=50,
        disabled=not supports_overlap,
        help="How many characters the next chunk repeats from the previous one. Prevents losing meaning at boundaries. Not used by Paragraph or Page strategies.",
    )

    min_chunk_len = p3.slider(
        "Min chunk length",
        min_value=0,
        max_value=200,
        value=50,
        step=10,
        help="Chunks shorter than this are discarded. Removes headers, page numbers, and blank fragments.",
    )

    # ── Live preview ───────────────────────────────────────────────────
    st.divider()
    st.subheader("🔬 Live Chunk Preview")
    st.caption("See exactly how the first page of your PDF will be chunked with the current settings — before committing to ingest.")

    preview_btn = st.button("👁️ Preview chunks from first page", disabled=pdf_file is None)

    if preview_btn and pdf_file:
        with st.spinner("Extracting first page…"):
            raw  = pypdf.PdfReader(io.BytesIO(pdf_file.getvalue()))
            page_text = ""
            for p in raw.pages[:3]:   # try first 3 pages to find one with text
                page_text = p.extract_text() or ""
                if page_text.strip():
                    break

        if not page_text.strip():
            st.error("No text found in the first pages — this PDF may be scanned.")
        else:
            helper = _DocAgent.__new__(_DocAgent)
            previews = helper.preview_chunks(
                page_text, chosen_strategy, chunk_size, chunk_overlap, min_chunk_len
            )
            st.success(f"**{len(previews)} chunks** would be created from this page using **{chosen_label}** strategy.")

            for row in previews:
                with st.expander(f"Chunk {row['chunk_#']}  —  {row['length']} characters"):
                    st.text(row["preview"])

    st.divider()

    # ── Ingest button ──────────────────────────────────────────────────
    ingest_btn = st.button("📥 Ingest Document", type="primary", disabled=pdf_file is None)

    if ingest_btn and pdf_file:
        pdf_bytes = pdf_file.read()
        extra: dict = {}
        if dept:   extra["department"] = dept
        if pol_id: extra["policy_id"]  = pol_id

        status_box = st.empty()
        progress   = st.progress(0, text="Reading PDF…")
        status_box.info(
            f"📄 DocumentIngestionAgent is processing **{pdf_file.name}** "
            f"using **{chosen_label}** strategy…"
        )
        progress.progress(10, text="Extracting and chunking text…")

        result = orchestrator.ingest_document(
            pdf_bytes,
            pdf_file.name,
            extra or None,
            strategy=chosen_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap if supports_overlap else 0,
            min_chunk_length=min_chunk_len,
        )

        progress.progress(100, text="✅ Done!")

        if result.success:
            data         = result.data
            total_pages  = data.get("total_pages", 0)
            total_chunks = data.get("total_chunks", 0)
            avg_len      = data.get("avg_chunk_length", 0)
            page_stats   = data.get("page_stats", [])
            empty_pages  = sum(1 for p in page_stats if p.get("chars", 0) == 0)

            status_box.success(f"✅ {result.message}")

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Pages",          total_pages)
            c2.metric("Chunks created", total_chunks)
            c3.metric("Avg chunk size", f"{avg_len} chars")
            c4.metric("Empty pages",    empty_pages)
            c5.metric("Time taken",     f"{result.duration_ms / 1000:.1f}s")

            # Strategy summary badge
            st.markdown(
                f"**Strategy used:** `{data.get('strategy')}` &nbsp;|&nbsp; "
                f"**Chunk size:** `{data.get('chunk_size')}` &nbsp;|&nbsp; "
                f"**Overlap:** `{data.get('chunk_overlap')}`"
            )

            if total_pages > 0 and empty_pages / total_pages > 0.5:
                st.error(
                    f"⚠️ {empty_pages} of {total_pages} pages had no text. "
                    "This PDF is likely scanned. Try a digitally created PDF."
                )
            elif total_chunks == 0:
                st.error("No text could be extracted. The PDF may be scanned or image-only.")
            else:
                st.info("✅ Document is now searchable. Go to the **💬 AI Assistant** tab and ask questions about it.")

            with st.expander(f"Page-level breakdown (first 50 of {total_pages} pages)"):
                st.dataframe(page_stats[:50])
        else:
            status_box.error(f"Ingestion failed: {result.message}")

# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — Knowledge Base / Vector Store Explorer
# ══════════════════════════════════════════════════════════════════════════

with tab_store:
    st.header("🗄️ Knowledge Base Explorer")
    st.caption("Browse and manage everything indexed in the vector store.")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("Indexed Sources")
        list_result = orchestrator.get_metadata(action="list")
        sources = (list_result.data or {}).get("sources", [])
        if not sources:
            st.warning("No documents indexed yet.")
        else:
            for src in sources:
                with st.expander(f"📁 {src['source']}"):
                    st.write(f"**Type:** {src['doc_type']}")
                    st.write(f"**Chunks:** {src['chunk_count']}")
                    if st.button(f"🗑️ Delete", key=f"del_{src['source']}"):
                        del_result = orchestrator.get_metadata(action="delete", source=src["source"])
                        st.success(del_result.message) if del_result.success else st.error(del_result.message)
                        st.rerun()

    with col_right:
        st.subheader("Semantic Search")
        search_q = st.text_input("Search the knowledge base", placeholder="flood damage claims 2023")
        top_k = st.slider("Results to retrieve", 1, 10, 5)
        if st.button("🔎 Search") and search_q:
            chunks = store.query(search_q, n_results=top_k)
            if not chunks:
                st.info("No results found.")
            for i, chunk in enumerate(chunks, 1):
                meta = chunk["metadata"]
                with st.expander(
                    f"Result {i} — {meta.get('source','?')} "
                    f"(page {meta.get('page_number','?')}) "
                    f"[dist={chunk.get('distance',0):.3f}]"
                ):
                    st.text(chunk["document"][:500] + ("…" if len(chunk["document"]) > 500 else ""))

        st.divider()
        st.subheader("Inventory Report")
        if st.button("📋 Generate Report"):
            with st.spinner("MetadataAgent generating report…"):
                rep_result = orchestrator.get_metadata(action="report")
            if rep_result.success:
                st.markdown(rep_result.data.get("report", ""))
            else:
                st.error(rep_result.message)
