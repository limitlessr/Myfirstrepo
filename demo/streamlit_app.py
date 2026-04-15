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
def load_system(key: str, collection: str):
    store = ChromaVectorStore(collection_name=collection)
    orch  = OrchestratorAgent(store)
    return store, orch


# ── Active collection state ────────────────────────────────────────────────
if "active_collection" not in st.session_state:
    st.session_state.active_collection = config.COLLECTION_NAME

store, orchestrator = load_system(api_key, st.session_state.active_collection)

# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    # Collection selector
    st.subheader("📦 Active Collection")
    all_cols = store.list_collections()
    col_names = [c["name"] for c in all_cols] or [st.session_state.active_collection]

    selected = st.selectbox(
        "Collection",
        options=col_names,
        index=col_names.index(st.session_state.active_collection)
              if st.session_state.active_collection in col_names else 0,
        label_visibility="collapsed",
    )
    if selected != st.session_state.active_collection:
        st.session_state.active_collection = selected
        st.cache_resource.clear()
        st.rerun()

    # Stats for active collection
    stats_result = orchestrator.get_metadata(action="stats")
    stats = stats_result.data or {}
    col1, col2 = st.columns(2)
    col1.metric("Chunks", stats.get("total_chunks", 0))
    col2.metric("Sources", stats.get("total_sources", 0))

    if st.button("🔄 Refresh", use_container_width=True):
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

    # ── Expert Mode toggle ─────────────────────────────────────────────
    exp_col, _ = st.columns([1, 3])
    chat_expert = exp_col.toggle("🧪 Expert Mode", value=False, key="chat_expert",
                                 help="Unlock retrieval settings and see agent internals")

    # Defaults
    chat_top_k       = 5
    chat_source      = None
    chat_doc_type    = None
    show_routing     = False
    show_sources     = False

    if chat_expert:
        st.info("**Expert Mode on** — configure retrieval settings and inspect agent internals.")

        sources_list = [s["source"] for s in (store.list_sources() or [])]

        x1, x2, x3 = st.columns(3)

        chat_top_k = x1.slider(
            "Chunks to retrieve (top-k)",
            min_value=1, max_value=15, value=5,
            help="How many passages from the knowledge base are fetched before Claude writes the answer. More = broader context, slower response.",
        )
        chat_doc_type = x2.selectbox(
            "Filter by document type",
            options=["All", "pdf", "image"],
            index=0,
            help="Restrict the search to only PDFs, only image analyses, or search everything.",
        )
        chat_doc_type = None if chat_doc_type == "All" else chat_doc_type

        source_options = ["All documents"] + sources_list
        chosen_source = x3.selectbox(
            "Filter by document",
            options=source_options,
            index=0,
            help="Search only within a specific uploaded file.",
        )
        chat_source = None if chosen_source == "All documents" else chosen_source

        show_routing = st.checkbox("Show agent routing", value=True,
                                   help="Display which agent handled the request and why.")
        show_sources = st.checkbox("Show source passages", value=True,
                                   help="Show the exact chunks retrieved from the knowledge base to build the answer.")

        st.divider()
    else:
        st.caption("Using defaults: top-k **5** · all documents · all types. Enable Expert Mode to customise.")

    # ── Chat history ───────────────────────────────────────────────────
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if chat_expert and msg.get("routing"):
                with st.expander("🎯 Agent routing"):
                    st.json(msg["routing"])
            if chat_expert and msg.get("sources"):
                with st.expander(f"📄 Source passages used ({len(msg['sources'])})"):
                    for src in msg["sources"]:
                        st.markdown(
                            f"**{src['source']}** — page {src['page'] or '—'}  "
                            f"*(relevance score: {1 - (src.get('distance') or 0):.2f})*"
                        )

    # ── Input area ─────────────────────────────────────────────────────
    with st.form("chat_form", clear_on_submit=True):
        user_input    = st.text_input("Your message", placeholder="e.g. What does this document say about claims?")
        attached_file = st.file_uploader("Optional attachment (image or PDF)", key="chat_file")
        submitted     = st.form_submit_button("Send ➤")

    if submitted and user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        attachments = {}
        if attached_file:
            fname   = attached_file.name
            content = attached_file.read()
            if fname.lower().endswith(".pdf"):
                attachments = {"pdf_bytes": content, "filename": fname}
            else:
                attachments = {"image_bytes": content, "filename": fname}

        with st.spinner("Agents working…"):
            result = orchestrator.chat(
                user_input,
                attachments or None,
                top_k=chat_top_k,
                source_filter=chat_source,
                doc_type=chat_doc_type,
            )

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

        entry = {"role": "assistant", "content": reply}
        if result.data and isinstance(result.data, dict):
            entry["sources"]  = result.data.get("sources", [])
        entry["routing"] = result.metadata.get("routing", {})

        st.session_state.chat_history.append(entry)
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

    st.divider()

    # ── Expert Mode toggle ─────────────────────────────────────────────
    expert_col, _ = st.columns([1, 3])
    expert_mode = expert_col.toggle("🧪 Expert Mode", value=False,
                                    help="Unlock chunking strategy and algorithm parameters")

    # Defaults used when Expert Mode is off
    chosen_strategy  = "recursive"
    chosen_label     = "Recursive (Recommended)"
    chunk_size       = 1000
    chunk_overlap    = 200
    min_chunk_len    = 50
    supports_overlap = True

    if expert_mode:
        st.info("**Expert Mode on** — configure how the document is split before being stored in the vector store.")

        # ── Strategy picker ────────────────────────────────────────────
        st.subheader("① Chunking Strategy")
        st.caption("Choose the algorithm that decides where to cut the document into pieces.")

        strategy_keys   = list(CHUNKING_STRATEGIES.keys())
        strategy_labels = [CHUNKING_STRATEGIES[k]["label"] for k in strategy_keys]
        chosen_label    = st.radio(
            "Strategy",
            strategy_labels,
            index=0,
            horizontal=True,
            label_visibility="collapsed",
        )
        chosen_strategy  = strategy_keys[strategy_labels.index(chosen_label)]
        strat_info       = CHUNKING_STRATEGIES[chosen_strategy]
        supports_overlap = strat_info["supports_overlap"]

        st.info(f"**{strat_info['label']}** — {strat_info['description']}")

        # ── Parameters ────────────────────────────────────────────────
        st.subheader("② Parameters")
        st.caption("Fine-tune how the chosen strategy behaves.")

        p1, p2, p3 = st.columns(3)

        chunk_size = p1.slider(
            "Chunk size (characters)",
            min_value=200, max_value=3000, value=1000, step=100,
            help="Max characters per chunk. Smaller = precise retrieval. Larger = more context.",
        )
        chunk_overlap = p2.slider(
            "Overlap (characters)",
            min_value=0, max_value=500,
            value=200 if supports_overlap else 0,
            step=50,
            disabled=not supports_overlap,
            help="Characters repeated between consecutive chunks to avoid losing context at boundaries. Disabled for Paragraph and Page strategies.",
        )
        min_chunk_len = p3.slider(
            "Min chunk length",
            min_value=0, max_value=200, value=50, step=10,
            help="Discard chunks shorter than this. Removes stray headers, page numbers, and blank lines.",
        )

        # ── Live preview ───────────────────────────────────────────────
        st.subheader("③ Live Chunk Preview")
        st.caption("See exactly how page 1 of your PDF will be split — without saving anything.")

        preview_btn = st.button("👁️ Preview chunks from first page", disabled=pdf_file is None)

        if preview_btn and pdf_file:
            with st.spinner("Extracting first page…"):
                raw = pypdf.PdfReader(io.BytesIO(pdf_file.getvalue()))
                page_text = ""
                for p in raw.pages[:3]:
                    page_text = p.extract_text() or ""
                    if page_text.strip():
                        break

            if not page_text.strip():
                st.error("No text found in the first pages — this PDF may be scanned.")
            else:
                helper   = _DocAgent.__new__(_DocAgent)
                previews = helper.preview_chunks(
                    page_text, chosen_strategy, chunk_size, chunk_overlap, min_chunk_len
                )
                st.success(
                    f"**{len(previews)} chunks** would be created from this page "
                    f"using **{chosen_label}** (size={chunk_size}, overlap={chunk_overlap})."
                )
                for row in previews:
                    with st.expander(f"Chunk {row['chunk_#']}  —  {row['length']} chars"):
                        st.text(row["preview"])

        st.divider()

    else:
        # Show a compact summary of what defaults will be used
        st.caption(
            "Using defaults: **Recursive** strategy · chunk size **1000** · overlap **200** · min length **50**. "
            "Enable Expert Mode above to customise."
        )

    st.divider()

    # ── Ingest button ──────────────────────────────────────────────────
    ingest_btn = st.button("📥 Ingest Document", type="primary", disabled=pdf_file is None)

    if ingest_btn and pdf_file:
        pdf_bytes = pdf_file.read()

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
    st.caption("Browse documents, search the vector store, and manage collections.")

    # ── Expert Mode toggle ─────────────────────────────────────────────
    kb_exp_col, _ = st.columns([1, 3])
    kb_expert = kb_exp_col.toggle("🧪 Expert Mode", value=False, key="kb_expert",
                                   help="Unlock collection create, switch, and delete controls")

    # ══ Section 1 — Collection Manager (Expert Mode only) ═════════════
    if kb_expert:
        st.subheader("📦 Collection Manager")
        st.caption(
            "Collections are separate knowledge bases — one per team, project, or document type. "
            "Switch between them in the sidebar. Active collection is used by all agents."
        )

        collections = store.list_collections()

        if not collections:
            st.info("No collections found — the default will be created on first upload.")
        else:
            cols = st.columns(min(len(collections), 3))
            for idx, col_info in enumerate(collections):
                with cols[idx % 3]:
                    is_active = col_info["active"]
                    border = "🟢" if is_active else "⚪"
                    st.markdown(f"**{border} {col_info['name']}**")
                    st.caption(col_info.get("description") or "No description")
                    st.markdown(f"`{col_info['chunks']}` chunks")

                    if is_active:
                        st.success("Active", icon="✅")
                    else:
                        bcol1, bcol2 = st.columns(2)
                        if bcol1.button("Switch", key=f"sw_{col_info['name']}", use_container_width=True):
                            st.session_state.active_collection = col_info["name"]
                            st.cache_resource.clear()
                            st.rerun()
                        if bcol2.button("Delete", key=f"dl_{col_info['name']}", use_container_width=True):
                            if store.delete_collection(col_info["name"]):
                                st.success(f"Deleted '{col_info['name']}'")
                                st.rerun()
                            else:
                                st.error("Could not delete.")

        st.divider()
        st.markdown("**Create a new collection**")
        nc1, nc2, nc3 = st.columns([2, 3, 1])
        new_col_name = nc1.text_input(
            "Name", placeholder="e.g. underwriting_2024",
            label_visibility="collapsed", key="new_col_name",
            help="Use only letters, numbers, hyphens, and underscores.",
        )
        new_col_desc = nc2.text_input(
            "Description (optional)", placeholder="e.g. Underwriting policies for 2024",
            label_visibility="collapsed", key="new_col_desc",
        )
        if nc3.button("➕ Create", use_container_width=True):
            if not new_col_name.strip():
                st.warning("Please enter a collection name.")
            else:
                created = store.create_new_collection(new_col_name.strip(), new_col_desc.strip())
                if created:
                    st.success(f"Collection **'{new_col_name}'** created. Switch to it from the sidebar.")
                    st.rerun()
                else:
                    st.error(f"A collection named **'{new_col_name}'** already exists.")

        st.divider()

    # ══ Section 2 — Documents in active collection ════════════════════
    st.subheader(f"📁 Documents in '{st.session_state.active_collection}'")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        list_result = orchestrator.get_metadata(action="list")
        sources = (list_result.data or {}).get("sources", [])
        if not sources:
            st.warning("No documents indexed in this collection yet.")
        else:
            for src in sources:
                with st.expander(f"📄 {src['source']}"):
                    st.write(f"**Type:** {src['doc_type']}")
                    st.write(f"**Chunks:** {src['chunk_count']}")
                    if st.button("🗑️ Delete document", key=f"del_{src['source']}"):
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
