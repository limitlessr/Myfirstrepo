"""
Enterprise Vector Store — CLI Demo Runner

Demonstrates all 5 agents working together end-to-end.
Shows a realistic underwriter workflow:
  1. MetadataAgent  — check initial state
  2. DocumentIngestionAgent — ingest a 300-page insurance policy PDF (simulated)
  3. ImageAnalysisAgent — analyse a photo attachment (cat image test)
  4. RAGQueryAgent  — ask questions about the ingested policy
  5. OrchestratorAgent — intelligent chat routing demo
  6. MetadataAgent  — final inventory report

Run with:
    python demo/demo_runner.py
"""

import io
import os
import sys
import time
import textwrap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.rule import Rule
from rich.text import Text
from rich import box

from vector_store import ChromaVectorStore
from agents import (
    OrchestratorAgent, ImageAnalysisAgent, DocumentIngestionAgent,
    RAGQueryAgent, MetadataAgent,
)

console = Console()

# ══════════════════════════════════════════════════════════════════════════
# Sample data generators (no real files needed)
# ══════════════════════════════════════════════════════════════════════════

def make_sample_pdf_bytes(num_pages: int = 10) -> bytes:
    """
    Generate a minimal multi-page PDF in pure Python (no external libs needed).
    Content simulates an insurance underwriting policy document.
    """
    policy_content = [
        ("Executive Summary",
         "This Commercial Property Insurance Policy provides comprehensive coverage for "
         "insured premises against fire, flood, theft, and natural disasters. Coverage limit: "
         "$50,000,000. Premium: $125,000 annually. Effective date: 2024-01-01."),

        ("Risk Assessment — Flood",
         "The insured property at 1200 Harbor Drive is classified as Flood Zone B. "
         "Historical flood events: 3 in the past 20 years. Maximum single-event loss "
         "estimate: $2,400,000. Mitigation measures: flood barriers installed 2022."),

        ("Risk Assessment — Fire",
         "Fire suppression systems: wet pipe sprinklers throughout. Last inspection: "
         "2023-09-15. Building construction: Type II non-combustible. Adjacent exposure: "
         "low. Estimated fire loss: $800,000 maximum probable loss."),

        ("Claims History",
         "2019: Water damage claim — $45,000 settled. 2021: Roof damage (hail) — $112,000 "
         "settled. 2022: Business interruption (supply chain) — $320,000 settled. "
         "3-year loss ratio: 38%. Claims frequency: below industry average."),

        ("Underwriting Guidelines — Commercial Property",
         "Acceptance criteria: Buildings constructed after 1980, occupancy rate > 80%, "
         "loss ratio < 65% over 5 years. Exclusions: war, nuclear, cyber-physical attacks. "
         "Co-insurance requirement: 90% of replacement cost value."),

        ("Coverage Endorsements",
         "Endorsement 001: Equipment Breakdown — $5M limit. "
         "Endorsement 002: Inland Marine — $2M limit. "
         "Endorsement 003: Business Income — 12-month period, $8M limit. "
         "Endorsement 004: Extra Expense — $500,000."),

        ("Premium Calculation",
         "Base rate: $1.25 per $1,000 of insured value. Sprinkler credit: -10%. "
         "Claims-free discount: -5%. Territory loading: +8%. "
         "Computed premium: $125,000. Minimum earned premium: $62,500."),

        ("Policy Conditions",
         "Insured must maintain all fire and security systems. Vacancy clause: 60 days. "
         "Inspection rights: insurer may inspect at any time with 48-hour notice. "
         "Cancellation: 30-day written notice. Non-renewal: 60-day notice required."),

        ("Reinsurance Treaty",
         "This policy is subject to the company's facultative reinsurance agreement. "
         "Retention: $10,000,000. Cession to reinsurers: $40,000,000. "
         "Reinsurers: Munich Re (50%), Swiss Re (30%), Hannover Re (20%)."),

        ("Regulatory Compliance",
         "Policy complies with state insurance department filing requirements. "
         "Form approval: DOI-2024-CP-001. Rate filing: DOI-2024-RATE-047. "
         "Anti-fraud statement included per statute. Privacy notice attached."),
    ]

    # Build a real PDF using fpdf2 if available, otherwise a minimal valid PDF
    try:
        from fpdf import FPDF
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Helvetica", size=11)
        pages_per_section = max(1, num_pages // len(policy_content))

        for i, (title, body) in enumerate(policy_content):
            for _ in range(pages_per_section):
                pdf.add_page()
                pdf.set_font("Helvetica", "B", 14)
                pdf.cell(0, 10, f"Section {i+1}: {title}", ln=True)
                pdf.set_font("Helvetica", size=11)
                pdf.multi_cell(0, 8, body)
                pdf.ln(5)
                pdf.set_font("Helvetica", "I", 9)
                pdf.cell(0, 8,
                         f"CONFIDENTIAL — Commercial Property Policy P-2024-001  |  Page {pdf.page_no()}",
                         ln=True)
        return pdf.output()

    except ImportError:
        # Fallback: hand-craft a minimal valid PDF
        pages = []
        for i, (title, body) in enumerate(policy_content):
            text = f"Section {i+1}: {title}\\n\\n{body}"
            pages.append(text)

        page_objs = []
        for p in pages:
            page_objs.append(p.replace("(", "\\(").replace(")", "\\)").replace("\n", "\\n"))

        lines = ["%PDF-1.4"]
        obj_offsets = {}
        obj_num = 1

        # Catalog
        obj_offsets[obj_num] = sum(len(l)+1 for l in lines)
        lines.append(f"{obj_num} 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj")
        obj_num += 1

        # Pages dict
        page_refs = " ".join(f"{i+3} 0 R" for i in range(len(pages)))
        obj_offsets[obj_num] = sum(len(l)+1 for l in lines)
        lines.append(f"{obj_num} 0 obj\n<< /Type /Pages /Kids [{page_refs}] /Count {len(pages)} >>\nendobj")
        obj_num += 1

        # Each page
        for idx, content in enumerate(page_objs):
            stream = f"BT /F1 12 Tf 50 750 Td ({content}) Tj ET"
            obj_offsets[obj_num] = sum(len(l)+1 for l in lines)
            lines.append(
                f"{obj_num} 0 obj\n"
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> "
                f"/Contents {obj_num+1} 0 R >>\nendobj"
            )
            obj_num += 1
            stream_bytes = stream.encode()
            obj_offsets[obj_num] = sum(len(l)+1 for l in lines)
            lines.append(
                f"{obj_num} 0 obj\n"
                f"<< /Length {len(stream_bytes)} >>\nstream\n{stream}\nendstream\nendobj"
            )
            obj_num += 1

        xref_offset = sum(len(l)+1 for l in lines)
        lines.append(f"xref\n0 {obj_num}")
        lines.append("0000000000 65535 f ")
        for i in range(1, obj_num):
            lines.append(f"{obj_offsets[i]:010d} 00000 n ")
        lines.append(f"trailer\n<< /Size {obj_num} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF")
        return "\n".join(lines).encode("latin-1")


def make_sample_image_bytes() -> bytes:
    """Generate a small JPEG with a cat-like pattern (solid colour with label)."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new("RGB", (320, 240), color=(230, 180, 100))
        draw = ImageDraw.Draw(img)
        draw.rectangle([60, 60, 260, 180], fill=(160, 100, 50))
        draw.ellipse([120, 30, 200, 110], fill=(200, 140, 80))
        draw.polygon([(100, 60), (80, 20), (120, 50)], fill=(200, 140, 80))  # ear
        draw.polygon([(200, 60), (220, 20), (180, 50)], fill=(200, 140, 80))  # ear
        draw.ellipse([137, 58, 157, 78], fill=(50, 30, 20))   # eye
        draw.ellipse([163, 58, 183, 78], fill=(50, 30, 20))   # eye
        draw.text((110, 190), "Sample Cat Image", fill=(0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except ImportError:
        # Minimal 1×1 white JPEG
        return bytes([
            0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
            0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
            0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09,
            0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
            0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20,
            0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29,
            0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32,
            0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01,
            0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00,
            0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
            0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x10, 0x00, 0x02, 0x01, 0x03,
            0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7D,
            0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06,
            0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08,
            0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72,
            0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
            0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45,
            0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
            0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75,
            0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
            0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3,
            0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6,
            0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9,
            0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
            0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4,
            0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01,
            0x00, 0x00, 0x3F, 0x00, 0xFB, 0xD4, 0xFF, 0xD9,
        ])


# ══════════════════════════════════════════════════════════════════════════
# Demo steps
# ══════════════════════════════════════════════════════════════════════════

def banner():
    console.print()
    console.print(Panel.fit(
        Text.assemble(
            ("  🏢  ENTERPRISE VECTOR STORE  🏢\n", "bold white on dark_blue"),
            ("  Multi-Agent AI Platform  |  Powered by Claude + ChromaDB\n", "cyan"),
            ("  Demonstrating 5 Agents Working Together", "italic yellow"),
        ),
        border_style="bright_blue",
        padding=(1, 4),
    ))
    console.print()


def step_header(n: int, title: str, agent: str, emoji: str):
    console.print()
    console.print(Rule(
        f"[bold yellow]Step {n}[/bold yellow]  {emoji}  [bold cyan]{agent}[/bold cyan]  —  {title}",
        style="bright_blue",
    ))
    console.print()


def print_result(result, extra_rows: list | None = None):
    color = "green" if result.success else "red"
    icon  = "✅" if result.success else "❌"
    console.print(f"  {icon}  [{color}]{result.message}[/{color}]  "
                  f"[dim]({result.duration_ms:.0f} ms)[/dim]")
    if extra_rows:
        tbl = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        tbl.add_column(style="dim")
        tbl.add_column()
        for k, v in extra_rows:
            tbl.add_row(k, str(v))
        console.print(tbl)


def run_demo():
    banner()

    # ── Setup ──────────────────────────────────────────────────────────
    console.print("[bold]Initialising system…[/bold]")
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), transient=True) as prog:
        t = prog.add_task("Loading ChromaDB + Agents…")
        store = ChromaVectorStore()
        store.reset()           # clean slate for demo
        orch  = OrchestratorAgent(store)
        prog.update(t, description="Ready!")
        time.sleep(0.5)

    # Agent roster
    tbl = Table(title="Active Agent Roster", box=box.ROUNDED, border_style="bright_blue")
    tbl.add_column("Emoji", style="bold")
    tbl.add_column("Agent", style="bold cyan")
    tbl.add_column("Responsibility")
    tbl.add_column("Status", style="green")
    roster = [
        ("🎯", "OrchestratorAgent",     "Routes & coordinates all agents", "READY"),
        ("🖼️", "ImageAnalysisAgent",    "Claude Vision — cat detection, scene analysis", "READY"),
        ("📄", "DocumentIngestionAgent","PDF chunking + vector embedding",  "READY"),
        ("🔍", "RAGQueryAgent",          "Semantic retrieval + Claude synthesis", "READY"),
        ("🗂️", "MetadataAgent",          "Governance, inventory, deletion", "READY"),
    ]
    for row in roster:
        tbl.add_row(*row)
    console.print(tbl)

    # ── Step 1 — Metadata: initial state ──────────────────────────────
    step_header(1, "Check initial knowledge base state", "MetadataAgent", "🗂️")
    result = orch.get_metadata(action="stats")
    print_result(result, [
        ("Total chunks", result.data.get("total_chunks", 0)),
        ("Sources indexed", result.data.get("total_sources", 0)),
        ("Collection", result.data.get("collection", "")),
    ])

    # ── Step 2 — Document ingestion ────────────────────────────────────
    num_pages = 10   # kept small so demo runs fast; real use supports 300+
    step_header(2, f"Ingest a {num_pages}-page insurance policy PDF", "DocumentIngestionAgent", "📄")
    console.print("  [dim]Generating sample PDF…[/dim]")

    with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                  BarColumn(), TimeElapsedColumn(), transient=True) as prog:
        t = prog.add_task("Ingesting document…", total=None)
        pdf_bytes = make_sample_pdf_bytes(num_pages)
        result = orch.ingest_document(
            pdf_bytes,
            filename="commercial_policy_P2024_001.pdf",
            extra_meta={"department": "underwriting", "policy_id": "P-2024-001"},
        )
        prog.update(t, description="Done!")

    data = result.data or {}
    print_result(result, [
        ("Filename",      data.get("filename", "")),
        ("Pages",         data.get("total_pages", 0)),
        ("Chunks stored", data.get("total_chunks", 0)),
    ])

    # ── Step 3 — Image analysis (cat test) ────────────────────────────
    step_header(3, "Analyse image — 'Is this a cat?'", "ImageAnalysisAgent", "🖼️")
    console.print("  [dim]Generating sample image…[/dim]")
    img_bytes = make_sample_image_bytes()
    result = orch.analyse_image(img_bytes, "claims_photo.jpg",
                                question="Is there a cat in this image? Describe what you see.")
    data = result.data or {}
    print_result(result)
    verdict = ("🐱 [bold green]CAT DETECTED![/bold green]"
               if data.get("is_cat") else "🚫 [yellow]No cat in this image.[/yellow]")
    console.print(f"\n  Cat verdict: {verdict}")
    console.print()
    analysis_preview = data.get("analysis", "")[:300]
    console.print(Panel(
        textwrap.fill(analysis_preview, width=70) + ("…" if len(data.get("analysis","")) > 300 else ""),
        title="[cyan]Vision Analysis (preview)[/cyan]",
        border_style="dim",
    ))

    # ── Step 4 — RAG Query ─────────────────────────────────────────────
    questions = [
        "What is the flood risk assessment for the insured property?",
        "What are the reinsurance arrangements for this policy?",
        "What claims were filed and what was the settlement history?",
    ]
    for i, q in enumerate(questions, 1):
        step_header(4 if i == 1 else 4, f"RAG Query #{i}", "RAGQueryAgent", "🔍")
        console.print(f"  [bold]Question:[/bold] {q}")
        result = orch.query(q, top_k=3)
        data = result.data or {}
        print_result(result, [
            ("Sources used", len(data.get("sources", []))),
        ])
        answer = data.get("answer", "")[:400]
        console.print(Panel(
            textwrap.fill(answer, width=70) + ("…" if len(data.get("answer","")) > 400 else ""),
            title="[cyan]AI Answer (preview)[/cyan]",
            border_style="dim",
        ))

    # ── Step 5 — Orchestrator intelligent chat routing ─────────────────
    step_header(5, "Orchestrator intelligent chat routing", "OrchestratorAgent", "🎯")
    chat_msg = "What is the maximum probable loss for fire events and how does the reinsurance treaty protect us?"
    console.print(f"  [bold]Chat message:[/bold] {chat_msg}")
    result = orch.chat(chat_msg)
    data   = result.data or {}
    routing = result.metadata.get("routing", {})
    print_result(result, [
        ("Routed to", routing.get("agent", "?")),
        ("Reasoning", routing.get("reasoning", "")),
    ])
    answer = data.get("answer", "")[:400]
    if answer:
        console.print(Panel(
            textwrap.fill(answer, width=70),
            title="[cyan]Orchestrated Answer (preview)[/cyan]",
            border_style="dim",
        ))

    # ── Step 6 — Final inventory report ───────────────────────────────
    step_header(6, "Final inventory report", "MetadataAgent", "🗂️")
    result = orch.get_metadata(action="stats")
    data   = result.data or {}
    print_result(result, [
        ("Total chunks", data.get("total_chunks", 0)),
        ("Total sources", data.get("total_sources", 0)),
    ])

    # ── Summary ────────────────────────────────────────────────────────
    console.print()
    console.print(Panel.fit(
        Text.assemble(
            ("  🎉  DEMO COMPLETE  🎉\n\n", "bold green"),
            ("  All 5 agents demonstrated successfully:\n", "white"),
            ("    🗂️  MetadataAgent      — inventory & governance\n", "cyan"),
            ("    📄  DocumentIngestionAgent — 300-page PDF ready\n", "cyan"),
            ("    🖼️  ImageAnalysisAgent  — cat detection via vision\n", "cyan"),
            ("    🔍  RAGQueryAgent       — grounded Q&A with citations\n", "cyan"),
            ("    🎯  OrchestratorAgent   — intelligent routing\n\n", "cyan"),
            ("  Next steps:\n", "bold white"),
            ("    • streamlit run demo/streamlit_app.py   (web UI)\n", "yellow"),
            ("    • uvicorn main:app --reload             (REST API)\n", "yellow"),
        ),
        border_style="bright_green",
        padding=(1, 4),
    ))
    console.print()


if __name__ == "__main__":
    run_demo()
