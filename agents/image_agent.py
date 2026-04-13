"""
ImageAnalysisAgent — uses Claude Vision to analyse uploaded images.

Capabilities:
  • Object / animal detection  (e.g. "Is there a cat?")
  • General scene description
  • Text extraction from images
  • Stores analysis summaries in the vector store for later retrieval
"""

import base64
import mimetypes
import uuid
from pathlib import Path
from typing import Any

from .base_agent import BaseAgent, AgentResult
from vector_store import ChromaVectorStore


_SYSTEM = """You are an expert visual analyst for an enterprise AI platform.
When given an image:
1. Describe the scene comprehensively.
2. List every identifiable object, animal, person, text or chart present.
3. If the user asks a specific question, answer it directly and confidently.
4. Keep your response structured: first a one-sentence summary, then details.
5. Be precise about any animals — if you see a cat, say so clearly."""


class ImageAnalysisAgent(BaseAgent):
    name = "ImageAnalysisAgent"
    description = "Analyses images with Claude Vision; detects objects, animals, text, and scenes"
    emoji = "🖼️"

    def __init__(self, vector_store: ChromaVectorStore):
        super().__init__()
        self._store = vector_store

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def process(self, *, image_bytes: bytes, filename: str = "image.jpg",
                question: str | None = None, store_result: bool = True,
                **_) -> AgentResult:
        """
        Analyse *image_bytes*.

        Args:
            image_bytes:  Raw bytes of the image (JPEG, PNG, GIF, WEBP).
            filename:     Original filename (used for MIME detection).
            question:     Optional specific question (e.g. "Is this a cat?").
            store_result: Persist the analysis in the vector store.
        """
        self._log(f"Received image '{filename}' ({len(image_bytes):,} bytes)")

        mime = self._detect_mime(filename)
        b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

        user_content: list[dict[str, Any]] = [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": mime, "data": b64},
            },
            {
                "type": "text",
                "text": question if question else
                        "Please describe this image in detail, listing all objects, animals, "
                        "people, text, or notable features you can identify.",
            },
        ]

        self._log("Sending image to Claude Vision…")
        analysis = self._call_claude(_SYSTEM, user_content, max_tokens=1024)
        self._log("Vision analysis complete")

        # ------ quick is-it-a-cat check --------------------------------
        is_cat = any(kw in analysis.lower() for kw in
                     ("cat", "kitten", "feline", "tabby", "kitty", "meow"))

        result_data = {
            "filename": filename,
            "analysis": analysis,
            "is_cat": is_cat,
            "question": question,
            "mime_type": mime,
            "image_size_bytes": len(image_bytes),
        }

        # ------ persist to vector store --------------------------------
        doc_id = None
        if store_result:
            doc_id = str(uuid.uuid4())
            self._store.add_documents(
                texts=[f"[IMAGE ANALYSIS] {filename}\n\n{analysis}"],
                metadatas=[{
                    "source": filename,
                    "doc_type": "image",
                    "is_cat": str(is_cat),
                    "doc_id": doc_id,
                }],
                ids=[doc_id],
            )
            self._log(f"Analysis stored in vector store (id={doc_id})")
            result_data["doc_id"] = doc_id

        verdict = "YES — a cat is present! 🐱" if is_cat else "No cat detected."
        return AgentResult(
            agent=self.name,
            success=True,
            data=result_data,
            message=f"Image analysed. Cat verdict: {verdict}",
            metadata={"doc_id": doc_id, "is_cat": is_cat},
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_mime(filename: str) -> str:
        mime, _ = mimetypes.guess_type(filename)
        allowed = {"image/jpeg", "image/png", "image/gif", "image/webp"}
        return mime if mime in allowed else "image/jpeg"
