import re
from typing import List, Optional, Tuple

import numpy as np
import pymupdf
from sentence_transformers import SentenceTransformer

# Lines that look like headings but are document metadata — skip them
_IGNORE_PATTERNS = [
    r"^\d+/\d+/\d+$",  # dates: 01/01/2019
    r"^Page \d+ of \d+$",  # page numbers
    r"^Issue \d+",  # issue numbers
    r"^Internal Use$",  # classification label
    r"^DATE OF ISSUE",  # metadata
    r"^FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER|JANUARY",
    r"^\s*$",  # blank lines
]

# Heading patterns ordered from most to least specific
_HEADING_PATTERNS = [
    r"^\d+\.\s+[A-Z][A-Za-z\s/&()\-,\d]+:?$",  # 1. Statutory Vacation Time:
    r"^[A-Z][A-Za-z\s/&()\-,\d]+:$",  # Title case with colon: Eligibility:
    r"^[A-Z][A-Z\s/&\-]{3,}$",  # ALL CAPS: OVERVIEW, PROCEDURE
    r"^[A-Z][A-Za-z\s/&()\-,\d]{3,69}$",  # Short title case (max 70 chars): Approval process
]


def _is_heading(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    # Skip metadata lines
    for pattern in _IGNORE_PATTERNS:
        if re.match(pattern, line):
            return False
    # Headings never end with a period — body sentences do
    if line.endswith("."):
        return False
    for pattern in _HEADING_PATTERNS:
        if re.match(pattern, line):
            return True
    return False


class SectionAwareChunker:
    """
    Two-stage chunker for structured policy PDFs:
      Stage 1 — detect section headings from raw text, group all sentences
                under the same heading into one chunk
      Stage 2 — if a section body exceeds max_chunk_size, use SBERT cosine
                similarity to further split it at semantic breakpoints
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L12-v2",
        similarity_threshold: float = 0.5,
        max_chunk_size: int = 2000,
    ):
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size

    # ------------------------------------------------------------------
    # Stage 1 — structural section detection
    # ------------------------------------------------------------------

    def detect_sections(self, raw_text: str) -> List[Tuple[str, str]]:
        """
        Split raw text (with \\n preserved) into (heading, body) pairs.
        Text before the first heading is stored with an empty heading string.
        """
        sections: List[Tuple[str, str]] = []
        current_heading = ""
        current_body: List[str] = []

        for line in raw_text.split("\n"):
            stripped = line.strip()
            if _is_heading(stripped):
                if current_body:
                    body = self._clean_body(" ".join(current_body))
                    if body:
                        sections.append((current_heading, body))
                current_heading = stripped
                current_body = []
            else:
                if stripped:
                    current_body.append(stripped)

        # flush last section
        if current_body:
            body = self._clean_body(" ".join(current_body))
            if body:
                sections.append((current_heading, body))

        return sections

    # ------------------------------------------------------------------
    # Stage 2 — SBERT semantic split for oversized sections
    # ------------------------------------------------------------------

    def _clean_body(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _split_sentences(self, text: str) -> List[str]:
        # Split on sentence-ending punctuation followed by a capital letter
        pattern = (
            r"(?<!Mr)(?<!Mrs)(?<!Ms)(?<!Dr)(?<!Prof)(?<!Sr)(?<!Jr)"
            r"(?<!vs)(?<!etc)(?<!e\.g)(?<!i\.e)"
            r"\.\s+(?=[A-Z])"
        )
        sentences = re.split(pattern, text)
        result = []
        for s in sentences:
            result.extend(re.split(r"[?!]\s+(?=[A-Z])", s))
        return [s.strip() for s in result if s.strip()]

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        return float(np.dot(v1, v2) / denom) if denom != 0 else 0.0

    def _semantic_split(self, text: str) -> List[str]:
        """Split a large body of text at semantic breakpoints using SBERT."""
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return [text]

        embeddings = self.model.encode(sentences, convert_to_numpy=True)
        breakpoints = [
            i + 1
            for i in range(len(sentences) - 1)
            if self._cosine_similarity(embeddings[i], embeddings[i + 1])
            < self.similarity_threshold
        ]

        chunks, start = [], 0
        for bp in breakpoints:
            chunks.append(" ".join(sentences[start:bp]))
            start = bp
        chunks.append(" ".join(sentences[start:]))
        return [c for c in chunks if c.strip()]

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def chunk(self, raw_text: str) -> List[str]:
        """
        Returns a list of chunks where each chunk contains all sentences
        from a single section. Oversized sections are further split by SBERT.
        """
        sections = self.detect_sections(raw_text)
        chunks = []

        for heading, body in sections:
            full = f"{heading}\n{body}".strip() if heading else body

            if len(full) <= self.max_chunk_size:
                chunks.append(full)
            else:
                # Section too large — use SBERT to find semantic breakpoints
                sub_chunks = self._semantic_split(body)
                for sub in sub_chunks:
                    label = f"{heading}\n{sub}".strip() if heading else sub
                    chunks.append(label)

        return chunks


def get_raw_page_texts(filepath: str) -> Optional[List[str]]:
    """Extract raw text per page, preserving \\n for structure detection."""
    try:
        doc = pymupdf.open(filename=filepath)
        return [page.get_text() for page in doc]
    except Exception as e:
        print(f"ERROR: {e}")
        return None


if __name__ == "__main__":
    chunker = SectionAwareChunker(
        # model_name="PM-AI/bi-encoder_msmarco_bert-base_german"
        max_chunk_size=500
    )

    files = [
        # "..\\files\\pdf\\English\\PTO Policy Chatsworth.pdf",
        "..\\files\\pdf\\English\\Canada Vacation-Flexible Time Off (FTO) Policy 2025.pdf",
        "..\\files\\pdf\\English\\sbert_test_2col.pdf",
        # "..\\files\\pdf\\English\\Extended Leave and Career Breaks Policy United Kingdom .pdf",
        # "..\\files\\pdf\\English\\ResMed SaaS Paid Time-Off Policy.pdf",
        # German
        # "..\\files\\pdf\\German\\archivierung_und_wiedervorlage.pdf",
    ]

    for file_path in files:
        pages = get_raw_page_texts(file_path)
        if not pages:
            print("Failed to read file")
            continue

        # Join pages with \n to preserve cross-page structure
        raw_text = "\n".join(pages)
        chunks = chunker.chunk(raw_text)

        for i, chunk in enumerate(chunks):
            print(f"\n[Chunk {i + 1}] ({len(chunk)} chars)")
            print(chunk)
            print("-" * 50)

        print(f"\nTotal chunks: {len(chunks)}")
