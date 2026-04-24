"""
Fixed-Size + Semantic Overlap (Hybrid) Chunker

Steps:
  1. Split PDF text into sentences.
  2. Group sentences into fixed-size windows by word count (max_words).
  3. Use SBERT to score adjacent fixed windows — merge pairs that are
     semantically similar (similarity >= merge_threshold).
  4. Add a sentence overlap between final chunks so context is never
     cut mid-thought when passed to an LLM.
"""

import argparse
import os
import pathlib
import re
from typing import Optional

import nltk
import pymupdf
from sentence_transformers import SentenceTransformer, util


def clean_text(text: str) -> str:
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_file_contents(filepath: str) -> Optional[list[str]]:
    try:
        doc = pymupdf.open(filename=filepath)
        return [clean_text(page.get_text()) for page in doc]
    except Exception as e:
        print(f"ERROR: {e}")


def get_sentence_tokens(file_contents: list[str]) -> list[str]:
    sentences = []
    for content in file_contents:
        sentences.extend(nltk.sent_tokenize(content))
    return sentences


def fixed_size_windows(sentences: list[str], max_words: int) -> list[list[str]]:
    """Group sentences into windows where total word count <= max_words."""
    windows: list[list[str]] = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        word_count = len(sentence.split())
        if current and current_words + word_count > max_words:
            windows.append(current)
            current = []
            current_words = 0
        current.append(sentence)
        current_words += word_count

    if current:
        windows.append(current)

    return windows


def merge_similar_windows(
    windows: list[list[str]],
    model: SentenceTransformer,
    merge_threshold: float,
) -> list[list[str]]:
    """
    Merge adjacent windows whose SBERT cosine similarity >= merge_threshold.
    Iterates until no more merges are possible.
    """
    window_texts = [" ".join(w) for w in windows]
    merged = True

    while merged and len(windows) > 1:
        merged = False
        embeddings = model.encode(window_texts)
        new_windows: list[list[str]] = []
        new_texts: list[str] = []
        i = 0

        while i < len(windows):
            if i < len(windows) - 1:
                similarity = util.cos_sim(embeddings[i], embeddings[i + 1]).item()
                if similarity >= merge_threshold:
                    # Merge this window with the next
                    combined = windows[i] + windows[i + 1]
                    new_windows.append(combined)
                    new_texts.append(" ".join(combined))
                    i += 2
                    merged = True
                    continue
            new_windows.append(windows[i])
            new_texts.append(window_texts[i])
            i += 1

        windows = new_windows
        window_texts = new_texts

    return windows


def add_sentence_overlap(
    windows: list[list[str]], overlap_sentences: int
) -> list[str]:
    """
    Build final chunk strings. Each chunk gets `overlap_sentences` sentences
    from the end of the previous chunk prepended, so context carries over.
    """
    chunks: list[str] = []
    for i, window in enumerate(windows):
        if i == 0 or overlap_sentences == 0:
            chunks.append(" ".join(window))
        else:
            tail = windows[i - 1][-overlap_sentences:]
            chunks.append(" ".join(tail + window))
    return chunks


def hybrid_chunker(
    sentences: list[str],
    max_words: int = 200,
    merge_threshold: float = 0.75,
    overlap_sentences: int = 1,
    model_name: str = "all-MiniLM-L6-v2",
) -> list[str]:
    model = SentenceTransformer(model_name)

    # Step 1: fixed-size windows
    windows = fixed_size_windows(sentences, max_words=max_words)

    # Step 2: merge semantically similar adjacent windows
    windows = merge_similar_windows(windows, model=model, merge_threshold=merge_threshold)

    # Step 3: add sentence overlap for LLM context continuity
    chunks = add_sentence_overlap(windows, overlap_sentences=overlap_sentences)

    return chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fixed-size + semantic overlap hybrid chunker"
    )
    parser.add_argument("pdf", nargs="?", help="Path to the PDF file")
    parser.add_argument(
        "--max-words",
        type=int,
        default=200,
        help="Max words per initial fixed-size window (default: 200). "
             "Lower = more, smaller initial windows.",
    )
    parser.add_argument(
        "--merge-threshold",
        type=float,
        default=0.75,
        help="SBERT cosine similarity threshold for merging adjacent windows "
             "(default: 0.75). Higher = fewer merges = more chunks.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=1,
        help="Number of sentences to carry over from the previous chunk "
             "as overlap (default: 1).",
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name (default: all-MiniLM-L6-v2)",
    )
    args = parser.parse_args()

    if args.pdf:
        pdf_path = pathlib.Path(args.pdf).resolve()
    else:
        pdf_dir = pathlib.Path(__file__).parent.parent / "files" / "pdf"
        pdf_files = os.listdir(pdf_dir)
        pdf_path = pdf_dir / pdf_files[0]

    file_contents = get_file_contents(str(pdf_path))
    sentences = get_sentence_tokens(file_contents)
    chunks = hybrid_chunker(
        sentences,
        max_words=args.max_words,
        merge_threshold=args.merge_threshold,
        overlap_sentences=args.overlap,
        model_name=args.model,
    )

    print(f"Total chunks: {len(chunks)}\n")
    for i, chunk in enumerate(chunks):
        print(f"[{i}] {chunk}")
        print("-" * 60)
