import re
from typing import Optional

import pymupdf
from sentence_transformers import SentenceTransformer, util


def split_sentences(text):
    # Simple sentence splitter; good enough for many cases
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def semantic_chunk_text(
    text,
    model_name="all-MiniLM-L12-v2",
    similarity_threshold=0.65,
    max_sentences_per_chunk=5,
):
    """
    Groups adjacent sentences into semantic chunks using SBERT embeddings.
    Sentences are merged if similarity with the previous sentence is high enough.
    """

    model = SentenceTransformer(model_name)
    sentences = split_sentences(text)

    if not sentences:
        return []

    embeddings = model.encode(sentences, convert_to_tensor=True)

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        prev_embedding = embeddings[i - 1]
        curr_embedding = embeddings[i]

        similarity = util.cos_sim(prev_embedding, curr_embedding).item()

        # Merge if semantically similar and chunk is not too long
        if (
            similarity >= similarity_threshold
            and len(current_chunk) < max_sentences_per_chunk
        ):
            current_chunk.append(sentences[i])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]

    # Add final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def get_raw_page_texts(filepath: str) -> Optional[list[str]]:
    """Extract raw text per page, preserving \\n for structure detection."""
    try:
        doc = pymupdf.open(filename=filepath)
        return [page.get_text() for page in doc]
    except Exception as e:
        print(f"ERROR: {e}")
        return None


if __name__ == "__main__":
    file_path = "..\\files\\pdf\\English\\Canada Vacation-Flexible Time Off (FTO) Policy 2025.pdf"
    pages = get_raw_page_texts(file_path)

    if not pages:
        print("Failed to read file.")
    else:
        text = "\n".join(pages)
        chunks = semantic_chunk_text(text, similarity_threshold=0.2)

        for idx, chunk in enumerate(chunks, 1):
            print(f"\nChunk {idx}:\n{chunk}")
        print(f"\nTotal chunks: {len(chunks)}")
