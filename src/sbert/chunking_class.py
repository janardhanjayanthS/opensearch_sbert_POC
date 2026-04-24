import re
from typing import List, Optional

import numpy as np
import pymupdf
from sentence_transformers import SentenceTransformer


class SemanticChunker:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L12-v2",  # Your specified model
        similarity_threshold: float = 0.5,
        min_chunk_size: int = 200,
        max_chunk_size: int = 5000,
    ):
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def split_into_sentences(self, text: str) -> List[str]:
        pattern = r"(?<!Mr)(?<!Mrs)(?<!Ms)(?<!Dr)(?<!Prof)(?<!Sr)(?<!Jr)(?<!vs)(?<!etc)(?<!e\.g)(?<!i\.e)\.\s+(?=[A-Z])"
        sentences = re.split(pattern, text)
        result = []
        for sentence in sentences:
            sub_sentences = re.split(r"[?!]\s+(?=[A-Z])", sentence)
            result.extend(sub_sentences)
        return [s.strip() for s in result if s.strip()]

    def compute_embeddings(self, sentences: List[str]) -> np.ndarray:
        return self.model.encode(sentences, convert_to_numpy=True)

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        return dot_product / norm_product if norm_product != 0 else 0.0

    def find_breakpoints(
        self, sentences: List[str], embeddings: np.ndarray
    ) -> List[int]:
        breakpoints = []
        for i in range(len(sentences) - 1):
            similarity = self.cosine_similarity(embeddings[i], embeddings[i + 1])
            if similarity < self.similarity_threshold:
                breakpoints.append(i + 1)
        return breakpoints

    def merge_small_chunks(self, chunks: List[str]) -> List[str]:
        if not chunks:
            return chunks
        merged = []
        buffer = chunks[0]
        for chunk in chunks[1:]:
            if len(buffer) < self.min_chunk_size:
                combined = buffer + " " + chunk
                if len(combined) <= self.max_chunk_size:
                    buffer = combined
                    continue
            merged.append(buffer)
            buffer = chunk
        merged.append(buffer)
        return merged

    def chunk(self, text: str) -> List[str]:
        sentences = self.split_into_sentences(text)
        if len(sentences) <= 1:
            return [text] if text.strip() else []
        embeddings = self.compute_embeddings(sentences)
        breakpoints = self.find_breakpoints(sentences, embeddings)
        chunks = []
        start_idx = 0
        for bp in breakpoints:
            chunk_text = " ".join(sentences[start_idx:bp])
            chunks.append(chunk_text)
            start_idx = bp
        if start_idx < len(sentences):
            chunks.append(" ".join(sentences[start_idx:]))
        return self.merge_small_chunks(chunks)


def clean_text(text: str) -> str:
    cleaned_text = re.sub(r"\n", " ", text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    return cleaned_text


def get_file_contents(filepath: str) -> Optional[list[str]]:
    try:
        file = pymupdf.open(filename=filepath)
        contents = []
        for page in file:
            text = page.get_text()
            contents.append(clean_text(text=text))
        return contents
    except Exception as e:
        print(f"ERROR: {e}")


# Usage
# if __name__ == "__main__":
#     chunker = SemanticChunker()
#     # file_path = "..\\files\\pdf\\English\\Canada Vacation-Flexible Time off (FTO) Policy 2025.pdf"
#     file_path = "..\\files\\pdf\\English\\sbert_test_2col.pdf"
    # file_contents = get_file_contents(file_path)
    # text = "".join(list(file_contents))
#     chunks = chunker.chunk(text)
#     for chunk in chunks:
#         print(chunk, len(chunk))
#         print("-" * 50)
#     print(len(chunks))
