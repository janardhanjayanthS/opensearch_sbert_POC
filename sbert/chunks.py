import os
import pathlib
import re
from typing import Optional

import nltk
import numpy as np
import pymupdf
from sentence_transformers import SentenceTransformer, util


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


def get_sentence_tokens(file_contents: list[str]):
    tokenized_content = []
    for content in file_contents:
        tokenized_content.extend(nltk.sent_tokenize(content))

    return tokenized_content


def semantic_chunker(text_list, window_size=3, threshold_percentile=90):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 1. Generate individual sentence embeddings
    embeddings = model.encode(text_list)

    # 2. The "Buffer" Method: Create combined signals
    # We create a "smoothed" embedding for each position by averaging the window
    combined_embeddings = []
    for i in range(len(embeddings)):
        start = max(0, i - window_size)
        end = min(len(embeddings), i + window_size + 1)
        combined_embeddings.append(np.mean(embeddings[start:end], axis=0))

    print(f"Combined Embeddings: {combined_embeddings}")

    # 3. Calculate Cosine Distances (1 - Similarity)
    # Distance is easier to visualize for "peaks" where topics change
    distances = []
    for i in range(len(combined_embeddings) - 1):
        similarity = util.cos_sim(combined_embeddings[i], combined_embeddings[i + 1])
        distances.append(1 - similarity.item())

    print(f"Distances: {distances}")
    # 4. Percentile Thresholding: Find the "outliers" (the biggest shifts)
    breakpoint_threshold = np.percentile(distances, threshold_percentile)

    print(f"BP Threshold: {breakpoint_threshold}")
    # 5. Build the Chunks
    chunks = []
    current_chunk = [text_list[0]]

    for i, distance in enumerate(distances):
        if distance > breakpoint_threshold:
            # Significant shift detected
            chunks.append(" ".join(current_chunk))
            current_chunk = [text_list[i + 1]]
        else:
            current_chunk.append(text_list[i + 1])

    chunks.append(" ".join(current_chunk))
    return chunks


if __name__ == "__main__":
    current_file_path = pathlib.Path(__file__)
    pdf_dir = str(current_file_path.parent.parent) + "\\files\\pdf\\"
    pdf_files = os.listdir(pdf_dir)
    file_contents = get_file_contents(filepath=f"{pdf_dir}\\{pdf_files[0]}")
    sentence_tokens = get_sentence_tokens(file_contents=file_contents)
    my_chunks = semantic_chunker(
        sentence_tokens, window_size=2, threshold_percentile=95
    )
    print(f"CHUNKS: {my_chunks}")
    print(f"CHUNKS length: {len(my_chunks)}")

    for i, chunk in enumerate(my_chunks):
        print(f"{i} - {chunk}")
        print("-" * 30)
