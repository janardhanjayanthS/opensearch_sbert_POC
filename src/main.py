import os
from time import perf_counter
from typing import Optional

from opensearch.opensearch import add_document, create_index, search
from sbert.chunking_class import SemanticChunker, get_file_contents

BASE_PATH = "files\\pdf\\English"
# BASE_PATH = "files\\pdf\\Books"


def get_file_paths() -> Optional[list[str]]:
    """
    Collect all file paths from the base PDF directory.

    Reads every file in ``BASE_PATH`` and returns their full relative paths.

    Returns:
        A list of file path strings, e.g.
        ``["files\\pdf\\English\\doc.pdf", ...]``.
    """
    pdfs = os.listdir(BASE_PATH)
    file_paths = []
    for pdf in pdfs:
        file_paths.append(BASE_PATH + "\\" + pdf)
    return file_paths


if __name__ == "__main__":
    """
    Entry point for the PDF ingestion and semantic search pipeline.

    Pipeline steps:
        1. Discover all PDFs under ``BASE_PATH``.
        2. Extract and clean text from each PDF page (via PyMuPDF).
        3. Semantically chunk the text using sentence embeddings
           (``all-MiniLM-L12-v2``) and cosine-similarity breakpoints.
        4. Embed each chunk with OpenAI ``text-embedding-3-large`` and
           store it as a KNN vector document in the OpenSearch index.
        5. Accept natural-language queries in a loop, embed the query,
           and return the top-5 most similar chunks from the index.
           Type ``e`` or ``exit`` to quit.
    """
    index = "openai_rag_index_one"

    file_paths = get_file_paths()
    # Chunker
    chunker = SemanticChunker()

    # Create a new index (table)
    create_index(index_name=index)
    for filepath in file_paths:
        file_contents = get_file_contents(filepath)
        text = "".join(list(file_contents))
        chunks = chunker.chunk(text)
        print(f"{len(chunks)} chunks from {filepath}")

        # embedding + storing
        for idx, chunk in enumerate(chunks, start=1):
            add_document(index_name=index, chunk_idx=idx, text=chunk, filepath=filepath)

    while True:
        query = input("Search: ")

        if query.lower() in {"e", "exit"}:
            break

        # searching
        start = perf_counter()
        search(index_name=index, user_query=query)
        end = perf_counter()
        print("-")
        print(f"Response time: {round(end - start, 4)} s")
        print("-")

    # delete_index(index_name=index)
