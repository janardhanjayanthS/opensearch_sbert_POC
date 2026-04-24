import os
from pathlib import Path
from typing import Optional

from src.categorizer.categorize import (
    check_similar_existing_category_else_return_new,
    get_category,
)
from src.opensearch.opensearch import (
    add_category,
    add_document,
    search_similar_category,
)

TXT_DIR_PATH = str(Path(__file__).parent.parent) + "\\files\\text\\"


def get_text_file_contents() -> Optional[list[str]]:
    """Read all .txt files from TXT_DIR_PATH and return their non-empty lines."""
    text_files = os.listdir(TXT_DIR_PATH)
    print(f"text file: {text_files}")
    for text_file in text_files:
        with open(TXT_DIR_PATH + text_file, "r") as file:
            contents = file.readlines()

    return [content.strip("\n") for content in contents if content != "\n"]


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
    embedding_index = "embedding_index"

    # Create a new index (table)
    # create_index()

    contents = get_text_file_contents()

    for content in contents:
        category = get_category(text=content)
        existing_categories: dict = search_similar_category(category=category)

        comparison_result = check_similar_existing_category_else_return_new(
            new_category=category, existing_categories=list(existing_categories.keys())
        )

        if comparison_result == category:
            category_id = add_category(category_name=category)
        else:
            category_id = existing_categories[comparison_result]

        add_document(text=content, category_id=category_id)

    # while True:
    #     query = input("Search: ")

    #     if query.lower() in {"e", "exit"}:
    #         break

    #     # searching
    #     start = perf_counter()
    #     search(index_name=embedding_index, user_query=query)
    #     end = perf_counter()
    #     print("-")
    #     print(f"Response time: {round(end - start, 4)} s")
    #     print("-")

    # delete_index(index_name=index)
