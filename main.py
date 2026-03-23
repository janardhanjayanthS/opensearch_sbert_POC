import os
from typing import Optional

from opensearch.opensearch import add_document, create_index, search
from sbert.chunking_class import SemanticChunker, get_file_contents

BASE_PATH = "files\\pdf\\English"


def get_file_paths() -> Optional[list[str]]:
    pdfs = os.listdir(BASE_PATH)
    file_paths = []
    for pdf in pdfs:
        file_paths.append(BASE_PATH + "\\" + pdf)
    return file_paths


if __name__ == "__main__":
    count = 1
    file_paths = get_file_paths()
    # Chunker
    chunker = SemanticChunker()

    # Create a new index (table)
    index = "openai_rag_index_one"
    create_index(index_name=index)
    for filepath in file_paths:
        file_contents = get_file_contents(filepath)
        text = "".join(list(file_contents))
        chunks = chunker.chunk(text)

        # embedding + storing
        for chunk in chunks:
            add_document(index_name=index, doc_id=count, text=chunk, filepath=filepath)
            count += 1

    while True:
        query = input("Search: ")

        if query.lower() in {"e", "exit"}:
            break

        # searching
        search(index_name=index, user_query=query)

    # delete index (table)
    # delete_index(index_name="openai_rag_index_one")
