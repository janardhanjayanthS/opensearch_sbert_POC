from opensearch.opensearch import add_document, create_index, search
from sbert.chunking_class import SemanticChunker, get_file_contents

if __name__ == "__main__":
    # Chunking
    chunker = SemanticChunker()
    filepath = (
        "files\\pdf\\English\\Canada Vacation-Flexible Time off (FTO) Policy 2025.pdf"
    )
    file_contents = get_file_contents(filepath)
    text = "".join(list(file_contents))
    chunks = chunker.chunk(text)

    # embedding + storing
    create_index()
    for id, chunk in enumerate(chunks, start=1):
        add_document(doc_id=id, text=chunk)

    query = input("Search: ")

    # searching
    search(user_query=query)
