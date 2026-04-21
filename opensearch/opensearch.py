from os import getenv

from dotenv import load_dotenv
from opensearchpy import OpenSearch

from embed.embedder import get_vectors

load_dotenv()

OPENSEARCH_ADMIN_PASSWORD = getenv("OPENSEARCH_ADMIN_PASSWORD")

os_client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_auth=("admin", OPENSEARCH_ADMIN_PASSWORD),
    use_ssl=True,
    verify_certs=False,
    ssl_show_warn=False,
)


def create_index(index_name: str = "my-openai-rag-index") -> None:
    """
    Create the OpenSearch index with KNN vector search enabled if it doesn't already exist.

    The index stores two fields per document:
    - ``text_chunk``: the raw text
    - ``embedding``: a 3072-dim KNN vector (HNSW via Faiss, cosine similarity),
      matching the output dimension of ``text-embedding-3-large``
    """
    index_body = {
        "settings": {"index.knn": True},
        "mappings": {
            "properties": {
                "text_chunk": {"type": "text"},
                "file_path": {"type": "text"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 3072,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "faiss",
                    },
                },
            }
        },
    }

    if not os_client.indices.exists(index=index_name):
        os_client.indices.create(index=index_name, body=index_body)
        print(f"Created index: {index_name}")
    else:
        print(f"Index {index_name} already exists.")


def add_document(index_name: str, doc_id: int, text: str, filepath: str) -> None:
    """
    Embed ``text`` using OpenAI's ``text-embedding-3-large`` and store it in OpenSearch.

    The document is indexed immediately (``refresh=True``) so it is
    searchable without waiting for the next index refresh cycle.

    Args:
        doc_id: Unique identifier for the document in the index.
        text: Raw text to embed and store.
    """
    vector = get_vectors(text=text)

    document = {"text_chunk": text, "embedding": vector, "file_path": filepath}

    os_client.index(index=index_name, body=document, id=doc_id, refresh=True)
    print(f"Added document {doc_id} to OpenSearch.")


def search(index_name: str, user_query: str) -> None:
    """
    Perform a KNN vector search against the index using ``user_query`` as input.

    The query is embedded with the same model used during indexing, then the
    top-5 nearest neighbours are retrieved by cosine similarity. Results are
    printed with their similarity score and source text.

    Args:
        user_query: Natural language query to search for semantically similar documents.
    """
    query_vector = get_vectors(text=user_query)

    search_query = {
        "size": 5,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_vector,
                    "k": 5,
                }
            }
        },
    }

    results = os_client.search(index=index_name, body=search_query)

    print("\n--- Search Results ---")
    for hit in results["hits"]["hits"]:
        score = hit["_score"]
        text = hit["_source"]["text_chunk"]
        file = hit["_source"]["file_path"]
        print(f"Score (Similarity): {score:.4f} | Text: {text}")
        print(f"File path: {file}")


def delete_index(index_name: str) -> None:
    os_client.indices.delete(index=index_name)


# FOR TESTING OPENSEARCH
# if __name__ == "__main__":
#     index = "sample_idx"
#     create_index(index_name=index)
#     add_document(
#         doc_id=1,
#         text="The Eiffel Tower is located in Paris, France.",
#         filepath="random file path 1",
#         index_name=index,
#     )
#     add_document(
#         doc_id=2,
#         text="Python is a popular programming language for AI.",
#         filepath="random file path 2",
#         index_name=index,
#     )
#     add_document(
#         doc_id=3,
#         text="OpenSearch is a distributed search engine fork of Elasticsearch.",
#         filepath="random file path 3",
#         index_name=index,
#     )
#     search(user_query="Where can I find a famous French landmark?", index_name=index)
