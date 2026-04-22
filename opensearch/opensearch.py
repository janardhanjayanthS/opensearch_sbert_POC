import hashlib
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


def add_document(index_name: str, chunk_idx: int, text: str, filepath: str) -> None:
    """
    Embed ``text`` using OpenAI's ``text-embedding-3-large``
    and store it in OpenSearch.

    The document is indexed immediately (``refresh=True``) so it is
    searchable without waiting for the next index refresh cycle.

    Args:
        chunk_idx: pos of chunk from all chunk_list starting from 1.
        text: Raw text to embed and store.
    """
    vector = get_vectors(text=text)

    document = {"text_chunk": text, "embedding": vector, "file_path": filepath}

    doc_id = hashlib.sha1(f"{filepath}:{chunk_idx}".encode()).hexdigest()
    os_client.index(index=index_name, body=document, id=doc_id, refresh=True)
    print(f"Added chunk {doc_id} to OpenSearch.")


def search(
    index_name: str,
    user_query: str,
    size: int = 5,
    k: int = 5,
    bm25_weight: float = 1.0,
    vector_weight: float = 1.5,
) -> None:
    """
    Perform a hybrid BM25 + KNN vector search against the index.

    BM25 matches ``text_chunk`` lexically; KNN matches ``embedding`` semantically.
    Scores are combined via a ``bool.should`` with per-clause ``boost`` weights.

    Args:
        user_query: Natural language query.
        size: Number of hits to return.
        k: KNN neighbours to retrieve before scoring.
        bm25_weight: Boost applied to BM25 clause.
        vector_weight: Boost applied to KNN clause.
    """
    query_text = user_query.lower()
    query_vector = get_vectors(text=query_text)

    search_query = {
        "size": size,
        "_source": ["text_chunk", "file_path"],
        "query": {
            "bool": {
                "should": [
                    {
                        "multi_match": {
                            "query": query_text,
                            "fields": ["text_chunk^2"],
                            "boost": bm25_weight,
                        }
                    },
                    {
                        "knn": {
                            "embedding": {
                                "vector": query_vector,
                                "k": k,
                                "boost": vector_weight,
                            }
                        }
                    },
                ]
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
        print("-")


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
