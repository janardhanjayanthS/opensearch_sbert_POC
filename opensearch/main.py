import os

from dotenv import load_dotenv
from openai import OpenAI
from opensearchpy import OpenSearch

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

os_client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_auth=("admin", "StrongPassword123!"),
    use_ssl=True,
    verify_certs=False,
    ssl_show_warn=False,
)

index_name = "my-openai-rag-index"


def create_index() -> None:
    """
    Create the OpenSearch index with KNN vector search enabled if it doesn't already exist.

    The index stores two fields per document:
    - ``text_chunk``: the raw text
    - ``embedding``: a 3072-dim KNN vector (HNSW via Faiss, cosine similarity),
      matching the output dimension of ``text-embedding-3-large``
    """
    index_body = {
        "settings": {
            "index.knn": True
        },
        "mappings": {
            "properties": {
                "text_chunk": {"type": "text"},
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


def add_document(doc_id: int, text: str) -> None:
    """
    Embed ``text`` using OpenAI's ``text-embedding-3-large`` and store it in OpenSearch.

    The document is indexed immediately (``refresh=True``) so it is
    searchable without waiting for the next index refresh cycle.

    Args:
        doc_id: Unique identifier for the document in the index.
        text: Raw text to embed and store.
    """
    response = openai_client.embeddings.create(
        model="text-embedding-3-large", input=text
    )
    vector = response.data[0].embedding

    document = {"text_chunk": text, "embedding": vector}

    os_client.index(index=index_name, body=document, id=doc_id, refresh=True)
    print(f"Added document {doc_id} to OpenSearch.")


def search(user_query: str) -> None:
    """
    Perform a KNN vector search against the index using ``user_query`` as input.

    The query is embedded with the same model used during indexing, then the
    top-5 nearest neighbours are retrieved by cosine similarity. Results are
    printed with their similarity score and source text.

    Args:
        user_query: Natural language query to search for semantically similar documents.
    """
    query_response = openai_client.embeddings.create(
        model="text-embedding-3-large", input=user_query
    )
    query_vector = query_response.data[0].embedding

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
        print(f"Score (Similarity): {score:.4f} | Text: {text}")


if __name__ == "__main__":
    create_index()
    add_document(doc_id=1, text="The Eiffel Tower is located in Paris, France.")
    add_document(doc_id=2, text="Python is a popular programming language for AI.")
    add_document(
        doc_id=3,
        text="OpenSearch is a distributed search engine fork of Elasticsearch.",
    )
    search("Where can I find a famous French landmark?")
