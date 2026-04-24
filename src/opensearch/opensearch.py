import hashlib
from os import getenv

from dotenv import load_dotenv
from opensearchpy import OpenSearch

from src.embed.embedder import get_vectors
from src.opensearch.index import indexes

load_dotenv()

OPENSEARCH_ADMIN_PASSWORD = getenv("OPENSEARCH_ADMIN_PASSWORD")

os_client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_auth=("admin", OPENSEARCH_ADMIN_PASSWORD),
    use_ssl=True,
    verify_certs=False,
    ssl_show_warn=False,
)


def create_index() -> None:
    """
    Create the OpenSearch index with KNN vector search enabled if it doesn't already exist.

    The index stores two fields per document:
    - ``text_chunk``: the raw text
    - ``embedding``: a 3072-dim KNN vector (HNSW via Faiss, cosine similarity),
      matching the output dimension of ``text-embedding-3-large``
    """

    for index_name, index in indexes.items():
        if not os_client.indices.exists(index=index_name):
            os_client.indices.create(index=index_name, body=index)
            print(f"Created index: {index_name}")
        else:
            print(f"Index {index_name} already exists.")


def add_document(text: str, category_id: str) -> None:
    """
    Embed ``text`` using OpenAI's ``text-embedding-3-large``
    and store it in OpenSearch.

    The document is indexed immediately (``refresh=True``) so it is
    searchable without waiting for the next index refresh cycle.

    Args:
        chunk_idx: pos of chunk from all chunk_list starting from 1.
        text: Raw text to embed and store.
        category_id: FK reference to a doc in category_index.
    """
    if not category_id:
        raise ValueError("category_id must not be empty")

    # get category for this text
    vector = get_vectors(text=text)

    document = {
        "text_chunk": text,
        "embedding": vector,
        "category_id": category_id,
    }

    doc_id = hashlib.sha1(text.encode()).hexdigest()
    os_client.index(index="embedding_index", body=document, id=doc_id, refresh=True)
    print(f"Added chunk {doc_id} to OpenSearch.")


def add_category(category_name: str) -> None:
    vector = get_vectors(text=category_name)

    content = {"category_name": category_name, "embedding": vector}

    category_id = hashlib.sha1(category_name.encode()).hexdigest()
    os_client.index(body=content, id=category_id, index="category_index", refresh=True)


def search_category(category: str, size: int = 5, k: int = 5) -> dict:
    query_vector = get_vectors(text=category.lower())
    search_query = {
        "size": size,
        "_source": ["category_id", "category_name"],
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_vector,
                    "k": k,
                }
            }
        },
    }
    results = os_client.search(index="category_index", body=search_query)

    category_to_id = {}
    for hit in results["hits"]["hits"]:
        category_id = hit["_source"]["category_id"]
        category_name = hit["_source"]["category_name"]
        score = hit["_score"]
        category_to_id[category_name] = category_id

        print(f"SCORE: {score} | id: {category_id} | category: {category_name}")

    return category_to_id


def search_documents(
    user_query: str,
    size: int = 5,
    k: int = 5,
) -> None:
    """
    Perform a KNN vector search against the index.

    Query is embedded with the same model used during ingest, then the top-``size``
    nearest neighbours are retrieved from the HNSW graph by cosine similarity.

    Args:
        user_query: Natural language query.
        size: Number of hits to return.
        k: KNN neighbours to retrieve from HNSW per shard.
    """
    query_text = user_query.lower()
    query_vector = get_vectors(text=query_text)

    search_query = {
        "size": size,
        "_source": ["text_chunk", "file_path"],
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_vector,
                    "k": k,
                }
            }
        },
    }

    results = os_client.search(index="embedding_index", body=search_query)

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
