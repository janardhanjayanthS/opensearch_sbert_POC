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
    """Create all indexes defined in index.py if they don't already exist."""

    for index_name, index in indexes.items():
        if not os_client.indices.exists(index=index_name):
            os_client.indices.create(index=index_name, body=index)
            print(f"Created index: {index_name}")
        else:
            print(f"Index {index_name} already exists.")


def add_document(text: str, category_id: str) -> None:
    """Embed text and store it in embedding_index.

    doc_id is SHA1 of text content — re-indexing the same text overwrites the existing doc.

    Args:
        text: Raw text to embed and store.
        category_id: ID of the associated doc in category_index.
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


def add_category(category_name: str) -> str:
    """Embed category_name and store it in category_index.

    doc_id (= category_id) is SHA1 of category_name — idempotent.

    Args:
        category_name: Human-readable category label.

    Returns:
        category_id (SHA1 hex string) used as the document _id.
    """
    vector = get_vectors(text=category_name)
    category_id = hashlib.sha1(category_name.encode()).hexdigest()

    content = {
        "category_name": category_name,
        "category_id": category_id,
        "embedding": vector,
    }

    os_client.index(body=content, id=category_id, index="category_index", refresh=True)
    return category_id


def search_similar_category(category: str, size: int = 5, k: int = 5) -> dict:
    """KNN search category_index for categories semantically similar to the input.

    Args:
        category: Category label to search against.
        size: Max results to return.
        k: KNN neighbours to retrieve per shard.

    Returns:
        Dict of {category_name: category_id} for top matches.
    """
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
    """KNN vector search over embedding_index.

    Embeds user_query with the same model used at ingest, retrieves top-k nearest
    neighbours by cosine similarity via HNSW/Faiss.

    Args:
        user_query: Natural language query.
        size: Number of hits to return.
        k: KNN neighbours to retrieve per shard.
    """
    query_text = user_query.lower()
    query_vector = get_vectors(text=query_text)

    search_query = {
        "size": size,
        "_source": ["text_chunk", "category_id"],
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
        category_id = hit["_source"]["category_id"]
        print(
            f"Score (Similarity): {score:.4f} | Category: {category_id} | Text: {text}"
        )
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
