import os

from dotenv import load_dotenv
from openai import OpenAI
from opensearchpy import OpenSearch

load_dotenv()

# --- 1. Initialize Clients ---
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Connect to the local OpenSearch Docker container
os_client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_auth=("admin", "StrongPassword123!"),  # Matches the Docker password
    use_ssl=True,
    verify_certs=False,  # Set to False for local Docker testing
    ssl_show_warn=False,
)

index_name = "my-openai-rag-index"


# --- 2. Create the Index (The "Table") ---
# We must define the index to accept 3072-dimensional vectors
def create_index():
    index_body = {
        "settings": {
            "index.knn": True  # Enables vector search
        },
        "mappings": {
            "properties": {
                "text_chunk": {"type": "text"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 3072,  # MUST match OpenAI text-embedding-3-large
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


# --- 3. Ingest Data ---
def add_document(doc_id, text):
    # Get the embedding from OpenAI
    response = openai_client.embeddings.create(
        model="text-embedding-3-large", input=text
    )
    vector = response.data[0].embedding

    # Construct the JSON document
    document = {"text_chunk": text, "embedding": vector}

    # Store it in OpenSearch
    os_client.index(index=index_name, body=document, id=doc_id, refresh=True)
    print(f"Added document {doc_id} to OpenSearch.")


# --- 4. Retrieve/Search Data ---
def search(user_query):
    # Turn the user's query into a vector so we can compare it
    query_response = openai_client.embeddings.create(
        model="text-embedding-3-large", input=user_query
    )
    query_vector = query_response.data[0].embedding

    # Formulate the OpenSearch k-NN query
    search_query = {
        "size": 5,  # Number of results to return
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_vector,
                    "k": 5,  # Look for the top 2 closest matches
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


# --- 5. Run the Workflow ---
if __name__ == "__main__":
    create_index()

    # Let's add some sample knowledge
    add_document(doc_id=1, text="The Eiffel Tower is located in Paris, France.")
    add_document(doc_id=2, text="Python is a popular programming language for AI.")
    add_document(
        doc_id=3,
        text="OpenSearch is a distributed search engine fork of Elasticsearch.",
    )

    # Let's ask a question that requires semantic understanding
    search("Where can I find a famous French landmark?")
