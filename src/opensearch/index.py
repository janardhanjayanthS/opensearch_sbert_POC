embedding = {
    "settings": {"index.knn": True},
    "mappings": {
        "properties": {
            "text_chunk": {"type": "text"},
            "file_path": {"type": "text"},
            "category_id": {"type": "keyword"},
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

category = {
    "mappings": {
        "settings": {"index.knn": True},
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "name": {"type": "text"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 3072,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "fiass"
                    }
                }
            }
        }
    }
}

indexes = {"embedding_index": embedding, "category_index": category}
