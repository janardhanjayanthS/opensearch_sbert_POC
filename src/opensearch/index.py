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
        "properties": {
            "category_id": {"type": "keyword"},
            "category_name": {"type": "text"},
        }
    }
}

indexes = {"embedding_index": embedding, "category_index": category}
