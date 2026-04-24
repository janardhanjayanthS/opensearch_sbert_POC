# search_documents() — Parameter Reference

## Function Signature

```python
def search_documents(
    user_query: str,
    size: int = 5,
    k: int = 5,
    bm25_weight: float = 1.0,
    vector_weight: float = 1.0,
) -> None
```

---

## Parameters

### `user_query` — `str`
Natural language input from the user. Gets lowercased and embedded before search.

---

### `size` — `int` (default: `5`)
Max number of results returned from OpenSearch.

| Value | Behavior |
|---|---|
| `5` | Top 5 results (default) |
| `10` | Broader result set, more noise possible |
| `1` | Fastest, single best match only |

---

### `k` — `int` (default: `5`)
Number of nearest neighbours the HNSW graph retrieves **per shard** during KNN search.

`k` must be `>= size`. If `k < size`, you may get fewer results than expected.

| Value | Behavior |
|---|---|
| `5` | Tight recall — fast |
| `20` | Wider candidate pool — slightly slower, more accurate ranking |
| `50` | Maximum recall — use for evaluation/debugging only |

---

### `bm25_weight` — `float` (default: `1.0`)
Boost multiplier applied to the BM25 (keyword) component score.

BM25 rewards **exact or partial word matches** in `text_chunk`. Higher values make keyword overlap matter more in the final ranking.

| `bm25_weight` | Effect |
|---|---|
| `0.0` | Pure semantic search — keywords ignored entirely |
| `1.0` | Balanced (default) |
| `2.0+` | Keyword matches dominate — better for specific names, numbers, dates |

**Good for:** queries with specific words like `"dentist"`, `"Zara"`, `"B-14"`, `"November 14th"`

---

### `vector_weight` — `float` (default: `1.0`)
Boost multiplier applied to the KNN cosine similarity score.

KNN rewards **semantic closeness** regardless of exact words. Higher values make meaning-based similarity matter more.

| `vector_weight` | Effect |
|---|---|
| `0.0` | Pure keyword search — embeddings ignored |
| `1.0` | Balanced (default) |
| `2.0+` | Meaning-based results dominate — better for vague or rephrased queries |

**Good for:** queries like `"where did I park"`, `"something about my health"`, `"book I was reading"`

---

## Weight Combinations

| `bm25_weight` | `vector_weight` | Best for |
|---|---|---|
| `1.0` | `1.0` | General use (default) |
| `2.0` | `1.0` | Queries with exact known words |
| `1.0` | `2.0` | Vague / rephrased natural language |
| `0.0` | `1.0` | Pure semantic — ignore all keywords |
| `1.0` | `0.0` | Pure keyword — ignore meaning |

---

## How Scores Combine

The query uses `bool.should` — OpenSearch sums the boosted scores from both components:

```
final_score = (bm25_score × bm25_weight) + (knn_score × vector_weight)
```

> BM25 scores and KNN cosine scores are on different scales. Use relative ratios between weights, not absolute values — `bm25=1, vector=2` is equivalent to `bm25=0.5, vector=1`.
