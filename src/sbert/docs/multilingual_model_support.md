# Multilingual Model Support for Semantic Chunking

## `all-MiniLM-L6-v2` — Multilingual Support?

**No.** `all-MiniLM-L6-v2` is trained primarily on **English** data. It will produce poor-quality embeddings for German text, resulting in bad semantic similarity scores and broken chunk boundaries.

---

## Recommended Multilingual Models (from `sentence-transformers`)

These all work as drop-in replacements for `SentenceTransformer(model_name)`:

| Model | Languages | Notes |
|---|---|---|
| `paraphrase-multilingual-MiniLM-L12-v2` | 50+ langs | Best balance of speed & quality; direct multilingual successor to `all-MiniLM-L6-v2` |
| `paraphrase-multilingual-mpnet-base-v2` | 50+ langs | Higher quality than the above, slower |
| `distiluse-base-multilingual-cased-v2` | 50+ langs | Good for semantic similarity tasks |
| `multilingual-e5-small` | 100 langs | Microsoft E5 family, fast |
| `multilingual-e5-base` | 100 langs | Better quality E5, moderate speed |
| `multilingual-e5-large` | 100 langs | Best E5 quality, slowest |
| `intfloat/multilingual-e5-large-instruct` | 100 langs | Instruction-tuned, highest quality |

---

## Recommendation for English + German PDF Chunking

The best pragmatic choices are:

1. **`paraphrase-multilingual-MiniLM-L12-v2`** — closest drop-in to `all-MiniLM-L6-v2`, fast, good quality
2. **`paraphrase-multilingual-mpnet-base-v2`** — better semantic quality if speed isn't critical

Change line 12 in `chunking_class.py`:
```python
model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
```

---

## German-Specific Models

> **Note:** `deepset/gbert-base` and `deepset/gbert-large` are raw masked language models (MLM) — they are **not** sentence transformers and cannot be loaded directly with `SentenceTransformer()`. Do not use them.

| Model | Notes |
|---|---|
| `T-Systems-onsite/german-roberta-sentence-transformer-v2` | Best overall German sentence transformer; RoBERTa-based, strong semantic similarity. Requires `sentencepiece` + `protobuf`: `pip install sentencepiece protobuf` |
| `deutsche-telekom/gbert-large-paraphrase-cosine` | Fine-tuned for paraphrase/similarity, works directly |
| `PM-AI/bi-encoder_msmarco_bert-base_german` | Fine-tuned for retrieval/search tasks (relevant for OpenSearch), works directly (working, kinda - creates more chunks) |

### Which to pick?

- **Pure German docs** → `T-Systems-onsite/german-roberta-sentence-transformer-v2` — purpose-built, best semantic chunking quality for German
- **Mixed English + German docs** → stick with `paraphrase-multilingual-mpnet-base-v2` — a German-only model will produce poor embeddings on English text
- **OpenSearch retrieval focus** → `PM-AI/bi-encoder_msmarco_bert-base_german` — trained on MSMARCO, closer to search/retrieval use cases

Since documents are predominantly German, `T-Systems-onsite/german-roberta-sentence-transformer-v2` is the strongest choice for chunking quality.

---

## Additional Note: German Sentence Splitting

The `split_into_sentences` regex in `chunking_class.py` (line 23) uses `(?=[A-Z])` as a sentence boundary signal. This **won't split German sentences correctly** — German capitalizes all nouns, so every noun would be treated as a sentence boundary. This is a separate issue worth addressing when supporting German documents.
