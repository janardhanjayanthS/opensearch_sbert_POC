import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import requests
from openai import OpenAI
from requests_aws4auth import AWS4Auth


def _print_retrieval_summary(
    source_metrics: dict,
    embedding_duration_ms: float,
    combined_results: list,
    multi_search_duration_ms: float,
):
    """Print formatted retrieval summary table."""
    total_requested = sum(m["requested"] for m in source_metrics.values())
    max_thread_time = max(
        (m["thread_time_ms"] for m in source_metrics.values()), default=0
    )

    summary = f"\n{'=' * 95}\n[open_search.py] 📊 RETRIEVAL SUMMARY\n{'=' * 95}\n"
    summary += f"{'Stage':<15} {'Requested':<12} {'Fetched':<12} {'Thread Time':<15} {'Total Time':<15} {'Status':<10}\n"
    summary += f"{'-' * 95}\n"
    summary += f"{'Embedding':<15} {'-':<12} {'-':<12} {'-':<15} {embedding_duration_ms:<14.2f}ms {'SUCCESS':<10}\n"
    summary += f"{'-' * 95}\n"

    for source_name, metrics in source_metrics.items():
        summary += f"{source_name:<15} {metrics['requested']:<12} {metrics['fetched']:<12} {metrics['thread_time_ms']:<14.2f}ms {metrics['total_time_ms']:<14.2f}ms {metrics['status']:<10}\n"

    summary += f"{'-' * 95}\n"
    summary += f"{'Search Total':<15} {total_requested:<12} {len(combined_results):<12} {max_thread_time:<14.2f}ms {multi_search_duration_ms:<14.2f}ms\n"
    summary += (
        f"{'(Parallel)':<15} {'':<12} {'':<12} {'(max)':<15} {'(wall-clock)':<15}\n"
    )
    summary += f"{'=' * 95}\n"

    print(summary)


def _get_aws_auth(region="us-west-2"):
    role_arn = os.environ.get("AWS_ROLE_ARN")
    token_file = os.environ.get("AWS_WEB_IDENTITY_TOKEN_FILE")

    if not role_arn or not token_file:
        raise RuntimeError(
            "IRSA environment variables missing — not running under EKS service account?"
        )

    with open(token_file, "r") as f:
        web_identity_token = f.read()

    sts = boto3.client("sts", region_name=region)
    resp = sts.assume_role_with_web_identity(
        RoleArn=role_arn,
        RoleSessionName="irsa-session",
        WebIdentityToken=web_identity_token,
    )

    creds = resp["Credentials"]

    return AWS4Auth(
        creds["AccessKeyId"],
        creds["SecretAccessKey"],
        region,
        "es",
        session_token=creds["SessionToken"],
    )


def _get_embedding(text, model_config: dict = {}):
    """Generate embedding for the given text using OpenAI's text-embedding-ada-002 model."""
    try:
        start_prev_hist = time.perf_counter()
        client = OpenAI(
            api_key=os.environ.get("PAT", "None"),
            base_url=os.environ.get(
                "base_url",
                "https://portal.ihs.amr-nonprod.devx-eks-stg.dht.live/api/llm",
            ),
        )
        response = client.embeddings.create(
            input=text,
            model=model_config.get(
                "embeddings_model",
                "global-openai-embedding-models/Text-embedding-3-large",
            ),
            extra_headers={"X-TFY-METADATA": '{"tfy_log_request":"true"}'},
        )
        # Defensive check for empty or malformed response
        if (
            not response.data
            or not hasattr(response.data[0], "embedding")
            or not response.data[0].embedding
        ):
            print("Embedding response is empty or malformed.")
            raise ValueError("Embedding response is empty or malformed.")
        prev_hist_duration_ms = (time.perf_counter() - start_prev_hist) * 1000

        return [embed.embedding for embed in response.data]
    except Exception as e:
        print(f"Failed to generate embedding: {e}")
        raise


def _hybrid_search_with_vector(
    query_text_lower: str,
    query_vector,
    size: int,
    k: int,
    bm_25_weight: float,
    vector_weight: float,
    index: str,
    model_config: dict,
    metadata_filters: dict | None = None,
):
    """Internal helper: perform hybrid search using a precomputed query_vector."""
    thread_id = threading.current_thread().ident
    thread_name = threading.current_thread().name
    source_name = metadata_filters.get("source", "N/A") if metadata_filters else "N/A"

    start_time = time.perf_counter()

    OPENSEARCH_HOST = os.environ.get(
        "OPENSEARCH_HOST",
        "https://vpc-unified-test-5mlw2xwoadfh77fm72azu4n5ga.us-west-2.es.amazonaws.com",
    )
    AWS_REGION = os.environ.get("AWS_REGION", "us-west-2")

    try:
        bool_query: dict = {
            "should": [
                {
                    "multi_match": {
                        "query": query_text_lower,
                        "fields": ["content^2"],
                        "boost": bm_25_weight,
                    }
                },
                {
                    "knn": {
                        "embeddings": {
                            "vector": query_vector[0],
                            "k": k,
                            "boost": vector_weight,
                        }
                    }
                },
            ],
        }

        # Optional: filter by metadata, such as `source`
        filters = []
        if metadata_filters:
            source_value = metadata_filters.get("source")
            if source_value is not None:
                if isinstance(source_value, list):
                    filters.append({"terms": {"source": source_value}})
                else:
                    filters.append({"term": {"source": source_value}})

        if filters:
            bool_query["filter"] = filters

        query = {
            "size": size,
            "_source": ["content", "url", "source"],
            "query": {"bool": bool_query},
        }

        url = f"{OPENSEARCH_HOST}/{index}/_search"
        awsauth = _get_aws_auth(region=AWS_REGION)

        headers = {"Content-Type": "application/json"}
        try:
            response = requests.get(
                url, auth=awsauth, headers=headers, data=json.dumps(query)
            )
        except Exception as e:
            print(f"OpenSearch request failed: {e}")
            raise
        if response.status_code != 200:
            print(f"❌ Search error: {response.status_code}")
            print(response.text)
            return "error"

        response_json = response.json()
        hits = response_json["hits"]["hits"]

        final_response = []

        for i in hits:
            src = i.get("_source", {})
            final_response.append(
                {
                    "content": src.get("content", ""),
                    "url": src.get("url"),
                    "source": src.get("source"),
                }
            )

        duration_ms = (time.perf_counter() - start_time) * 1000

        return {"results": final_response, "thread_duration_ms": duration_ms}
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        print(f"❌ Search error for {source_name}: {e}")
        return {"results": "error", "thread_duration_ms": duration_ms}


def hybrid_search(
    query_text: str,
    size: int = 100,
    k: int = 80,
    bm_25_weight: float = 1.0,
    vector_weight: float = 1.5,
    index: str = "bhoka6_knowledge",
    model_config: dict = {},
    metadata_filters: dict | None = None,
):
    """Public API: perform hybrid search, computing embeddings internally."""
    hybrid_search_start = time.perf_counter()
    print("🔎 [HYBRID_SEARCH START] Starting hybrid search execution")

    query_text_lower = query_text.lower()
    query_vector = _get_embedding(query_text_lower, model_config)

    result = _hybrid_search_with_vector(
        query_text_lower,
        query_vector,
        size,
        k,
        bm_25_weight,
        vector_weight,
        index,
        model_config,
        metadata_filters,
    )

    hybrid_search_duration_ms = (time.perf_counter() - hybrid_search_start) * 1000
    print(
        f"✅ [HYBRID_SEARCH END] Total hybrid_search execution time: {hybrid_search_duration_ms:.2f}ms"
    )

    # Handle new return format
    if isinstance(result, dict) and "results" in result:
        return result["results"]
    return result


def multi_source_hybrid_search(
    query_text: str,
    source_k_config: dict,
    bm_25_weight: float = 1.0,
    vector_weight: float = 1.5,
    index: str = "bhoka6_knowledge",
    model_config: dict | None = None,
):
    """Run hybrid (BM25 + vector) search per source with its own K.

    Example source_k_config:

        {"workday": 2, "s3": 3}

        This will:
            - compute the embedding once,
            - for each source, run a filtered hybrid search with that source
              and k equal to the configured value,
            - execute those hybrid searches in parallel threads,
            - return the concatenated list of results.
    """
    multi_search_start = time.perf_counter()

    model_cfg = model_config or {}
    query_text_lower = query_text.lower()

    embedding_start = time.perf_counter()
    query_vector = _get_embedding(query_text_lower, model_cfg)
    embedding_duration_ms = (time.perf_counter() - embedding_start) * 1000

    if not source_k_config:
        print(
            "--------- No Source K Config Provided; Running Single Hybrid Search ---------"
        )
        # Fallback to a single hybrid search using default size and k
        result = _hybrid_search_with_vector(
            query_text_lower,
            query_vector,
            size=50,
            k=45,
            bm_25_weight=bm_25_weight,
            vector_weight=vector_weight,
            index=index,
            model_config=model_cfg,
            metadata_filters=None,
        )
        # Handle new return format
        if isinstance(result, dict) and "results" in result:
            return result["results"]
        return result

    combined_results: list = []

    # Prepare valid source/k pairs
    source_k_items = []
    for source_name, k_value in source_k_config.items():
        try:
            k_int = int(k_value)
        except (TypeError, ValueError):
            continue
        if k_int <= 0:
            continue
        source_k_items.append((source_name, k_int))

    if not source_k_items:
        return []

    # Track metrics for each source
    source_metrics = {}

    # Run one hybrid search per source in parallel threads (capped workers)
    max_workers = min(len(source_k_items), 4)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_source = {}
        for source_name, k_int in source_k_items:
            source_start_time = time.perf_counter()
            future = executor.submit(
                _hybrid_search_with_vector,
                query_text_lower,
                query_vector,
                k_int,  # size
                k_int,  # k
                bm_25_weight,
                vector_weight,
                index,
                model_cfg,
                {"source": source_name},
            )
            future_to_source[future] = (source_name, k_int, source_start_time)

        for future in as_completed(future_to_source):
            source_name, k_requested, source_start_time = future_to_source[future]
            source_duration_ms = (time.perf_counter() - source_start_time) * 1000
            try:
                result = future.result()

                # Handle new return format
                if isinstance(result, dict):
                    results = result.get("results", [])
                    thread_duration_ms = result.get("thread_duration_ms", 0)
                else:
                    results = result
                    thread_duration_ms = 0

            except Exception as exc:
                print(
                    f"❌ [EXCEPTION] Source {source_name} generated an exception: {exc}"
                )
                source_metrics[source_name] = {
                    "requested": k_requested,
                    "fetched": 0,
                    "total_time_ms": source_duration_ms,
                    "thread_time_ms": 0,
                    "status": "ERROR",
                }
                continue

            if isinstance(results, list):
                combined_results.extend(results)
                source_metrics[source_name] = {
                    "requested": k_requested,
                    "fetched": len(results),
                    "total_time_ms": source_duration_ms,
                    "thread_time_ms": thread_duration_ms,
                    "status": "SUCCESS",
                }
            else:
                source_metrics[source_name] = {
                    "requested": k_requested,
                    "fetched": 0,
                    "total_time_ms": source_duration_ms,
                    "thread_time_ms": thread_duration_ms,
                    "status": "ERROR",
                }

    multi_search_duration_ms = (time.perf_counter() - multi_search_start) * 1000

    # Print summary table
    _print_retrieval_summary(
        source_metrics,
        embedding_duration_ms,
        combined_results,
        multi_search_duration_ms,
    )

    return combined_results
