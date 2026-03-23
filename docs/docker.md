# OpenSearch — Docker Setup

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

---

## Run OpenSearch with Docker

### Single node (dev)

```bash
docker run -d \
  --name opensearch \
  -p 9200:9200 -p 9600:9600 \
  -e "discovery.type=single-node" \
  -e "OPENSEARCH_INITIAL_ADMIN_PASSWORD=StrongPassword123!" \
  opensearchproject/opensearch:latest
```

### Verify it's running

```bash
curl -k -u admin:StrongPassword123! https://localhost:9200
```

You should see a JSON response with `"cluster_name"` and `"status"`.

---

## Run with Docker Compose (recommended)

Create a `docker-compose.yml` in the project root:

```yaml
services:
  opensearch:
    image: opensearchproject/opensearch:latest
    container_name: opensearch
    environment:
      - discovery.type=single-node
      - OPENSEARCH_INITIAL_ADMIN_PASSWORD=StrongPassword123!
    ports:
      - "9200:9200"
      - "9600:9600"
    volumes:
      - opensearch-data:/usr/share/opensearch/data

  opensearch-dashboards:
    image: opensearchproject/opensearch-dashboards:latest
    container_name: opensearch-dashboards
    ports:
      - "5601:5601"
    environment:
      - OPENSEARCH_HOSTS=["https://opensearch:9200"]
      - OPENSEARCH_SSL_VERIFICATIONMODE=none
    depends_on:
      - opensearch

volumes:
  opensearch-data:
```

Then start it:

```bash
docker compose up -d
```

| Service | URL |
|---|---|
| OpenSearch API | https://localhost:9200 |
| OpenSearch Dashboards | http://localhost:5601 |

---

## Stop / Remove

```bash
docker compose down        # stop containers
docker compose down -v     # stop + delete volume (all indexed data)
```

---

## Notes

- SSL is enabled by default. The Python client in this project connects with `verify_certs=False` (dev only).
- The password `StrongPassword123!` must match `OPENSEARCH_INITIAL_ADMIN_PASSWORD` in the container and the credentials in `opensearch/opensearch.py`.
- The KNN plugin (`opensearch-knn`) is included in the official image — no extra installation needed.
