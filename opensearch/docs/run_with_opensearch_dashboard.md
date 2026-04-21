# OpenSearch Dashboard — Local Setup Guide

## Prerequisites

- Docker & Docker Compose installed
- `.env` file in the same directory as `docker-compose.yml`

---

## 1. Environment Variables

Create a `.env` file:

```env
OPENSEARCH_ADMIN_PASSWORD=YourStrongPassword123!
```

> Add `.env` to `.gitignore` and commit a `.env.example` with placeholder values instead.

---

## 2. docker-compose.yml

```yaml
version: '3'
services:
  opensearch-node:
    image: opensearchproject/opensearch:latest
    container_name: opensearch-local
    environment:
      - cluster.name=rag-cluster
      - node.name=rag-node
      - discovery.type=single-node
      - bootstrap.memory_lock=true
      - "OPENSEARCH_JAVA_OPTS=-Xms512m -Xmx512m"
      - OPENSEARCH_INITIAL_ADMIN_PASSWORD=${OPENSEARCH_ADMIN_PASSWORD}
    ulimits:
      memlock:
        soft: -1
        hard: -1
    ports:
      - 9200:9200

  opensearch-dashboards:
    image: opensearchproject/opensearch-dashboards:latest
    container_name: opensearch-dashboards
    ports:
      - 5601:5601
    environment:
      - OPENSEARCH_HOSTS=["https://opensearch-node:9200"]
      - OPENSEARCH_USERNAME=admin
      - OPENSEARCH_PASSWORD=${OPENSEARCH_ADMIN_PASSWORD}
    depends_on:
      - opensearch-node
```

---

## 3. Start the Containers

```bash
docker-compose up -d
```

Wait **30–60 seconds** for OpenSearch to fully boot before opening the dashboard.

---

## 4. Open the Dashboard

Go to **http://localhost:5601** and log in:

| Field    | Value                          |
|----------|--------------------------------|
| Username | `admin`                        |
| Password | Value from your `.env` file    |

---

## 5. Create an Index Pattern

The dashboard needs to know which index to display — do this once per index.

1. Click the **☰ hamburger menu** (top left)
2. Go to **Management → Dashboards Management → Index Patterns**
3. Click **Create index pattern**
4. Type your exact index name (e.g. `openai_rag_index_one`) — the list of available indices is shown below the input field
5. Click **Next step** → **Create index pattern**

> **Note:** The name must match exactly. If `Next step` is greyed out, the name doesn't match any existing index — check the list shown on the same page and use that name.

---

## 6. View Your Data

1. **☰ menu → Discover**
2. Select your index pattern from the dropdown (top left)
3. All stored documents appear in a table — each row is one document with its fields (`text_chunk`, `file_path`, `embedding`, etc.)

---

## 7. Run Raw Queries (Dev Tools)

Like a SQL console, but for OpenSearch.

1. **☰ menu → Dev Tools**
2. Use the editor on the left and hit the **▶ play button** to run

**Fetch all documents:**
```json
GET /openai_rag_index_one/_search
{
  "query": { "match_all": {} }
}
```

**Count total documents:**
```json
GET /openai_rag_index_one/_count
```

**View index schema (like `DESCRIBE TABLE`):**
```json
GET /openai_rag_index_one/_mapping
```

---

## 8. Dark Mode (Optional)

- Click your **user icon** (top right) → **Turn on dark mode**

Or via settings:
1. **☰ → Management → Dashboards Management → Advanced Settings**
2. Search `dark` → toggle **Dark mode** on → Save
