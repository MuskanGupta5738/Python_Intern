# Knowledge Base Improvement Advisory Agent (KnowledgeAI) - Backend Documentation

FastAPI backend and Python analysis engine for the **Knowledge Base Improvement Advisory Agent (KnowledgeAI)** college project.

This backend provides intelligent analysis of customer service help articles and customer inquiry logs using natural language processing (NLP), **TF-IDF**, and **Cosine Similarity**, with zero paid APIs, no OpenAI dependency, and no external database required.

---

## 1. Features Implemented

1. **Relevant Article Search (`POST /search`)**:
   - Searches help articles based on customer questions.
   - Computes TF-IDF vectors and Cosine Similarity scores.
   - Applies rule-based query rewriting to match terms like "parcel" to "package/shipment".
   - Returns ranked matching articles with titles, content, article IDs, and similarity scores.

2. **Similar Customer Queries Detection (`GET /similar-queries`)**:
   - Analyzes customer query logs from `data/queries.csv`.
   - Identifies repeated or semantically similar customer inquiries.
   - Returns query pairs with similarity scores and configurable thresholds.

3. **Similar / Duplicate Article Detection (`GET /article-audit`)**:
   - Detects overlapping or duplicate help articles in `data/articles.csv`.
   - Returns article pairs with titles, IDs, similarity scores, and review reasons.

4. **Outdated Article Audit (`GET /article-audit`)**:
   - Inspects `updated_at` dates against a configurable review-age threshold (default: 180 days).
   - Flags older articles needing content refresh or verification.

5. **Rule-Based Query Rewriting (Vocabulary Normalization)**:
   - Expands common e-commerce synonyms (e.g., *parcel* → *package*, *passcode/pwd* → *password*, *broken goods* → *damaged defective*, *money back* → *refund*).
   - Improves recall while preserving original keywords to prevent false alerts.

6. **Knowledge Gap Recommendations (`POST /recommendations`)**:
   - Identifies customer questions with low similarity scores against existing help articles (even after query rewriting).
   - Can evaluate individual queries or batch-scan the entire `queries.csv` dataset.
   - Returns actionable advice and human-understandable reasons for creating new help topics.

---

## 2. Project Directory Structure

```text
knowledgeai-backend/
├── main.py                     # FastAPI routes, schemas, CORS, and error handlers
├── analysis_engine.py          # TF-IDF, Cosine Similarity, and auditing logic
├── requirements.txt            # Python dependencies
├── README_BACKEND.md           # Backend documentation & API contract
├── data/
│   ├── articles.csv            # 12 sample help articles (fictional)
│   └── queries.csv             # 32 sample customer queries (fictional)
└── tests/
    ├── test_analysis_engine.py # Analysis engine unit & integration tests
    └── test_api.py             # FastAPI endpoint and status code tests
```

---

## 3. Windows & VS Code Setup Instructions

### Prerequisites
- Python 3.9+ installed on Windows (Python 3.10, 3.11, 3.12, 3.13 supported).
- VS Code (Visual Studio Code).

### Step 1: Open Project in VS Code
1. Open VS Code.
2. Click **File** -> **Open Folder...** and select the `knowledgeai-backend` folder.
3. Open a terminal inside VS Code: press `` Ctrl + ` `` or go to **Terminal** -> **New Terminal**.

### Step 2: Create and Activate Virtual Environment (Recommended)
In the VS Code terminal (PowerShell or Command Prompt):
```powershell
# Create a virtual environment named .venv
python -m venv .venv

# Activate the virtual environment on Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# (If running in Command Prompt / cmd.exe instead):
# .venv\Scripts\activate.bat
```

> **Note on PowerShell Script Execution**: If you see an `ExecutionPolicy` error when activating the virtual environment, run:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
> ```
> and re-run `.\.venv\Scripts\Activate.ps1`.

### Step 3: Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 4. How to Run the Backend Server

Start the FastAPI server with auto-reload:
```powershell
uvicorn main:app --reload
```

The server will start on:
```text
http://127.0.0.1:8000
```

### Interactive API Documentation (Swagger UI)
Open your browser and navigate to:
```text
http://127.0.0.1:8000/docs
```
You can interactively test every endpoint, inspect request bodies, and check JSON responses directly in your browser.

An alternative ReDoc view is available at:
```text
http://127.0.0.1:8000/redoc
```

---

## 5. Frontend Integration Contract (Streamlit Teammate Guide)

The Streamlit frontend should connect to the backend URL (configurable):
`http://127.0.0.1:8000`

### Summary of JSON Fields
| Field Name | Description | Used In |
| :--- | :--- | :--- |
| `results` | Primary array of search items, query pairs, or recommendations | `/search`, `/similar-queries`, `/recommendations` |
| `article_id` | Unique ID of article (e.g. `"A01"`) | `/search`, `/article-audit` |
| `title` | Article headline | `/search`, `/article-audit` |
| `content` | Article body text | `/search` |
| `similarity` | Cosine similarity score (float between 0.0 and 1.0) | `/search`, `/similar-queries`, `/article-audit` |
| `query_1`, `query_2` | Query text for similar query pairs | `/similar-queries` |
| `updated_at` | Article last updated date (YYYY-MM-DD) | `/article-audit` |
| `reason` | Human-readable explanation for audit flag or recommendation | `/article-audit`, `/recommendations` |
| `message`, `error` | Meaningful error details | Error responses (400, 422, 500) |

---

### Endpoint 1: Backend Health Check
- **Method**: `GET`
- **Path**: `/`
- **Purpose**: Verify backend is reachable and dataset counts.

#### Response Example (`200 OK`):
```json
{
  "status": "healthy",
  "service": "Knowledge Base Improvement Advisory Agent",
  "version": "1.0.0",
  "message": "KnowledgeAI backend is running smoothly.",
  "data_status": {
    "articles_count": 12,
    "queries_count": 32
  }
}
```

#### Streamlit Python Requests Snippet:
```python
import requests

BACKEND_URL = "http://127.0.0.1:8000"
response = requests.get(f"{BACKEND_URL}/")
if response.status_code == 200:
    data = response.json()
    print("Backend Status:", data["status"])
```

---

### Endpoint 2: Relevant Article Search
- **Method**: `POST`
- **Path**: `/search`
- **Purpose**: Search existing help articles using TF-IDF and cosine similarity with vocabulary normalization.

#### Request Example:
```json
{
  "question": "Where is my order?",
  "top_k": 3,
  "min_threshold": 0.0,
  "use_rewrite": true
}
```
*(Only `"question"` is required; `"top_k"`, `"min_threshold"`, and `"use_rewrite"` are optional)*

#### Response Example (`200 OK`):
```json
{
  "query": "Where is my order?",
  "rewritten_query": "where is my track tracking order",
  "results": [
    {
      "article_id": "A01",
      "title": "How to Track Your Order",
      "content": "To track your order log into your account and navigate to My Orders. Select your order to view carrier tracking number shipment status and parcel delivery updates.",
      "similarity": 0.69
    },
    {
      "article_id": "A07",
      "title": "Order Cancellation and Modification Guidelines",
      "content": "You can cancel or modify an order within 30 minutes of purchase from your Order History by clicking Cancel Order...",
      "similarity": 0.60
    },
    {
      "article_id": "A02",
      "title": "Track Your Order and Package Status",
      "content": "To track your order and package status log into your account or enter your tracking number...",
      "similarity": 0.58
    }
  ]
}
```

#### Streamlit Python Requests Snippet:
```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/search",
    json={"question": "Where is my order?"}
)
if response.status_code == 200:
    results = response.json().get("results", [])
    for article in results:
        print(article["article_id"], article["title"], article["similarity"])
```

---

### Endpoint 3: Similar Customer Queries
- **Method**: `GET`
- **Path**: `/similar-queries`
- **Query Parameters**:
  - `threshold` (optional float, default `0.5`, range `0.0` to `1.0`)

#### Request Example:
```text
GET http://127.0.0.1:8000/similar-queries?threshold=0.5
```

#### Response Example (`200 OK`):
```json
{
  "total_pairs": 6,
  "threshold": 0.5,
  "results": [
    {
      "query_1": "Where is my parcel?",
      "query_2": "Can you track my parcel delivery?",
      "similarity": 0.78
    },
    {
      "query_1": "When will I receive my refund to my credit card?",
      "query_2": "When will my refund be credited to my payment card?",
      "similarity": 0.75
    },
    {
      "query_1": "Where is my order?",
      "query_2": "Where is my package right now?",
      "similarity": 0.66
    }
  ],
  "similar_queries": [ ... ]
}
```

#### Streamlit Python Requests Snippet:
```python
import requests

response = requests.get("http://127.0.0.1:8000/similar-queries", params={"threshold": 0.5})
if response.status_code == 200:
    data = response.json()
    for pair in data["results"]:
        print(f"{pair['query_1']} <--> {pair['query_2']} (score: {pair['similarity']})")
```

---

### Endpoint 4: Outdated & Similar Article Audit
- **Method**: `GET`
- **Path**: `/article-audit`
- **Query Parameters**:
  - `days_threshold` (optional integer, default `180` days)
  - `similarity_threshold` (optional float, default `0.5`, range `0.0` to `1.0`)

#### Request Example:
```text
GET http://127.0.0.1:8000/article-audit?days_threshold=180&similarity_threshold=0.5
```

#### Response Example (`200 OK`):
```json
{
  "outdated_articles": [
    {
      "article_id": "A11",
      "title": "International Shipping and Customs Duties",
      "updated_at": "2023-08-14",
      "days_since_update": 1168,
      "reason": "Article has not been updated in 1168 days (threshold is 180 days). Review recommended."
    },
    {
      "article_id": "A06",
      "title": "Accepted Payment Methods and Billing Inquiries",
      "updated_at": "2023-11-20",
      "days_since_update": 1070,
      "reason": "Article has not been updated in 1070 days (threshold is 180 days). Review recommended."
    }
  ],
  "similar_articles": [
    {
      "article_id_1": "A01",
      "title_1": "How to Track Your Order",
      "article_id_2": "A02",
      "title_2": "Track Your Order and Package Status",
      "similarity": 0.68,
      "reason": "High content overlap (68% similarity) detected between these articles."
    },
    {
      "article_id_1": "A03",
      "title_1": "Standard Return and Refund Policy",
      "article_id_2": "A12",
      "title_2": "Customer Return Policy and Refund Guidelines",
      "similarity": 0.67,
      "reason": "High content overlap (67% similarity) detected between these articles."
    }
  ],
  "summary": {
    "total_articles": 12,
    "outdated_count": 5,
    "similar_pair_count": 2,
    "days_threshold": 180,
    "similarity_threshold": 0.5
  }
}
```

#### Streamlit Python Requests Snippet:
```python
import requests

response = requests.get(
    "http://127.0.0.1:8000/article-audit",
    params={"days_threshold": 180, "similarity_threshold": 0.5}
)
if response.status_code == 200:
    audit = response.json()
    print("Outdated Articles Count:", len(audit["outdated_articles"]))
    print("Duplicate Articles Count:", len(audit["similar_articles"]))
```

---

### Endpoint 5: Knowledge Gap Recommendations
- **Method**: `POST`
- **Path**: `/recommendations`
- **Purpose**:
  - Test a specific question to see if it is a knowledge gap.
  - Or omit the question to scan `data/queries.csv` for all customer topics lacking coverage.

#### Request Option A: Check Specific Customer Question
```json
{
  "question": "Can I pay using cryptocurrency like Bitcoin?",
  "gap_threshold": 0.30
}
```

#### Response Option A (`200 OK`):
```json
{
  "gap_threshold": 0.30,
  "total_gaps": 1,
  "results": [
    {
      "query": "Can I pay using cryptocurrency like Bitcoin?",
      "rewritten_query": "can i pay using crypto cryptocurrency bitcoin like bitcoin for payment",
      "highest_similarity": 0.25,
      "recommended_action": "Create new help article",
      "suggested_topic": "Can I Pay Using Cryptocurrency Like Bitcoin",
      "reason": "No relevant help article found. Highest similarity score is 0.25 (below threshold 0.30) even after vocabulary normalization. Customers asking this question cannot find answers.",
      "is_gap": true
    }
  ]
}
```

#### Request Option B: Batch Scan All Customer Queries (Dataset-wide)
```json
{}
```

#### Response Option B (`200 OK`):
```json
{
  "gap_threshold": 0.30,
  "total_gaps": 5,
  "results": [
    {
      "query_id": "Q31",
      "query": "How do I join your loyalty rewards program?",
      "rewritten_query": "how do i join your loyalty rewards program",
      "highest_similarity": 0.0,
      "recommended_action": "Create new help article",
      "suggested_topic": "How Do I Join Your Loyalty Rewards Program",
      "reason": "Customer query has maximum similarity score of 0.00 (below threshold 0.30) across all articles, even after rule-based query rewriting.",
      "is_gap": true
    },
    {
      "query_id": "Q30",
      "query": "Do you accept cryptocurrency like Bitcoin for payment?",
      "rewritten_query": "do you accept crypto cryptocurrency bitcoin like bitcoin for payment",
      "highest_similarity": 0.25,
      "recommended_action": "Create new help article",
      "suggested_topic": "Do You Accept Cryptocurrency Like Bitcoin For Payment",
      "reason": "Customer query has maximum similarity score of 0.25 (below threshold 0.30) across all articles, even after rule-based query rewriting.",
      "is_gap": true
    }
  ]
}
```

---

## 6. Error Handling Contract

All endpoints return standard HTTP status codes with structured JSON error objects:

| Status Code | Scenario | Example Response |
| :--- | :--- | :--- |
| `400 Bad Request` | Empty question or invalid input | `{"error": "The 'question' field cannot be empty. Please provide a valid customer question.", "message": "..."}` |
| `422 Unprocessable Entity` | Pydantic validation error (e.g. missing JSON body) | Standard FastAPI validation details array |
| `500 Internal Server Error` | Unexpected backend error | `{"error": "InternalServerError", "message": "Error details..."}` |

---

## 7. Running the Automated Tests

The backend includes a comprehensive automated test suite with **31 unit and integration tests** covering:
- Text normalization and vocabulary expansion
- Article TF-IDF cosine similarity search
- Similar query detection
- Similar/duplicate article detection
- Outdated article audit dates
- Knowledge gap recommendations
- Empty input validation and missing CSV handling
- FastAPI endpoint responses and HTTP status codes

Run tests using pytest:
```powershell
python -m pytest tests/ -v
```

Expected output:
```text
======================== 31 passed in 4.51s ========================
```

---

## 8. Dataset Fictional Data Disclaimer

The datasets provided in `data/articles.csv` and `data/queries.csv` contain **entirely realistic fictional data** created exclusively for academic and demonstration purposes for this college project. No real customer information, personal data, or proprietary commercial content is included.
