"""
API Endpoint Tests for KnowledgeAI Backend (FastAPI)
Covers:
- GET / health check
- POST /search with valid input, rewritten terms, and empty validation error (400)
- GET /similar-queries with default and custom threshold
- GET /article-audit with outdated and duplicate articles
- POST /recommendations for single questions and batch dataset analysis
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Tests for GET /"""

    def test_health_check_status_code_and_fields(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "Knowledge Base" in data["service"]
        assert "version" in data
        assert "data_status" in data
        assert data["data_status"]["articles_count"] > 0
        assert data["data_status"]["queries_count"] > 0


class TestSearchEndpoint:
    """Tests for POST /search"""

    def test_search_valid_order_query(self):
        payload = {"question": "Where is my order?"}
        response = client.post("/search", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) > 0

        first = data["results"][0]
        assert "article_id" in first
        assert "title" in first
        assert "content" in first
        assert "similarity" in first
        assert first["similarity"] > 0.0

    def test_search_with_vocabulary_rewrite(self):
        payload = {"question": "Where is my parcel?"}
        response = client.post("/search", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) > 0
        # Results should match order/package tracking articles
        top_ids = [r["article_id"] for r in data["results"]]
        assert "A01" in top_ids or "A02" in top_ids

    def test_search_empty_question_returns_400(self):
        payload = {"question": "   "}
        response = client.post("/search", json=payload)
        assert response.status_code == 400
        data = response.json()
        assert "error" in data or "message" in data

    def test_search_missing_question_returns_422(self):
        payload = {}
        response = client.post("/search", json=payload)
        assert response.status_code == 422


class TestSimilarQueriesEndpoint:
    """Tests for GET /similar-queries"""

    def test_get_similar_queries_default(self):
        response = client.get("/similar-queries")
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "similar_queries" in data
        assert "total_pairs" in data
        assert data["total_pairs"] > 0

        pair = data["results"][0]
        assert "query_1" in pair
        assert "query_2" in pair
        assert "similarity" in pair
        assert pair["similarity"] >= 0.5

    def test_get_similar_queries_custom_threshold(self):
        response = client.get("/similar-queries?threshold=0.7")
        assert response.status_code == 200
        data = response.json()
        assert data["threshold"] == 0.7
        for pair in data["results"]:
            assert pair["similarity"] >= 0.7

    def test_get_similar_queries_invalid_threshold_returns_422(self):
        response = client.get("/similar-queries?threshold=2.5")
        assert response.status_code == 422


class TestArticleAuditEndpoint:
    """Tests for GET /article-audit"""

    def test_get_article_audit_default(self):
        response = client.get("/article-audit")
        assert response.status_code == 200
        data = response.json()

        assert "outdated_articles" in data
        assert "similar_articles" in data
        assert "summary" in data

        # Validate outdated articles
        assert len(data["outdated_articles"]) >= 4
        outdated_item = data["outdated_articles"][0]
        assert "article_id" in outdated_item
        assert "title" in outdated_item
        assert "updated_at" in outdated_item
        assert "days_since_update" in outdated_item
        assert "reason" in outdated_item

        # Validate similar articles
        assert len(data["similar_articles"]) >= 2
        similar_item = data["similar_articles"][0]
        assert "article_id_1" in similar_item
        assert "title_1" in similar_item
        assert "article_id_2" in similar_item
        assert "title_2" in similar_item
        assert "similarity" in similar_item
        assert "reason" in similar_item

    def test_get_article_audit_custom_parameters(self):
        response = client.get("/article-audit?days_threshold=1000&similarity_threshold=0.6")
        assert response.status_code == 200
        data = response.json()
        assert data["summary"]["days_threshold"] == 1000
        assert data["summary"]["similarity_threshold"] == 0.6


class TestRecommendationsEndpoint:
    """Tests for POST /recommendations"""

    def test_recommendation_for_uncovered_question(self):
        payload = {"question": "Can I pay using cryptocurrency like Bitcoin?"}
        response = client.post("/recommendations", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 1
        item = data["results"][0]
        assert item["is_gap"] is True
        assert item["recommended_action"] == "Create new help article"
        assert "reason" in item

    def test_recommendation_for_covered_question(self):
        payload = {"question": "Where is my order?"}
        response = client.post("/recommendations", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        item = data["results"][0]
        assert item["is_gap"] is False
        assert "Topic adequately covered" in item["recommended_action"]

    def test_recommendations_batch_no_question(self):
        # Empty payload evaluates all queries in queries.csv
        response = client.post("/recommendations", json={})
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert data["total_gaps"] > 0
        first_gap = data["results"][0]
        assert "query" in first_gap
        assert "highest_similarity" in first_gap
        assert "recommended_action" in first_gap
        assert "reason" in first_gap
