"""
Unit and Integration Tests for AnalysisEngine
Covers:
- Text normalization
- Vocabulary normalization (query rewriting)
- Help article search with TF-IDF and cosine similarity
- Similar customer query detection
- Similar/duplicate article detection
- Outdated article audit
- Knowledge gap recommendations
- Edge cases (empty inputs, non-existent files)
"""

import os
import pytest
from datetime import datetime
import pandas as pd

import sys
# Add parent directory to path so analysis_engine can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis_engine import (
    AnalysisEngine,
    normalize_text,
    rewrite_query,
    DEFAULT_ARTICLES_PATH,
    DEFAULT_QUERIES_PATH,
)


@pytest.fixture(scope="module")
def engine():
    """Initializes and returns an AnalysisEngine instance with default datasets."""
    return AnalysisEngine(
        articles_path=DEFAULT_ARTICLES_PATH,
        queries_path=DEFAULT_QUERIES_PATH,
    )


class TestNormalizationAndRewriting:
    """Tests for text normalization and rule-based query rewriting."""

    def test_normalize_text(self):
        raw = "Hello! Where IS my... Parcel???  "
        normalized = normalize_text(raw)
        assert normalized == "hello where is my parcel"
        assert normalize_text("") == ""
        assert normalize_text(None) == ""

    def test_rewrite_query_vocab_expansion(self):
        query = "Where is my parcel?"
        rewritten = rewrite_query(query)
        assert "package" in rewritten
        assert "parcel" in rewritten

    def test_rewrite_query_passcode(self):
        query = "Reset my passcode please"
        rewritten = rewrite_query(query)
        assert "password" in rewritten

    def test_rewrite_query_broken_goods(self):
        query = "I received broken goods"
        rewritten = rewrite_query(query)
        assert "damaged" in rewritten or "defective" in rewritten

    def test_rewrite_query_empty_input(self):
        assert rewrite_query("") == ""
        assert rewrite_query("   ") == ""
        assert rewrite_query(None) == ""


class TestArticleSearch:
    """Tests for TF-IDF help article search."""

    def test_search_exact_concept(self, engine):
        res = engine.search_articles("Where is my order?", top_k=3)
        assert "results" in res
        assert len(res["results"]) > 0
        top = res["results"][0]
        assert "article_id" in top
        assert "title" in top
        assert "similarity" in top
        assert top["similarity"] > 0.4
        # A01 or A02 should be in top results
        top_ids = [r["article_id"] for r in res["results"]]
        assert "A01" in top_ids or "A02" in top_ids

    def test_search_with_vocabulary_mismatch(self, engine):
        # 'parcel' should be rewritten to match 'package' / 'shipment'
        res = engine.search_articles("Where is my parcel?", top_k=3, use_rewrite=True)
        top_ids = [r["article_id"] for r in res["results"]]
        assert "A01" in top_ids or "A02" in top_ids
        assert res["results"][0]["similarity"] > 0.3

    def test_search_empty_input(self, engine):
        res = engine.search_articles("", top_k=5)
        assert res["results"] == []
        res_whitespace = engine.search_articles("   ")
        assert res_whitespace["results"] == []


class TestSimilarQueries:
    """Tests for similar customer queries detection."""

    def test_find_similar_queries(self, engine):
        pairs = engine.find_similar_queries(threshold=0.5)
        assert isinstance(pairs, list)
        assert len(pairs) > 0
        for pair in pairs:
            assert "query_1" in pair
            assert "query_2" in pair
            assert "similarity" in pair
            assert pair["similarity"] >= 0.5

    def test_threshold_filtering(self, engine):
        high_pairs = engine.find_similar_queries(threshold=0.7)
        low_pairs = engine.find_similar_queries(threshold=0.4)
        assert len(low_pairs) >= len(high_pairs)


class TestSimilarArticles:
    """Tests for duplicate or overlapping help articles."""

    def test_find_similar_articles(self, engine):
        pairs = engine.find_similar_articles(threshold=0.5)
        assert isinstance(pairs, list)
        assert len(pairs) >= 2
        for pair in pairs:
            assert "article_id_1" in pair
            assert "title_1" in pair
            assert "article_id_2" in pair
            assert "title_2" in pair
            assert "similarity" in pair
            assert "reason" in pair
            assert pair["similarity"] >= 0.5

        # Near duplicates A01 vs A02 and A03 vs A12
        pair_ids = [(p["article_id_1"], p["article_id_2"]) for p in pairs]
        assert ("A01", "A02") in pair_ids or ("A02", "A01") in pair_ids
        assert ("A03", "A12") in pair_ids or ("A12", "A03") in pair_ids


class TestOutdatedArticles:
    """Tests for outdated help articles audit."""

    def test_audit_outdated_articles(self, engine):
        outdated = engine.audit_outdated_articles(days_threshold=180)
        assert isinstance(outdated, list)
        assert len(outdated) >= 4
        for item in outdated:
            assert "article_id" in item
            assert "title" in item
            assert "updated_at" in item
            assert "days_since_update" in item
            assert item["days_since_update"] > 180

        # Articles from 2023/2024 should be flagged
        outdated_ids = [o["article_id"] for o in outdated]
        assert "A06" in outdated_ids
        assert "A11" in outdated_ids
        assert "A05" in outdated_ids

    def test_get_article_audit_summary(self, engine):
        audit = engine.get_article_audit(days_threshold=180, similarity_threshold=0.5)
        assert "outdated_articles" in audit
        assert "similar_articles" in audit
        assert "summary" in audit
        summary = audit["summary"]
        assert summary["total_articles"] == 12
        assert summary["outdated_count"] >= 4
        assert summary["similar_pair_count"] >= 2


class TestKnowledgeGaps:
    """Tests for knowledge gap detection and recommendations."""

    def test_uncovered_query_gap(self, engine):
        res = engine.get_knowledge_gap_recommendations(
            question="Do you accept cryptocurrency like Bitcoin for payment?",
            gap_threshold=0.30,
        )
        assert "results" in res
        assert len(res["results"]) == 1
        gap_item = res["results"][0]
        assert gap_item["is_gap"] is True
        assert gap_item["recommended_action"] == "Create new help article"
        assert "reason" in gap_item
        assert gap_item["highest_similarity"] < 0.30

    def test_covered_query_no_gap(self, engine):
        res = engine.get_knowledge_gap_recommendations(
            question="Where is my order?",
            gap_threshold=0.30,
        )
        assert "results" in res
        gap_item = res["results"][0]
        assert gap_item["is_gap"] is False
        assert "Topic adequately covered" in gap_item["recommended_action"]

    def test_dataset_wide_gaps(self, engine):
        res = engine.get_knowledge_gap_recommendations(gap_threshold=0.30)
        assert "results" in res
        assert res["total_gaps"] > 0
        for gap in res["results"]:
            assert "query" in gap
            assert "highest_similarity" in gap
            assert "reason" in gap
            assert gap["highest_similarity"] < 0.30


class TestEdgeCasesAndErrors:
    """Tests for edge cases and missing data."""

    def test_missing_articles_file(self):
        with pytest.raises(FileNotFoundError):
            AnalysisEngine(articles_path="non_existent_articles.csv")

    def test_invalid_article_columns(self, tmp_path):
        bad_csv = tmp_path / "bad_articles.csv"
        bad_csv.write_text("wrong_col1,wrong_col2\n1,2", encoding="utf-8")
        with pytest.raises(ValueError):
            AnalysisEngine(articles_path=str(bad_csv))
