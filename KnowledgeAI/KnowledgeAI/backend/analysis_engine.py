"""
Knowledge Base Improvement Advisory Agent (KnowledgeAI)
Analysis Engine Module

Features implemented:
1. TF-IDF & Cosine Similarity search on help articles.
2. Pairwise similarity detection among customer queries.
3. Similar & duplicate article detection.
4. Outdated article auditing based on last update date and review threshold.
5. Rule-based query rewriting for vocabulary normalization.
6. Knowledge gap recommendations for missing or poorly covered customer queries.
"""

import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Default dataset paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ARTICLES_PATH = os.path.join(BASE_DIR, "data", "articles.csv")
DEFAULT_QUERIES_PATH = os.path.join(BASE_DIR, "data", "queries.csv")

# Vocabulary normalization mappings for rule-based query rewriting
# Preserves original terms and adds standard knowledge base keywords
VOCABULARY_EXPANSIONS = {
    r"\bparcels?\b": "parcel package shipment",
    r"\bshipments?\b": "shipment package shipping delivery",
    r"\bpasscodes?\b": "passcode password",
    r"\bpwd\b": "password",
    r"\bbroken goods\b": "broken goods damaged defective merchandise",
    r"\bbroken\b": "broken damaged defective",
    r"\bmoney back\b": "money back refund return",
    r"\bcash back\b": "cash back refund",
    r"\bstatus of order\b": "order tracking status",
    r"\bwhere is my\b": "where is my track tracking",
    r"\bvouchers?\b": "voucher gift card",
    r"\bcrypto\b": "crypto cryptocurrency bitcoin",
    r"\bchange order\b": "modify cancel order",
    r"\bcard\b": "card payment debit credit",
}


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def rewrite_query(query: str) -> str:
    """
    Feature 5: Rule-based vocabulary normalization.
    Expands or normalizes common customer terms to match knowledge base keywords.
    For example: 'parcel' -> 'parcel package shipment'
    """
    if not query or not isinstance(query, str):
        return ""

    cleaned = normalize_text(query)
    rewritten = cleaned

    for pattern, replacement in VOCABULARY_EXPANSIONS.items():
        rewritten = re.sub(pattern, replacement, rewritten)

    # Normalize whitespace
    rewritten = re.sub(r"\s+", " ", rewritten).strip()
    return rewritten


class AnalysisEngine:
    """
    Main analysis engine for Knowledge Base search, query similarity,
    article audit, and knowledge gap recommendations.
    """

    def __init__(
        self,
        articles_path: Optional[str] = None,
        queries_path: Optional[str] = None,
    ):
        self.articles_path = articles_path or DEFAULT_ARTICLES_PATH
        self.queries_path = queries_path or DEFAULT_QUERIES_PATH
        self.articles_df: pd.DataFrame = pd.DataFrame()
        self.queries_df: pd.DataFrame = pd.DataFrame()
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.article_matrix = None

        self.load_data()

    def load_data(self) -> None:
        """Loads and pre-indexes articles and customer queries datasets."""
        if not os.path.exists(self.articles_path):
            raise FileNotFoundError(f"Articles file not found at: {self.articles_path}")

        self.articles_df = pd.read_csv(self.articles_path)
        required_article_cols = {"article_id", "title", "content", "updated_at"}
        if not required_article_cols.issubset(self.articles_df.columns):
            missing = required_article_cols - set(self.articles_df.columns)
            raise ValueError(f"articles.csv missing required columns: {missing}")

        self.articles_df["article_id"] = self.articles_df["article_id"].astype(str)
        self.articles_df["title"] = self.articles_df["title"].fillna("").astype(str)
        self.articles_df["content"] = self.articles_df["content"].fillna("").astype(str)
        self.articles_df["updated_at"] = self.articles_df["updated_at"].fillna("").astype(str)

        # Combined text for TF-IDF indexing: weighting title twice gives stronger relevance
        self.articles_df["combined_text"] = (
            self.articles_df["title"] + " " + self.articles_df["title"] + " " + self.articles_df["content"]
        ).apply(normalize_text)

        # Initialize TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(stop_words="english")
        corpus = self.articles_df["combined_text"].tolist()
        if corpus and any(len(c.strip()) > 0 for c in corpus):
            self.article_matrix = self.vectorizer.fit_transform(corpus)
        else:
            self.article_matrix = None

        # Load queries if present
        if os.path.exists(self.queries_path):
            self.queries_df = pd.read_csv(self.queries_path)
            required_query_cols = {"query_id", "customer_query"}
            if required_query_cols.issubset(self.queries_df.columns):
                self.queries_df["query_id"] = self.queries_df["query_id"].astype(str)
                self.queries_df["customer_query"] = (
                    self.queries_df["customer_query"].fillna("").astype(str)
                )

    def search_articles(
        self,
        question: str,
        top_k: int = 5,
        min_threshold: float = 0.0,
        use_rewrite: bool = True,
    ) -> Dict[str, Any]:
        """
        Feature 1 & Feature 5:
        Search help articles using TF-IDF and Cosine Similarity.
        Applies query rewriting to boost search relevance.
        """
        if not question or not question.strip():
            return {
                "query": question or "",
                "rewritten_query": "",
                "results": [],
            }

        if self.article_matrix is None or self.vectorizer is None or self.articles_df.empty:
            return {
                "query": question,
                "rewritten_query": rewrite_query(question) if use_rewrite else question,
                "results": [],
            }

        rewritten = rewrite_query(question) if use_rewrite else normalize_text(question)

        # Search with both raw normalized query and rewritten query
        queries_to_test = [normalize_text(question)]
        if use_rewrite and rewritten != normalize_text(question):
            queries_to_test.append(rewritten)

        best_sims = None
        for q in queries_to_test:
            if not q:
                continue
            try:
                q_vec = self.vectorizer.transform([q])
                sims = cosine_similarity(q_vec, self.article_matrix).flatten()
                if best_sims is None:
                    best_sims = list(sims)
                else:
                    best_sims = [max(a, b) for a, b in zip(best_sims, sims)]
            except Exception:
                continue

        if best_sims is None:
            return {
                "query": question,
                "rewritten_query": rewritten,
                "results": [],
            }

        # Build results
        results = []
        for idx, sim in enumerate(best_sims):
            sim_score = round(float(sim), 2)
            if sim_score >= min_threshold:
                row = self.articles_df.iloc[idx]
                results.append(
                    {
                        "article_id": str(row["article_id"]),
                        "title": str(row["title"]),
                        "content": str(row["content"]),
                        "similarity": sim_score,
                    }
                )

        # Sort descending by similarity score
        results.sort(key=lambda x: x["similarity"], reverse=True)
        results = results[:top_k]

        return {
            "query": question,
            "rewritten_query": rewritten,
            "results": results,
        }

    def find_similar_queries(self, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Feature 2:
        Identify repeated or similar customer questions from queries.csv.
        Returns query pairs with query_1, query_2, and similarity score.
        """
        if self.queries_df.empty:
            return []

        raw_queries = self.queries_df["customer_query"].tolist()
        # Apply normalization and rewriting to capture semantic variants like 'parcel' vs 'order'
        processed_queries = [rewrite_query(q) for q in raw_queries]

        vec = TfidfVectorizer()
        try:
            tfidf_mat = vec.fit_transform(processed_queries)
        except ValueError:
            return []

        sim_matrix = cosine_similarity(tfidf_mat, tfidf_mat)
        n = len(raw_queries)
        pairs: List[Dict[str, Any]] = []

        for i in range(n):
            for j in range(i + 1, n):
                score = round(float(sim_matrix[i, j]), 2)
                if score >= threshold:
                    pairs.append(
                        {
                            "query_1": raw_queries[i],
                            "query_2": raw_queries[j],
                            "similarity": score,
                        }
                    )

        pairs.sort(key=lambda x: x["similarity"], reverse=True)
        return pairs

    def find_similar_articles(self, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Feature 3:
        Identify potentially duplicate or highly similar help articles.
        Returns article IDs, titles, and similarity scores.
        """
        if self.article_matrix is None or len(self.articles_df) < 2:
            return []

        sim_matrix = cosine_similarity(self.article_matrix, self.article_matrix)
        n = len(self.articles_df)
        pairs: List[Dict[str, Any]] = []

        for i in range(n):
            for j in range(i + 1, n):
                score = round(float(sim_matrix[i, j]), 2)
                if score >= threshold:
                    row_i = self.articles_df.iloc[i]
                    row_j = self.articles_df.iloc[j]
                    pairs.append(
                        {
                            "article_id_1": str(row_i["article_id"]),
                            "title_1": str(row_i["title"]),
                            "article_id_2": str(row_j["article_id"]),
                            "title_2": str(row_j["title"]),
                            "similarity": score,
                            "reason": f"High content overlap ({int(score * 100)}% similarity) detected between these articles.",
                        }
                    )

        pairs.sort(key=lambda x: x["similarity"], reverse=True)
        return pairs

    def audit_outdated_articles(
        self,
        days_threshold: int = 180,
        reference_date: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Feature 4:
        Flag articles older than days_threshold for human review.
        """
        if self.articles_df.empty:
            return []

        if reference_date is None:
            reference_date = datetime.now()

        outdated: List[Dict[str, Any]] = []
        for _, row in self.articles_df.iterrows():
            updated_str = str(row["updated_at"]).strip()
            days_elapsed = None

            try:
                dt = datetime.strptime(updated_str, "%Y-%m-%d")
                days_elapsed = (reference_date - dt).days
            except Exception:
                days_elapsed = 9999

            if days_elapsed is not None and days_elapsed > days_threshold:
                outdated.append(
                    {
                        "article_id": str(row["article_id"]),
                        "title": str(row["title"]),
                        "updated_at": updated_str,
                        "days_since_update": days_elapsed,
                        "reason": f"Article has not been updated in {days_elapsed} days (threshold is {days_threshold} days). Review recommended.",
                    }
                )

        outdated.sort(key=lambda x: x["days_since_update"], reverse=True)
        return outdated

    def get_article_audit(
        self,
        days_threshold: int = 180,
        similarity_threshold: float = 0.5,
        reference_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Combines outdated article detection and similar/duplicate article detection.
        Matches frontend audit dashboard requirements.
        """
        outdated = self.audit_outdated_articles(
            days_threshold=days_threshold, reference_date=reference_date
        )
        similar = self.find_similar_articles(threshold=similarity_threshold)

        # Build combined list of audit items for table display (e.g. Streamlit pd.DataFrame)
        combined_rows: List[Dict[str, Any]] = []
        for o in outdated:
            combined_rows.append({
                "issue_type": "Outdated Content",
                "article_id": o["article_id"],
                "title": o["title"],
                "updated_at": o["updated_at"],
                "similarity": "N/A",
                "reason": o["reason"],
            })
        for s in similar:
            combined_rows.append({
                "issue_type": "Similar / Duplicate",
                "article_id": f"{s['article_id_1']} & {s['article_id_2']}",
                "title": f"{s['title_1']} vs {s['title_2']}",
                "updated_at": "N/A",
                "similarity": f"{int(s['similarity'] * 100)}%",
                "reason": s["reason"],
            })

        return {
            "outdated_articles": outdated,
            "similar_articles": similar,
            "results": combined_rows,
            "summary": {
                "total_articles": len(self.articles_df),
                "outdated_count": len(outdated),
                "similar_pair_count": len(similar),
                "days_threshold": days_threshold,
                "similarity_threshold": similarity_threshold,
            },
        }

    def get_knowledge_gap_recommendations(
        self,
        question: Optional[str] = None,
        gap_threshold: float = 0.30,
    ) -> Dict[str, Any]:
        """
        Feature 6:
        Knowledge Gap Recommendations.
        If a question has low similarity with all existing articles (even after query rewriting),
        suggests creating a new article with understandable reasons.

        Can evaluate a single customer question or scan the entire queries.csv dataset.
        """
        if question and question.strip():
            raw_res = self.search_articles(question, top_k=1, use_rewrite=False)
            rewritten_res = self.search_articles(question, top_k=1, use_rewrite=True)

            raw_score = raw_res["results"][0]["similarity"] if raw_res["results"] else 0.0
            rewritten_score = (
                rewritten_res["results"][0]["similarity"] if rewritten_res["results"] else 0.0
            )
            best_score = max(raw_score, rewritten_score)
            rewritten_text = rewritten_res.get("rewritten_query", question)

            is_gap = best_score < gap_threshold

            if is_gap:
                action = "Create new help article"
                reason = (
                    f"No relevant help article found. Highest similarity score is {best_score:.2f} "
                    f"(below threshold {gap_threshold:.2f}) even after vocabulary normalization. "
                    "Customers asking this question cannot find answers."
                )
            else:
                top_art = rewritten_res["results"][0]
                action = "Topic adequately covered"
                reason = (
                    f"Relevant article '{top_art['title']}' (ID: {top_art['article_id']}) matches "
                    f"with a similarity score of {best_score:.2f}."
                )

            item = {
                "query": question,
                "rewritten_query": rewritten_text,
                "highest_similarity": best_score,
                "recommended_action": action,
                "suggested_topic": question.strip().rstrip("?").title(),
                "reason": reason,
                "is_gap": is_gap,
            }

            return {
                "results": [item],
                "total_gaps": 1 if is_gap else 0,
            }

        # If no specific question provided, scan queries.csv for unaddressed customer questions
        if self.queries_df.empty:
            return {"results": [], "total_gaps": 0}

        gap_recommendations: List[Dict[str, Any]] = []
        for _, qrow in self.queries_df.iterrows():
            q_text = str(qrow["customer_query"])
            res = self.search_articles(q_text, top_k=1, use_rewrite=True)
            top_sim = res["results"][0]["similarity"] if res["results"] else 0.0

            if top_sim < gap_threshold:
                rewritten_text = res.get("rewritten_query", q_text)
                gap_recommendations.append(
                    {
                        "query_id": str(qrow.get("query_id", "")),
                        "query": q_text,
                        "rewritten_query": rewritten_text,
                        "highest_similarity": top_sim,
                        "recommended_action": "Create new help article",
                        "suggested_topic": q_text.strip().rstrip("?").title(),
                        "reason": (
                            f"Customer query has maximum similarity score of {top_sim:.2f} "
                            f"(below threshold {gap_threshold:.2f}) across all articles, even after "
                            "rule-based query rewriting."
                        ),
                        "is_gap": True,
                    }
                )

        gap_recommendations.sort(key=lambda x: x["highest_similarity"])

        return {
            "results": gap_recommendations,
            "total_gaps": len(gap_recommendations),
        }
