"""
Knowledge Base Improvement Advisory Agent (KnowledgeAI)
FastAPI Backend Application

Endpoints:
- GET /                    : Backend health check
- POST /search             : Relevant article search with TF-IDF and query rewriting
- GET /similar-queries     : Similar customer queries detection
- GET /article-audit       : Outdated and similar article audit
- POST /recommendations    : Knowledge gap recommendations and suggestions
"""

import os
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Query, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from analysis_engine import AnalysisEngine, rewrite_query

# Initialize FastAPI application
app = FastAPI(
    title="Knowledge Base Improvement Advisory Agent API",
    description=(
        "Backend API for analyzing help articles, identifying similar customer queries, "
        "auditing outdated/duplicate articles, and recommending knowledge base improvements."
    ),
    version="1.0.0",
)

# Enable CORS for cross-origin frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the analysis engine with default datasets
try:
    engine = AnalysisEngine()
except Exception as e:
    engine = None
    print(f"Warning: Could not initialize AnalysisEngine on startup: {e}")


# -----------------------------------------------------------------------------
# Pydantic Request & Response Models
# -----------------------------------------------------------------------------

class SearchRequest(BaseModel):
    question: str = Field(
        ...,
        description="The customer inquiry or question to search for in help articles.",
        examples=["Where is my order?"]
    )
    top_k: Optional[int] = Field(
        5,
        ge=1,
        le=20,
        description="Maximum number of top matching articles to return."
    )
    min_threshold: Optional[float] = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold (0.0 to 1.0)."
    )
    use_rewrite: Optional[bool] = Field(
        True,
        description="Whether to apply rule-based query rewriting before searching."
    )


class RecommendationRequest(BaseModel):
    question: Optional[str] = Field(
        None,
        description="Optional customer question to test for knowledge gaps. If omitted or empty, all queries in queries.csv are evaluated."
    )
    gap_threshold: Optional[float] = Field(
        0.30,
        ge=0.0,
        le=1.0,
        description="Similarity score threshold below which a topic is flagged as a knowledge gap."
    )


# -----------------------------------------------------------------------------
# Global Exception Handlers
# -----------------------------------------------------------------------------

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Formats HTTP exceptions consistently with error and message keys."""
    detail = exc.detail
    if isinstance(detail, dict):
        return JSONResponse(status_code=exc.status_code, content=detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": detail, "message": str(detail)},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catches unhandled server exceptions."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": f"An unexpected server error occurred: {str(exc)}",
        },
    )


def get_engine() -> AnalysisEngine:
    """Helper to ensure AnalysisEngine is loaded."""
    global engine
    if engine is None:
        try:
            engine = AnalysisEngine()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"AnalysisEngine could not load datasets: {str(e)}",
            )
    return engine


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------

@app.get(
    "/",
    summary="Backend Health Check",
    tags=["Health"],
)
def health_check() -> Dict[str, Any]:
    """
    Returns backend health status, API metadata, and dataset availability.
    """
    eng = get_engine()
    total_articles = len(eng.articles_df) if not eng.articles_df.empty else 0
    total_queries = len(eng.queries_df) if not eng.queries_df.empty else 0

    return {
        "status": "healthy",
        "service": "Knowledge Base Improvement Advisory Agent",
        "version": "1.0.0",
        "message": "KnowledgeAI backend is running smoothly.",
        "data_status": {
            "articles_count": total_articles,
            "queries_count": total_queries,
        },
    }


@app.post(
    "/search",
    summary="Search Relevant Articles",
    tags=["Search"],
)
def search_articles_endpoint(payload: SearchRequest) -> Dict[str, Any]:
    """
    Feature 1 & Feature 5:
    Accepts a customer question, performs vocabulary normalization (query rewriting),
    and searches existing help articles using TF-IDF and Cosine Similarity.
    """
    cleaned_question = payload.question.strip() if payload.question else ""
    if not cleaned_question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'question' field cannot be empty. Please provide a valid customer question.",
        )

    eng = get_engine()
    search_data = eng.search_articles(
        question=cleaned_question,
        top_k=payload.top_k or 5,
        min_threshold=payload.min_threshold or 0.0,
        use_rewrite=payload.use_rewrite if payload.use_rewrite is not None else True,
    )

    return search_data


@app.get(
    "/similar-queries",
    summary="Detect Similar Customer Queries",
    tags=["Queries Analysis"],
)
def get_similar_queries(
    threshold: float = Query(
        0.5,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for query clustering (default 0.5)."
    )
) -> Dict[str, Any]:
    """
    Feature 2:
    Identifies repeated or similar customer questions from the customer query log.
    Returns query pairs and their similarity scores.
    """
    eng = get_engine()
    pairs = eng.find_similar_queries(threshold=threshold)

    return {
        "total_pairs": len(pairs),
        "threshold": threshold,
        "results": pairs,
        "similar_queries": pairs,
    }


@app.get(
    "/article-audit",
    summary="Audit Outdated and Similar Articles",
    tags=["Article Audit"],
)
def article_audit(
    days_threshold: int = Query(
        180,
        ge=1,
        description="Review age threshold in days (articles older than this are flagged as outdated)."
    ),
    similarity_threshold: float = Query(
        0.5,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold to flag duplicate or highly similar articles."
    ),
) -> Dict[str, Any]:
    """
    Feature 3 & Feature 4:
    Performs a knowledge base audit:
    - Flags potentially outdated articles needing review based on update date.
    - Flags potentially duplicate or overlapping help articles.
    """
    eng = get_engine()
    audit_data = eng.get_article_audit(
        days_threshold=days_threshold,
        similarity_threshold=similarity_threshold,
    )

    return audit_data


@app.post(
    "/recommendations",
    summary="Knowledge Gap Recommendations",
    tags=["Recommendations"],
)
def get_recommendations_endpoint(
    payload: Optional[RecommendationRequest] = None,
) -> Dict[str, Any]:
    """
    Feature 6:
    Identifies knowledge gaps and provides improvement suggestions.
    - If a 'question' is provided in the JSON body, evaluates that specific question.
    - If omitted or empty, scans all queries in queries.csv for unaddressed customer topics.
    - Checks rewritten queries to prevent false alerts.
    """
    eng = get_engine()
    question = payload.question if payload else None
    gap_threshold = payload.gap_threshold if payload and payload.gap_threshold is not None else 0.30

    recommendation_data = eng.get_knowledge_gap_recommendations(
        question=question,
        gap_threshold=gap_threshold,
    )

    results_list = recommendation_data.get("results", [])
    rewritten = (
        results_list[0].get("rewritten_query", question or "")
        if results_list
        else (question or "Not provided")
    )

    # Format recommendations as clean rows for Streamlit table display
    table_rows = []
    for item in results_list:
        table_rows.append({
            "Topic": item.get("suggested_topic", ""),
            "Action": item.get("recommended_action", ""),
            "Similarity": f"{int(item.get('highest_similarity', 0.0) * 100)}%",
            "Reason": item.get("reason", ""),
        })

    return {
        "gap_threshold": gap_threshold,
        "total_gaps": recommendation_data.get("total_gaps", 0),
        "rewritten_query": rewritten,
        "recommendations": table_rows if table_rows else results_list,
        "results": results_list,
    }
