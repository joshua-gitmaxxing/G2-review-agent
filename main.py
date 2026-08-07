from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Any, Dict

app = FastAPI(
    title="G2 Review Signal Middleware",
    description="Middleware service connecting n8n triggers to AI reasoning layer.",
    version="0.1.0",
)


class AnalysisResponse(BaseModel):
    reviewer_name: str
    reviewer_title: str
    reviewer_company: str
    review_text: str
    review_date: str
    pain_category: str
    pain_summary: str
    confidence_score: float
    confidence_label: str
    recency_score: int
    seniority_score: int
    pain_intensity_score: int
    total_score: int
    tier: str
    drafted_email: str
    competitor_product: str


@app.get("/")
def health_check():
    return {"status": "healthy", "service": "G2 Review Signal Middleware"}


@app.post("/analyze", response_model=AnalysisResponse)
def analyze_review(payload: Optional[Dict[str, Any]] = None):
    """
    Accepts raw G2 review payload from n8n / Apify and returns
    structured analysis JSON matching Section 6 schema.
    Currently hardcoded for Step 1 validation.
    """
    if payload is None:
        payload = {}

    reviewer_name = str(payload.get("reviewer_name") or "Jane Doe")
    reviewer_title = str(payload.get("reviewer_title") or "VP of Sales")
    reviewer_company = str(payload.get("reviewer_company") or "Acme Corp")
    review_text = str(
        payload.get("review_text")
        or "The platform constantly crashes during peak hours and support takes days to reply."
    )
    review_date = str(payload.get("review_date") or "2026-08-01")
    competitor_product = str(payload.get("competitor_product") or "CompetitorX")

    return {
        "reviewer_name": reviewer_name,
        "reviewer_title": reviewer_title,
        "reviewer_company": reviewer_company,
        "review_text": review_text,
        "review_date": review_date,
        "pain_category": "Performance / Reliability Issues",
        "pain_summary": "Frequent platform downtime during high-traffic periods with delayed support response.",
        "confidence_score": 0.85,
        "confidence_label": "high",
        "recency_score": 30,
        "seniority_score": 40,
        "pain_intensity_score": 30,
        "total_score": 100,
        "tier": "Tier 1",
        "drafted_email": (
            "Hi Jane, noticed downtime can really disrupt sales momentum during peak hours. "
            "Our platform maintains 99.99% uptime with dedicated SLA support."
        ),
        "competitor_product": competitor_product,
    }
