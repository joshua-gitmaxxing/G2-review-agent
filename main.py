from dotenv import load_dotenv

load_dotenv(override=True)

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Any, Dict
from agent import AntigravityReviewAgent

app = FastAPI(
    title="G2 Review Signal Middleware",
    description="Middleware service connecting n8n triggers to Antigravity AI agent.",
    version="0.2.0",
)

agent_service = AntigravityReviewAgent()


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


HARDCODED_RESPONSE = {
    "reviewer_name": "Alex Smith",
    "reviewer_title": "Head of Operations",
    "reviewer_company": "TechCorp",
    "review_text": "Integration with our CRM breaks frequently, causing massive sync errors.",
    "review_date": "2026-08-05",
    "pain_category": "Integration Failures",
    "pain_summary": "CRM integration breaks frequently causing sync errors.",
    "confidence_score": 0.95,
    "confidence_label": "high",
    "recency_score": 10,
    "seniority_score": 30,
    "pain_intensity_score": 20,
    "total_score": 60,
    "tier": "Tier 2",
    "drafted_email": "Hi Alex, constant CRM sync errors can disrupt TechCorp's revenue operations. We built our platform to eliminate integration dropouts with real-time sync. Would you be open to a quick comparison?",
    "competitor_product": "CompetitorY"
}


@app.post("/analyze", response_model=AnalysisResponse)
def analyze_review(payload: Optional[Dict[str, Any]] = None):
    """
    Temporarily bypasses external LLM API calls and returns
    a hardcoded JSON response matching Section 6 schema.
    """
    return HARDCODED_RESPONSE


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
