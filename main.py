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
    reviewer_company: Optional[str] = None
    reviewer_company_size: Optional[str] = None
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
    drafted_email: Optional[str] = None
    competitor_product: str


@app.get("/")
def health_check():
    return {"status": "healthy", "service": "G2 Review Signal Middleware"}


@app.post("/analyze", response_model=AnalysisResponse)
def analyze_review(payload: Optional[Dict[str, Any]] = None):
    """
    Accepts raw G2 review payload from n8n / Apify and executes real Antigravity Agent logic
    for pain extraction, lead scoring, confidence check, tiering, and email generation.
    Returns structured JSON matching Section 6 schema.
    """
    if payload is None:
        payload = {}

    return agent_service.analyze(payload)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
