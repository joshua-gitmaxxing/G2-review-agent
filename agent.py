import os
import json
from datetime import datetime, date
from typing import Dict, Any, Tuple

CATEGORIES = [
    "Onboarding Friction",
    "Poor Customer Support",
    "Integration Failures",
    "Pricing / Value Mismatch",
    "Performance / Reliability Issues",
    "Missing Features",
    "Steep Learning Curve",
    "Poor Reporting / Analytics",
    "Clunky UI / UX",
    "Lack of Scalability",
    "Other",
]


class AntigravityReviewAgent:
    """
    Antigravity Reasoning Agent for G2 Review Signal Processing.
    Handles Stage 2 (Pain Extraction & Confidence), Stage 3 (Lead Scoring & Tiering),
    and Stage 7 (Campaign Outreach Generation).
    """

    def analyze(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        reviewer_name = str(payload.get("reviewer_name") or "Anonymous Reviewer").strip()
        reviewer_title = str(payload.get("reviewer_title") or "Unknown Role").strip()
        reviewer_company = str(payload.get("reviewer_company") or "Unknown Company").strip()
        review_text = str(payload.get("review_text") or "").strip()
        review_date = str(payload.get("review_date") or "").strip()
        competitor_product = str(payload.get("competitor_product") or "Competitor Product").strip()

        # If LLM API Key is set, attempt LLM call; fallback to local agent engine
        llm_result = self._try_llm_inference(
            reviewer_name, reviewer_title, reviewer_company, review_text, review_date, competitor_product
        )
        if llm_result:
            return llm_result

        # Stage 2: Pain Extraction & Confidence
        pain_category, pain_summary, confidence_score = self._extract_pain(review_text)
        confidence_label = "high" if confidence_score >= 0.70 else "low"

        # Stage 3: Lead Scoring & Tiering
        recency_score = self._score_recency(review_date)
        seniority_score = self._score_seniority(reviewer_title)
        pain_intensity_score = self._score_pain_intensity(review_text)

        total_score = recency_score + seniority_score + pain_intensity_score

        if total_score >= 70:
            tier = "Tier 1"
        elif total_score >= 40:
            tier = "Tier 2"
        else:
            tier = "Discard"

        # Stage 7: Campaign Generation
        first_name = reviewer_name.split()[0] if reviewer_name else "there"
        drafted_email = self._generate_email(
            first_name=first_name,
            company=reviewer_company,
            pain_category=pain_category,
            pain_summary=pain_summary,
            competitor_product=competitor_product,
        )

        return {
            "reviewer_name": reviewer_name,
            "reviewer_title": reviewer_title,
            "reviewer_company": reviewer_company,
            "review_text": review_text,
            "review_date": review_date or datetime.now().strftime("%Y-%m-%d"),
            "pain_category": pain_category,
            "pain_summary": pain_summary,
            "confidence_score": round(confidence_score, 2),
            "confidence_label": confidence_label,
            "recency_score": recency_score,
            "seniority_score": seniority_score,
            "pain_intensity_score": pain_intensity_score,
            "total_score": total_score,
            "tier": tier,
            "drafted_email": drafted_email,
            "competitor_product": competitor_product,
        }

    def _extract_pain(self, review_text: str) -> Tuple[str, str, float]:
        text_lower = review_text.lower()
        if not text_lower:
            return "Other", "No detailed review text provided.", 0.50

        category_matches = {
            "Performance / Reliability Issues": ["crash", "down", "bug", "error", "slow", "latency", "unstable", "outage", "freeze"],
            "Integration Failures": ["integration", "api", "sync", "connector", "webhook", "crm", "zapier"],
            "Poor Customer Support": ["support", "ticket", "response time", "helpdesk", "customer service", "agent", "unresponsive"],
            "Pricing / Value Mismatch": ["price", "expensive", "cost", "subscription", "hidden fee", "billing", "renew"],
            "Onboarding Friction": ["onboard", "setup", "implementation", "kickoff", "getting started"],
            "Missing Features": ["missing", "wish", "feature request", "lack", "doesn't support", "cannot"],
            "Steep Learning Curve": ["learning curve", "complex", "confusing", "hard to learn", "documentation"],
            "Poor Reporting / Analytics": ["report", "analytics", "dashboard", "export", "metric", "insight"],
            "Clunky UI / UX": ["ui", "ux", "interface", "clunky", "outdated", "navigation"],
            "Lack of Scalability": ["scale", "enterprise", "volume", "limit", "capacity"],
        }

        matched_cat = "Other"
        max_hits = 0
        for cat, keywords in category_matches.items():
            hits = sum(1 for kw in keywords if kw in text_lower)
            if hits > max_hits:
                max_hits = hits
                matched_cat = cat

        if matched_cat != "Other" and max_hits >= 2:
            confidence = 0.85
        elif matched_cat != "Other" and max_hits == 1:
            confidence = 0.72
        else:
            confidence = 0.55

        # Format summary
        summary = f"Reviewer expressed friction related to {matched_cat.lower()}."
        if "crash" in text_lower or "sync" in text_lower or "support" in text_lower:
            summary = review_text[:120].strip() + ("..." if len(review_text) > 120 else "")

        return matched_cat, summary, confidence

    def _score_recency(self, review_date_str: str) -> int:
        if not review_date_str:
            return 20

        try:
            dt = datetime.strptime(review_date_str.strip()[:10], "%Y-%m-%d").date()
            delta_days = (date.today() - dt).days
            if delta_days < 0:
                delta_days = 0

            if delta_days <= 7:
                return 30
            elif delta_days <= 30:
                return 20
            elif delta_days <= 90:
                return 10
            else:
                return 0
        except Exception:
            lower = review_date_str.lower()
            if "today" in lower or "yesterday" in lower or "day" in lower:
                return 30
            elif "week" in lower or "month" in lower:
                return 20
            return 10

    def _score_seniority(self, title: str) -> int:
        if not title:
            return 5
        t_lower = title.lower()

        c_suite = ["c-level", "chief", "ceo", "cto", "cfo", "coo", "cro", "cmo", "vp", "vice president", "owner", "founder", "president"]
        director = ["director", "head of", "lead", "principal"]
        manager = ["manager", "supervisor"]

        for kw in c_suite:
            if kw in t_lower:
                return 40
        for kw in director:
            if kw in t_lower:
                return 30
        for kw in manager:
            if kw in t_lower:
                return 20
        if len(t_lower.strip()) > 0 and t_lower != "unknown":
            return 10
        return 5

    def _score_pain_intensity(self, review_text: str) -> int:
        t_lower = review_text.lower()
        if not t_lower:
            return 10

        high_pain = ["cancel", "switch", "terrible", "worst", "unusable", "nightmare", "constant crash", "frustrated", "leaving"]
        med_pain = ["issue", "problem", "broken", "annoying", "disappointed", "slow", "fail"]

        if any(kw in t_lower for kw in high_pain):
            return 30
        elif any(kw in t_lower for kw in med_pain):
            return 20
        elif len(t_lower) > 20:
            return 10
        return 0

    def _generate_email(
        self, first_name: str, company: str, pain_category: str, pain_summary: str, competitor_product: str
    ) -> str:
        """
        Drafts outreach email following Stage 7 rules:
        - No G2 reference
        - Opens with implied pain
        - Peer-to-peer tone
        - Under 100 words
        - No subject line
        - Distinct per pain category
        """
        hooks = {
            "Performance / Reliability Issues": (
                f"Hi {first_name}, handling unexpected system downtime and reliability issues can stall critical workflows at {company}. "
                f"Teams evaluating alternatives to {competitor_product} often need high-availability architecture with zero latency spikes. "
                "Our platform delivers 99.99% uptime with enterprise SLAs. Open to comparing reliability benchmark data this week?"
            ),
            "Integration Failures": (
                f"Hi {first_name}, broken data syncs and failing CRM connectors usually create headaches for operational teams at {company}. "
                f"If syncing data across your tech stack with {competitor_product} has required manual workarounds, our native webhooks "
                "and pre-built connectors resolve those sync errors instantly. Worth a 5-minute look?"
            ),
            "Poor Customer Support": (
                f"Hi {first_name}, delayed support ticket responses when dealing with business-critical tools can freeze key projects at {company}. "
                f"We hear from teams moving away from {competitor_product} that dedicated engineering support makes all the difference. "
                "Every customer gets a dedicated Slack channel with under 15-minute response times. Open to connecting?"
            ),
            "Pricing / Value Mismatch": (
                f"Hi {first_name}, unexpected price escalations without matching product value can make scaling difficult at {company}. "
                f"If {competitor_product}'s current tiering feels restrictive, our transparent seat-based model keeps ROI predictable. "
                "Happy to share a quick price-to-performance breakdown if helpful."
            ),
            "Missing Features": (
                f"Hi {first_name}, hitting feature limits in your workflow software usually slows down key initiatives at {company}. "
                f"We built advanced customization specifically for teams outgrowing {competitor_product}'s core feature set. "
                "Would you be open to seeing how we plug those functional gaps?"
            ),
        }

        default_email = (
            f"Hi {first_name}, scaling operations at {company} often brings unexpected bottlenecks when legacy tools fall short. "
            f"If your current setup with {competitor_product} isn't keeping pace with your team's requirements, our platform is designed "
            "for seamless scalability and fast execution. Worth a quick conversation?"
        )

        return hooks.get(pain_category, default_email)

    def _try_llm_inference(self, *args, **kwargs) -> Any:
        # Return None to use high-performance local agent reasoning engine unless external credentials set
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        # Optional LLM integration point
        return None
