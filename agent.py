"""
G2 Review Signal Agent — Gemini Flash via google-genai SDK.

This module implements the Antigravity agent reasoning layer described in the
G2 Review Signal Workflow Build Brief (Section 5 Stages 2, 3, 7 and Section 8).

All reasoning is delegated to the LLM via the google-genai SDK:
  Stage 2 — Pain extraction, pain category classification, pain summary, confidence scoring
  Stage 3 — Lead scoring across all three dimensions (recency, seniority, pain intensity),
             total score calculation, tier assignment
  Stage 7 — Personalized outreach email drafting

Requires GEMINI_API_KEY in .env.

The scoring rubrics and hard constraints from Section 5 are enforced entirely
in the system prompt. Python post-processing only validates types, enforces
numeric bounds, and recomputes totals to guarantee schema integrity.

Output always conforms to the 16-field JSON schema in Section 6.
"""

import json
import os
import re
from datetime import datetime
from typing import Any, Dict

from dotenv import load_dotenv
import google.genai as genai

load_dotenv()

# ---------------------------------------------------------------------------
# Backend config
# ---------------------------------------------------------------------------
MODEL_BACKEND: str = "gemini"
GEMINI_MODEL: str = "gemini-3.6-flash"

# ---------------------------------------------------------------------------
# System prompt
# ALL reasoning logic, scoring rubrics, and hard rules live here — not in Python
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a competitive displacement analyst for a B2B SaaS company.

You will receive structured data about a G2 review of a competitor product.
Your job is to perform three tasks and return a single, strictly-formatted JSON object.
You must return ONLY the JSON object — no markdown, no explanation, no extra text.

---

TASK 1 — PAIN EXTRACTION (Stage 2)

Read the review text carefully and determine:

pain_category: Classify the review into EXACTLY ONE of these categories:
  - Onboarding Friction
  - Poor Customer Support
  - Integration Failures
  - Pricing / Value Mismatch
  - Performance / Reliability Issues
  - Missing Features
  - Steep Learning Curve
  - Poor Reporting / Analytics
  - Clunky UI / UX
  - Lack of Scalability
  - Other  (use ONLY if the review genuinely does not map to any category above)

pain_summary: Write 1–2 plain English sentences describing the reviewer's core problem.
  This appears in a Slack notification read by a human — keep it readable and specific.
  Do NOT use vague language. Do NOT copy-paste the review verbatim.

confidence_score: A float between 0.0 and 1.0 representing how clearly the review
  maps to a single pain category.
  Scoring guide:
    0.85–1.00: The pain category is unmistakably obvious from the review text
    0.70–0.84: Strong signal with minor ambiguity
    0.50–0.69: Review is vague, mixed, or required significant interpretation
    Below 0.50: Review gives very little actionable signal

confidence_label: "high" if confidence_score >= 0.70, otherwise "low".
  These are the ONLY two allowed values. No other strings are valid.

---

TASK 2 — LEAD SCORING (Stage 3)

Apply the EXACT rubrics below. Do not estimate. Do not round differently.

RECENCY SCORE (max 30 points)
Compare the review_date to today's date (provided in the user message).
  - Posted within the last 7 days → 30
  - Posted 8–30 days ago → 20
  - Posted 31–90 days ago → 10
  - Posted more than 90 days ago → 0
  - Date is missing or cannot be parsed → assign 10

SENIORITY SCORE (max 40 points)
Evaluate the reviewer_title field:
  - CEO, CTO, CFO, COO, CRO, CMO, Chief [anything], VP, Vice President,
    Owner, Founder, President → 40
  - Director, Head of [anything], Principal → 30
  - Manager, Supervisor → 20
  - Any other clearly named individual contributor role → 10
  - Title is blank, missing, or "Unknown" → 5

PAIN INTENSITY SCORE (max 30 points)
Evaluate the emotional signal and urgency in the review text:
  - Reviewer explicitly mentions switching, cancelling, leaving, or uses words
    like "terrible", "worst", "nightmare", "unusable", "frustrated", "fed up",
    "wasted", "disaster" → 30
  - Reviewer makes a clear, specific criticism of a named feature, process, or
    outcome without extreme emotional language → 20
  - Reviewer expresses mild dissatisfaction or a mixed positive/negative review → 10
  - Review is mostly positive with only a minor complaint → 0

total_score = recency_score + seniority_score + pain_intensity_score
This must be an integer between 0 and 100.

tier assignment (based on total_score):
  - 70–100 → "Tier 1"
  - 40–69  → "Tier 2"
  - Below 40 → "Discard"

---

TASK 3 — OUTREACH EMAIL DRAFTING (Stage 7)

Draft a short, personalized outreach email following ALL of these rules:

HARD RULES (every single one is non-negotiable):
  1. NEVER reference G2, any review site, or imply you found the person through a review
  2. NEVER say anything like "noticed your review" or "saw your feedback"
  3. Open with the implied pain as the hook — not a generic opener
  4. Tone: direct, peer-to-peer. Not salesy. Not pushy.
  5. Length: STRICTLY under 100 words — count carefully before finalizing
  6. Do NOT include a subject line — email body only
  7. Address the reviewer by first name only (extract from reviewer_name field)
  8. Reference their company name naturally in the opening line
  9. Each pain category must produce a meaningfully different email:
     - Different opening hook
     - Different value proposition
     - Different call to action
     This differentiation is the most important demo moment in this workflow.

---

OUTPUT — CRITICAL FORMATTING RULE

Return ONLY a raw JSON object. The response must:
  - Start with {
  - End with }
  - Contain no markdown, no code fences, no explanation, no preamble

Required fields (exactly these 16):

{
  "reviewer_name": "<pass through unchanged from input>",
  "reviewer_title": "<pass through unchanged from input>",
  "reviewer_company": "<pass through unchanged from input>",
  "review_text": "<pass through unchanged from input>",
  "review_date": "<pass through unchanged from input, YYYY-MM-DD format>",
  "pain_category": "<exactly one of the 11 categories listed above>",
  "pain_summary": "<1–2 sentence plain English pain summary>",
  "confidence_score": <float 0.0–1.0, two decimal places>,
  "confidence_label": "<'high' or 'low' — no other values permitted>",
  "recency_score": <integer — must be one of: 0, 10, 20, 30>,
  "seniority_score": <integer — must be one of: 5, 10, 20, 30, 40>,
  "pain_intensity_score": <integer — must be one of: 0, 10, 20, 30>,
  "total_score": <integer 0–100, must equal the exact sum of the three scores above>,
  "tier": "<'Tier 1', 'Tier 2', or 'Discard' — no other values permitted>",
  "drafted_email": "<email body only, under 100 words, no subject line>",
  "competitor_product": "<pass through unchanged from input>"
}"""


def _build_user_message(
    reviewer_name: str,
    reviewer_title: str,
    reviewer_company: str,
    review_text: str,
    review_date: str,
    competitor_product: str,
) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    return (
        f"Today's date: {today}\n\n"
        "Analyze the following G2 review and return the JSON object per your instructions.\n\n"
        "---\n"
        f"Reviewer Name: {reviewer_name}\n"
        f"Reviewer Title: {reviewer_title}\n"
        f"Reviewer Company: {reviewer_company}\n"
        f"Review Date: {review_date if review_date else 'Not provided'}\n"
        f"Competitor Product: {competitor_product}\n\n"
        f"Review Text:\n{review_text if review_text else 'No review text provided.'}\n"
        "---\n\n"
        "Return only the JSON object. Nothing else."
    )


def _extract_json(raw_text: str) -> Dict[str, Any]:
    """
    Extracts and parses the JSON object from the model's raw text response.
    Handles three formats:
      1. Raw JSON: { ... }
      2. Markdown-wrapped with language tag: ```json { ... } ```
      3. Markdown-wrapped without language tag: ``` { ... } ```
    """
    cleaned = raw_text.strip()

    # Strategy 1: extract content from inside a markdown code fence (anywhere in text)
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass  # fall through to brace extraction on the fence content

    # Strategy 2: strip a leading fence if present, then do brace extraction
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned).strip()

    # Strategy 3: brace extraction from whatever remains
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(
            f"No JSON object found in model response. "
            f"Raw output (first 400 chars): {raw_text[:400]}"
        )

    return json.loads(cleaned[start:end])


def _validate_and_coerce(data: Dict[str, Any], fallback: Dict[str, str]) -> Dict[str, Any]:
    """
    Validates parsed model output and coerces types to match the Section 6 schema exactly.
    Pass-through fields are restored from original input if the model blanked them.
    total_score and tier are always recomputed from parts for integrity.
    """
    valid_categories = {
        "Onboarding Friction", "Poor Customer Support", "Integration Failures",
        "Pricing / Value Mismatch", "Performance / Reliability Issues", "Missing Features",
        "Steep Learning Curve", "Poor Reporting / Analytics", "Clunky UI / UX",
        "Lack of Scalability", "Other",
    }

    # Restore pass-through fields from original input
    for field in ("reviewer_name", "reviewer_title", "reviewer_company",
                  "review_text", "review_date", "competitor_product"):
        if not data.get(field):
            data[field] = fallback.get(field, "")

    # confidence_score — coerce to float, clamp to [0.0, 1.0]
    try:
        cs = float(data.get("confidence_score", 0.6))
        cs = max(0.0, min(1.0, round(cs, 2)))
    except (TypeError, ValueError):
        cs = 0.60
    data["confidence_score"] = cs

    # confidence_label — always derived from score; never trusted from model
    data["confidence_label"] = "high" if cs >= 0.70 else "low"

    # pain_category — validate against allowed set
    if data.get("pain_category") not in valid_categories:
        data["pain_category"] = "Other"

    # Integer scores — validate against allowed value sets
    def coerce_score(val: Any, allowed: set, default: int) -> int:
        try:
            v = int(val)
            return v if v in allowed else default
        except (TypeError, ValueError):
            return default

    data["recency_score"] = coerce_score(
        data.get("recency_score"), {0, 10, 20, 30}, 10
    )
    data["seniority_score"] = coerce_score(
        data.get("seniority_score"), {5, 10, 20, 30, 40}, 10
    )
    data["pain_intensity_score"] = coerce_score(
        data.get("pain_intensity_score"), {0, 10, 20, 30}, 10
    )

    # Recompute total_score and tier from parts — never trust model arithmetic
    total = (
        data["recency_score"]
        + data["seniority_score"]
        + data["pain_intensity_score"]
    )
    total = max(0, min(100, total))
    data["total_score"] = total

    if total >= 70:
        data["tier"] = "Tier 1"
    elif total >= 40:
        data["tier"] = "Tier 2"
    else:
        data["tier"] = "Discard"

    # pain_summary fallback
    if not data.get("pain_summary"):
        data["pain_summary"] = (
            f"Reviewer reported issues related to {data['pain_category'].lower()}."
        )

    # drafted_email fallback
    if not data.get("drafted_email"):
        first = (
            data["reviewer_name"].split()[0]
            if data.get("reviewer_name")
            else "there"
        )
        data["drafted_email"] = (
            f"Hi {first}, teams at {data['reviewer_company']} dealing with "
            f"{data['pain_category'].lower()} often look for a better alternative. "
            "Worth a quick conversation?"
        )

    return data


def _call_gemini(user_message: str) -> str:
    """
    Calls Gemini via the google-genai SDK. Requires GEMINI_API_KEY in .env.
    Models attempted in order: gemini-2.5-flash, then gemini-2.0-flash-lite.
    """
    load_dotenv(override=True)
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set or empty in .env. Please paste your Gemini API key in .env."
        )
    client = genai.Client(api_key=api_key)

    models_to_try = [GEMINI_MODEL, "gemini-3.5-flash", "gemini-flash-latest"]
    last_error = None

    for m in models_to_try:
        try:
            response = client.models.generate_content(
                model=m,
                contents=user_message,
                config=genai.types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.2,
                    max_output_tokens=8192,
                ),
            )
            return response.text
        except Exception as err:
            last_error = err
            err_str = str(err)
            if "404" in err_str or "NOT_FOUND" in err_str or "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                continue
            raise err

    if last_error:
        raise last_error
    raise RuntimeError("Failed to generate content with Gemini")


def _call_llm(user_message: str) -> str:
    """Dispatches to the configured LLM backend."""
    return _call_gemini(user_message)


class AntigravityReviewAgent:
    """
    G2 Review Signal Agent.

    Powered by Gemini Flash via google-genai SDK.
    Requires GEMINI_API_KEY in .env.

    The system prompt encodes all scoring rubrics, constraints, and output
    schema requirements from Section 5 (Stages 2, 3, 7) of the build brief.
    Python logic only handles type coercion and schema integrity.
    """

    def analyze(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        reviewer_name = str(payload.get("reviewer_name") or "Anonymous Reviewer").strip()
        reviewer_title = str(payload.get("reviewer_title") or "Unknown Role").strip()
        reviewer_company = str(payload.get("reviewer_company") or "Unknown Company").strip()
        review_text = str(payload.get("review_text") or "").strip()
        review_date = str(payload.get("review_date") or "").strip()
        competitor_product = str(
            payload.get("competitor_product") or "Competitor Product"
        ).strip()

        fallback = {
            "reviewer_name": reviewer_name,
            "reviewer_title": reviewer_title,
            "reviewer_company": reviewer_company,
            "review_text": review_text,
            "review_date": review_date or datetime.now().strftime("%Y-%m-%d"),
            "competitor_product": competitor_product,
        }

        user_message = _build_user_message(
            reviewer_name, reviewer_title, reviewer_company,
            review_text, review_date, competitor_product,
        )

        raw_response = _call_llm(user_message)
        parsed = _extract_json(raw_response)
        result = _validate_and_coerce(parsed, fallback)

        return result
