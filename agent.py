"""
G2 Review Signal Agent — Gemini Flash via google-genai SDK.

This module implements the Antigravity agent reasoning layer described in the
G2 Review Signal Workflow Build Brief (Section 5 Stages 2, 3, 7 and Section 8).

All reasoning is delegated to the LLM via the google-genai SDK:
  Stage 2 — Pain extraction, pain category classification, pain summary, confidence scoring
  Stage 3 — Lead scoring across all three dimensions (recency, seniority, pain intensity),
             total score calculation, tier assignment
  Stage 7 — Deterministic, claim-safe outreach email drafting

Requires GEMINI_API_KEY in .env.

The model extracts and scores the review. Python then validates the result,
recomputes totals, applies qualification gates, and generates claim-safe email
copy from approved context.

Output always conforms to the 17-field JSON schema.
"""

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv
import google.genai as genai

load_dotenv()

# ---------------------------------------------------------------------------
# Backend config
# ---------------------------------------------------------------------------
MODEL_BACKEND: str = "gemini"
GEMINI_MODEL: str = "gemini-3.6-flash"

# ---------------------------------------------------------------------------
# Helpers for company identity and normalization
# ---------------------------------------------------------------------------
def is_g2_size_label(val: Optional[str]) -> bool:
    """Returns True if the value matches a known G2 company size label."""
    if not val or not isinstance(val, str):
        return False
    s = val.strip().lower()
    if s in {
        "small-business", "small business", "small_business",
        "mid-market", "mid market", "mid_market",
        "enterprise",
    }:
        return True
    if re.match(r"^(small[-_ ]?business|mid[-_ ]?market|enterprise)\s*\(.*\)$", s):
        return True
    if re.match(r"^(<=?\s*\d+|\d+\s*[-–]\s*\d+|\>\s*\d+|\d+\+?)\s*(emp\.?|employees)$", s):
        return True
    return False


def is_missing_company(val: Optional[str]) -> bool:
    """Returns True if the company value represents a missing or dummy company name."""
    if not val or not isinstance(val, str):
        return True
    s = val.strip().lower()
    if s in {"", "none", "null", "unknown", "unknown company", "n/a", "na", "undefined", "not provided"}:
        return True
    return False


def normalize_company_info(
    company: Any,
    company_size: Any,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Normalizes reviewer_company and reviewer_company_size:
    - Missing or placeholder company names (None, 'Unknown Company', 'None', etc.) become None.
    - Missing or placeholder size labels become None.
    - If reviewer_company contains a recognized G2 size label, move it to
      reviewer_company_size (if not already set) and leave reviewer_company as None.
    """
    norm_company = str(company).strip() if company is not None else None
    norm_size = str(company_size).strip() if company_size is not None else None

    if is_missing_company(norm_company):
        norm_company = None
    if is_missing_company(norm_size):
        norm_size = None

    if norm_company and is_g2_size_label(norm_company):
        if not norm_size:
            norm_size = norm_company
        norm_company = None

    return norm_company, norm_size


def email_treats_size_as_employer(email: str, company_size: Optional[str] = None) -> bool:
    """
    Checks if an email treats a company size value or placeholder as an employer name.
    Matches phrases like 'teams at Small-Business (50 or fewer emp.)' or 'at Mid-Market (51-1000 emp.)',
    while allowing legitimate descriptive phrases such as 'mid-market teams'.
    """
    if not email or not isinstance(email, str):
        return False

    # Check if the specific normalized company_size for this review is used after 'at' or 'teams at'
    if company_size and isinstance(company_size, str) and company_size.strip():
        escaped_size = re.escape(company_size.strip())
        if re.search(rf"\b(?:teams\s+at|at)\s+{escaped_size}\b", email, flags=re.IGNORECASE):
            return True

    # General check for any G2 size labels or parenthesized employee count suffixes after 'at' or 'teams at'
    size_pattern = (
        r"\b(?:teams\s+at|at)\s+"
        r"(?:small[-_ ]?business|mid[-_ ]?market|enterprise)"
        r"(?:\s*\([^)]*\))?"
    )
    if re.search(size_pattern, email, flags=re.IGNORECASE):
        return True

    # Check for placeholder strings after 'at' or 'teams at'
    placeholder_pattern = (
        r"\b(?:teams\s+at|at)\s+"
        r"(?:unknown\s+company|unknown|none|null)\b"
    )
    if re.search(placeholder_pattern, email, flags=re.IGNORECASE):
        return True

    return False


CATEGORY_TEMPLATES: Dict[str, Dict[str, str]] = {
    "Onboarding Friction": {
        "hook": "managing user onboarding friction can quickly create bottlenecks for {company_context}",
        "cta": "Is streamlining the onboarding process currently a priority for your team?",
    },
    "Poor Customer Support": {
        "hook": "experiencing customer support delays can be challenging for {company_context}",
        "cta": "Is improving support turnaround time something you are currently evaluating?",
    },
    "Integration Failures": {
        "hook": "dealing with integration issues can create operational friction for {company_context}",
        "cta": "Is improving integration reliability currently a priority?",
    },
    "Pricing / Value Mismatch": {
        "hook": "evaluating software pricing and contract value is often an ongoing focus for {company_context}",
        "cta": "Is evaluating software spend in this area currently a priority?",
    },
    "Performance / Reliability Issues": {
        "hook": "managing software performance and reliability challenges can disrupt day-to-day work for {company_context}",
        "cta": "Is improving platform stability currently a priority for your team?",
    },
    "Missing Features": {
        "hook": "encountering functional limitations can create workflow friction for {company_context}",
        "cta": "Are you currently exploring approaches to address those functional gaps?",
    },
    "Steep Learning Curve": {
        "hook": "navigating a steep software learning curve can slow down team adoption for {company_context}",
        "cta": "Is reducing software onboarding complexity currently a focus for your team?",
    },
    "Poor Reporting / Analytics": {
        "hook": "navigating reporting and analytics challenges can make tracking progress difficult for {company_context}",
        "cta": "Is improving reporting clarity something you are currently evaluating?",
    },
    "Clunky UI / UX": {
        "hook": "dealing with interface usability friction can impact daily efficiency for {company_context}",
        "cta": "Is improving day-to-day user experience a current priority for your team?",
    },
    "Lack of Scalability": {
        "hook": "managing system capacity constraints often becomes critical as {company_context} grows",
        "cta": "Are you currently evaluating approaches to handle increased volume and demand?",
    },
    "Other": {
        "hook": "resolving software friction and workflow challenges is often an ongoing effort for {company_context}",
        "cta": "Is addressing these operational challenges currently a priority for your team?",
    },
}


def _clean_pain_summary(
    summary: Optional[str],
    reviewer_name: Optional[str] = None,
) -> Optional[str]:
    """Cleans and validates pain_summary for natural inclusion in outreach drafts."""
    if not summary or not isinstance(summary, str):
        return None
    s = summary.strip()
    if not s:
        return None
    lower_s = s.lower()
    # Exclude generic boilerplate, source references, and third-person analyst
    # language that sounds unnatural in recipient-facing copy.
    if (
        "reviewer reported issues related to" in lower_s
        or "no actionable pain" in lower_s
        or "overwhelmingly positive" in lower_s
        or "g2" in lower_s
        or re.match(r"^(?:the\s+)?reviewer\b", lower_s)
    ):
        return None
    if reviewer_name:
        first_name = str(reviewer_name).strip().split()[0].lower()
        if first_name and re.match(rf"^{re.escape(first_name)}\b", lower_s):
            return None
    if not s.endswith((".", "!", "?")):
        s += "."
    return s


def _generate_fallback_email(
    data: Dict[str, Any],
    outreach_context: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Generates outreach email deterministically in code to prevent unsupported product claims (TEST-002).
    - If tier is 'Discard', returns None (TEST-004).
    - Distinct templates for each pain_category with unique hooks and CTAs.
    - Incorporates verified pain_summary naturally when available.
    - Never mentions G2 or any review site.
    - Never adds product claims unless they appear verbatim in approved value_propositions.
    - If only product_name exists without approved value propositions, makes no capability claims.
    - Preserves company-name and company-size protections (TEST-005).
    - Keeps drafts strictly under 100 words.
    """
    if data.get("tier") == "Discard":
        return None

    first = (
        data["reviewer_name"].split()[0]
        if data.get("reviewer_name") and data["reviewer_name"] != "Anonymous Reviewer"
        else "there"
    )
    company = data.get("reviewer_company")
    if company and not is_missing_company(company) and not is_g2_size_label(company):
        company_context = f"teams at {company}"
    else:
        company_context = "teams"

    raw_category = str(data.get("pain_category") or "Other").strip()
    template = CATEGORY_TEMPLATES.get(raw_category)
    if not template:
        for cat_name, cat_tmpl in CATEGORY_TEMPLATES.items():
            if cat_name.lower() == raw_category.lower():
                template = cat_tmpl
                break
        if not template:
            template = CATEGORY_TEMPLATES["Other"]

    clean_summary = _clean_pain_summary(
        data.get("pain_summary"),
        data.get("reviewer_name"),
    )

    # Extract and clean approved outreach context
    product_name = None
    approved_vps = []
    if outreach_context and isinstance(outreach_context, dict):
        pname = outreach_context.get("product_name")
        if pname and isinstance(pname, str) and pname.strip():
            product_name = pname.strip()

        vps = outreach_context.get("value_propositions")
        if isinstance(vps, list):
            approved_vps = [str(v).strip() for v in vps if str(v).strip()]
        elif isinstance(vps, str) and vps.strip():
            approved_vps = [vps.strip()]

    # Build context sentence (only verbatim approved propositions, zero invented claims)
    context_sentence = None
    if product_name and approved_vps:
        vp_text = "; ".join(approved_vps)
        context_sentence = f"At {product_name}, {vp_text}."
    elif approved_vps:
        vp_text = "; ".join(approved_vps)
        context_sentence = f"{vp_text}."
    elif product_name:
        # Zero capability claim: never claims built for, solves, or improves
        context_sentence = f"Reaching out from {product_name}."

    def build_email_with_context(ctx_str: str) -> str:
        hook_text = template["hook"].format(company_context=ctx_str)
        hook_sentence = hook_text[0].upper() + hook_text[1:] + "."
        parts = [f"Hi {first}, {hook_sentence}"]
        if clean_summary:
            parts.append(clean_summary)
        if context_sentence:
            parts.append(context_sentence)
        parts.append(template["cta"])
        return " ".join(parts)

    email = build_email_with_context(company_context)

    # TEST-005 protection: Never treat company size as employer name
    if email_treats_size_as_employer(email, data.get("reviewer_company_size")):
        email = build_email_with_context("teams")

    return email


# ---------------------------------------------------------------------------
# System Prompts
# ---------------------------------------------------------------------------
ANALYSIS_SYSTEM_PROMPT = """You are a competitive displacement analyst for a B2B SaaS company.

You will receive structured data about a G2 review of a competitor product.
Your job is to perform two tasks and return a single, strictly-formatted JSON object.
You must return ONLY the JSON object — no markdown, no explanation, no extra text.

---

TASK 1 — PAIN EXTRACTION (Stage 2)

Read the review carefully and extract the core buyer pain.

CRITICAL RULES FOR PAIN IDENTIFICATION (TEST-006 & TEST-001):
1. PRIORITIZE EXPLICIT DISLIKES: Unresolved buyer pain must be identified primarily from what the reviewer dislikes or struggles with.
2. DO NOT MISTAKE BENEFITS FOR PAIN: Do not treat features the reviewer liked, praises, or problems already solved by the competitor product as unresolved buyer pain. If a reviewer notes that a problem was solved by the competitor, that is a benefit, NOT a pain signal for displacement.
3. NO ACTIONABLE PAIN = DISCARD: If the review expresses only praise, positive feedback, or minor suggestions without genuine unresolved pain or dissatisfaction, set pain_intensity_score to 0.

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
  - Other  (use ONLY if the review genuinely does not map to any category above, or has no actionable pain)

pain_summary: Write 1–2 plain English sentences describing the reviewer's core problem.
  This appears in a Slack notification read by a human — keep it readable and specific.
  Do NOT use vague language. Do NOT copy-paste the review verbatim.
  If no actionable pain exists, state clearly that no actionable pain was identified.

confidence_score: A float between 0.0 and 1.0 representing how clearly the review
  maps to a single pain category.
  Scoring guide:
    0.85–1.00: The pain category is unmistakably obvious from the review text
    0.70–0.84: Strong signal with minor ambiguity
    0.50–0.69: Review is vague, mixed, or required significant interpretation
    Below 0.50: Review gives very little actionable signal

confidence_label: "high" if confidence_score >= 0.70, otherwise "low".
  These are the ONLY two allowed values. No other strings are valid.
  NOTE: Low confidence reviews with genuine pain are flagged for human review in Slack — do not discard them solely for low confidence.

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
Evaluate the emotional signal and urgency in the reviewer's dislikes and pain:
  - Reviewer explicitly mentions switching, cancelling, leaving, or uses words
    like "terrible", "worst", "nightmare", "unusable", "frustrated", "fed up",
    "wasted", "disaster" → 30
  - Reviewer makes a clear, specific criticism of a named feature, process, or
    outcome without extreme emotional language → 20
  - Reviewer expresses mild dissatisfaction or a mixed positive/negative review → 10
  - Review is mostly positive, expresses praise, solved problems, or has no actionable complaint → 0

QUALIFICATION GATE & TIER ASSIGNMENT (TEST-001):
- A review with NO ACTIONABLE PAIN (pain_intensity_score = 0) MUST ALWAYS BE ASSIGNED "Discard", regardless of recency or seniority score. A high seniority executive with a recent review that contains no pain is still "Discard".
- For reviews with genuine pain (pain_intensity_score > 0):
  - total_score = recency_score + seniority_score + pain_intensity_score (0–100)
  - 70–100 → "Tier 1"
  - 40–69  → "Tier 2"
  - Below 40 → "Discard"

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
  "reviewer_company": "<pass through unchanged from input, or null if not provided>",
  "reviewer_company_size": "<pass through unchanged from input, or null if not provided>",
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
  "competitor_product": "<pass through unchanged from input>"
}"""

EMAIL_SYSTEM_PROMPT = """You are a competitive displacement outreach copywriter for a B2B SaaS company.

Your job is to draft a short, personalized outreach email based on a qualified review of a competitor product.
You must return ONLY a raw JSON object with the single key "drafted_email" — no markdown, no explanation, no preamble.

---

TASK — OUTREACH EMAIL DRAFTING (Stage 7)

Draft a short, personalized outreach email following ALL of these rules:

HARD RULES (every single one is non-negotiable):
  1. NEVER reference G2, any review site, or imply you found the person through a review.
  2. NEVER say anything like "noticed your review" or "saw your feedback".
  3. Open with the implied pain as the hook — not a generic opener.
  4. Tone: direct, peer-to-peer. Not salesy. Not pushy.
  5. Length: STRICTLY under 100 words — count carefully before finalizing.
  6. Do NOT include a subject line — email body only.
  7. Address the reviewer by first name only (extract from reviewer_name field).
  8. Reference their company name naturally in the opening line ONLY if a verified company name is provided (TEST-005).
     If reviewer_company is null, missing, or "Not provided":
       - Do NOT invent or guess a company name.
       - NEVER use company size (such as "Small-Business", "Mid-Market", "Enterprise"), "Unknown Company", "None", or "null" as an employer name.
       - Instead, address their role or team generally (e.g., "teams dealing with...", "leaders dealing with...").
  9. PRODUCT CLAIMS & APPROVED CONTEXT (TEST-002):
     - NEVER invent features, pricing, integrations, capabilities, or performance claims.
     - If approved outreach context (approved product name, approved value propositions) is provided in the prompt:
       - You may mention the approved product name and use ONLY the approved value propositions.
       - Do NOT embellish or make claims beyond the approved value propositions.
     - If approved outreach context is NOT provided (or marked as not provided):
       - You MUST produce a neutral, pain-focused draft focusing entirely on the reviewer's problem and empathy.
       - Do NOT make any product claims, feature claims, integration promises, or performance statistics.
  10. Each pain category must produce a meaningfully different email (different hook, different perspective, different CTA).

---

OUTPUT FORMAT:
Return ONLY a JSON object:
{
  "drafted_email": "<email body only, under 100 words, no subject line>"
}"""

# Backward compatibility alias
SYSTEM_PROMPT = ANALYSIS_SYSTEM_PROMPT


def _build_analysis_user_message(
    reviewer_name: str,
    reviewer_title: str,
    reviewer_company: Optional[str],
    reviewer_company_size: Optional[str],
    review_text: str,
    review_date: str,
    competitor_product: str,
    likes: Optional[str] = None,
    dislikes: Optional[str] = None,
    recommendations: Optional[str] = None,
    problems_solved: Optional[str] = None,
) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    company_display = reviewer_company if reviewer_company else "Not provided"
    size_display = reviewer_company_size if reviewer_company_size else "Not provided"

    message_parts = [
        f"Today's date: {today}\n",
        "Analyze the following G2 review and return the JSON object per your instructions.\n",
        "---",
        f"Reviewer Name: {reviewer_name}",
        f"Reviewer Title: {reviewer_title}",
        f"Reviewer Company: {company_display}",
        f"Reviewer Company Size: {size_display}",
        f"Review Date: {review_date if review_date else 'Not provided'}",
        f"Competitor Product: {competitor_product}\n",
    ]

    has_structured = any(f for f in (dislikes, likes, problems_solved, recommendations) if f and f.strip())
    if has_structured:
        message_parts.append("STRUCTURED REVIEW FEEDBACK (TEST-006):")
        if dislikes and dislikes.strip():
            message_parts.append(f"Review Dislikes (Complaints / Pain): {dislikes.strip()}")
        if likes and likes.strip():
            message_parts.append(f"Review Likes (Benefits / What they like): {likes.strip()}")
        if problems_solved and problems_solved.strip():
            message_parts.append(f"Problems Solved by Competitor: {problems_solved.strip()}")
        if recommendations and recommendations.strip():
            message_parts.append(f"Recommendations to Others: {recommendations.strip()}")
        if review_text and review_text.strip():
            message_parts.append(f"Additional Review Text: {review_text.strip()}")
    else:
        message_parts.append(f"Review Text:\n{review_text if review_text else 'No review text provided.'}")

    message_parts.append("---\n\nReturn only the JSON object. Nothing else.")
    return "\n".join(message_parts)


def _build_user_message(
    reviewer_name: str,
    reviewer_title: str,
    reviewer_company: Optional[str],
    reviewer_company_size: Optional[str],
    review_text: str,
    review_date: str,
    competitor_product: str,
    likes: Optional[str] = None,
    dislikes: Optional[str] = None,
    recommendations: Optional[str] = None,
    problems_solved: Optional[str] = None,
) -> str:
    """Backward-compatible wrapper for building the analysis user message."""
    return _build_analysis_user_message(
        reviewer_name=reviewer_name,
        reviewer_title=reviewer_title,
        reviewer_company=reviewer_company,
        reviewer_company_size=reviewer_company_size,
        review_text=review_text,
        review_date=review_date,
        competitor_product=competitor_product,
        likes=likes,
        dislikes=dislikes,
        recommendations=recommendations,
        problems_solved=problems_solved,
    )


def _build_email_user_message(
    data: Dict[str, Any],
    outreach_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Builds the prompt for Stage 7 outreach email drafting."""
    company_display = data.get("reviewer_company") if data.get("reviewer_company") else "Not provided"
    size_display = data.get("reviewer_company_size") if data.get("reviewer_company_size") else "Not provided"

    message_parts = [
        f"Reviewer Name: {data.get('reviewer_name', 'Anonymous Reviewer')}",
        f"Reviewer Title: {data.get('reviewer_title', 'Unknown Role')}",
        f"Reviewer Company: {company_display}",
        f"Reviewer Company Size: {size_display}",
        f"Pain Category: {data.get('pain_category', 'Other')}",
        f"Pain Summary: {data.get('pain_summary', '')}",
        f"Competitor Product: {data.get('competitor_product', 'Competitor Product')}",
        f"Review Text: {data.get('review_text', '')}\n",
    ]

    product_name = None
    value_props = None
    if outreach_context and isinstance(outreach_context, dict):
        product_name = outreach_context.get("product_name")
        value_props = outreach_context.get("value_propositions")

    if product_name or value_props:
        message_parts.append("APPROVED OUTREACH CONTEXT (TEST-002):")
        if product_name:
            message_parts.append(f"Approved Product Name: {product_name}")
        if value_props:
            if isinstance(value_props, list):
                props_str = "; ".join(str(p) for p in value_props)
            else:
                props_str = str(value_props)
            message_parts.append(f"Approved Value Propositions: {props_str}")
        message_parts.append("Rule: Use ONLY this approved context. Do NOT invent unapproved features, pricing, or claims.\n")
    else:
        message_parts.append(
            "OUTREACH CONTEXT: None provided. You MUST produce a neutral, pain-focused draft without product claims (TEST-002).\n"
        )

    message_parts.append("Draft the outreach email following your system instructions. Return ONLY the JSON object with key 'drafted_email'.")
    return "\n".join(message_parts)


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


def _extract_email_text(raw_response: str) -> str:
    """Extracts drafted email body from model response, whether returned as JSON or raw text."""
    cleaned = raw_response.strip()
    try:
        parsed = _extract_json(cleaned)
        if isinstance(parsed, dict) and "drafted_email" in parsed:
            return str(parsed["drafted_email"] or "").strip()
    except Exception:
        pass

    # If markdown fences wrap plain text, strip them
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned).strip()
    return cleaned


def _validate_and_coerce(
    data: Dict[str, Any],
    fallback: Dict[str, str],
    outreach_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Validates parsed model output and coerces types to match the Section 6 schema exactly.
    Pass-through fields are restored from original input if the model blanked them.
    total_score and tier are always recomputed from parts for integrity.
    Enforces qualification gate: no actionable pain (pain_intensity_score == 0) -> Discard (TEST-001).
    Discards leads receive drafted_email: None (TEST-004).
    """
    valid_categories = {
        "Onboarding Friction", "Poor Customer Support", "Integration Failures",
        "Pricing / Value Mismatch", "Performance / Reliability Issues", "Missing Features",
        "Steep Learning Curve", "Poor Reporting / Analytics", "Clunky UI / UX",
        "Lack of Scalability", "Other",
    }

    # Restore pass-through fields from original input
    for field in ("reviewer_name", "reviewer_title",
                  "review_text", "review_date", "competitor_product"):
        if not data.get(field):
            data[field] = fallback.get(field, "")

    # Restore both company fields from normalized fallback (TEST-005).
    # NEVER trust the model to supply, invent, or alter identity information.
    data["reviewer_company"] = fallback.get("reviewer_company")
    data["reviewer_company_size"] = fallback.get("reviewer_company_size")

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

    # Recompute total_score from parts — never trust model arithmetic
    total = (
        data["recency_score"]
        + data["seniority_score"]
        + data["pain_intensity_score"]
    )
    total = max(0, min(100, total))
    data["total_score"] = total

    # Qualification gate (TEST-001):
    # A review with no actionable pain (pain_intensity_score == 0) must be Discard,
    # regardless of recency or seniority.
    if data["pain_intensity_score"] == 0:
        data["tier"] = "Discard"
    elif total >= 70:
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

    # drafted_email generation and validation (TEST-002, TEST-004, TEST-005):
    # Enforces in code that an LLM-hallucinated claim cannot reach drafted_email.
    if data["tier"] == "Discard":
        data["drafted_email"] = None
    else:
        data["drafted_email"] = _generate_fallback_email(data, outreach_context)

    return data



def _call_gemini(user_message: str, system_prompt: Optional[str] = None) -> str:
    """
    Calls Gemini via the google-genai SDK. Requires GEMINI_API_KEY in .env.
    Models attempted in order: gemini-3.6-flash, gemini-3.5-flash, gemini-flash-latest.
    """
    load_dotenv(override=True)
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set or empty in .env. Please paste your Gemini API key in .env."
        )
    client = genai.Client(api_key=api_key)

    models_to_try = [
        GEMINI_MODEL,          # gemini-3.6-flash (primary)
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-flash-lite-latest",
    ]
    last_error = None
    instruction = system_prompt or ANALYSIS_SYSTEM_PROMPT

    for m in models_to_try:
        try:
            response = client.models.generate_content(
                model=m,
                contents=user_message,
                config=genai.types.GenerateContentConfig(
                    system_instruction=instruction,
                    temperature=0.2,
                    max_output_tokens=8192,
                ),
            )
            # response.text raises ValueError if the response is blocked or empty
            # (e.g. finish_reason=SAFETY). Treat this the same as a 404/429 and
            # try the next model rather than crashing.
            try:
                text = response.text
            except (ValueError, AttributeError) as text_err:
                last_error = text_err
                continue

            if not text or not text.strip():
                last_error = ValueError(f"Model {m} returned an empty response.")
                continue

            return text

        except Exception as err:
            last_error = err
            err_str = str(err)
            if any(token in err_str for token in ("404", "NOT_FOUND", "429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE")):
                continue
            raise err

    if last_error:
        raise last_error
    raise RuntimeError("Failed to generate content with Gemini")


def _call_llm(user_message: str, system_prompt: Optional[str] = None) -> str:
    """Dispatches to the configured LLM backend."""
    return _call_gemini(user_message, system_prompt=system_prompt)


class AntigravityReviewAgent:
    """
    G2 Review Signal Agent.

    Powered by Gemini Flash via google-genai SDK.
    Requires GEMINI_API_KEY in .env.

    Stages:
      Stage 2 & 3: Pain extraction & lead scoring
      Python Qualification Gate (TEST-001): zero pain -> Discard
      Stage 7: Outreach email drafting for qualified leads only (TEST-004)
    """

    def __init__(self, outreach_context: Optional[Dict[str, Any]] = None):
        self.outreach_context = outreach_context

    def analyze(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if payload is None:
            payload = {}

        reviewer_name = str(payload.get("reviewer_name") or "Anonymous Reviewer").strip()
        reviewer_title = str(payload.get("reviewer_title") or "Unknown Role").strip()
        reviewer_company, reviewer_company_size = normalize_company_info(
            payload.get("reviewer_company"),
            payload.get("reviewer_company_size"),
        )
        review_date = str(payload.get("review_date") or "").strip()
        competitor_product = str(
            payload.get("competitor_product") or "Competitor Product"
        ).strip()

        # Structured review feedback (TEST-006)
        likes = str(payload.get("likes") or "").strip() or None
        dislikes = str(payload.get("dislikes") or "").strip() or None
        recommendations = str(payload.get("recommendations") or "").strip() or None
        problems_solved = str(
            payload.get("problemsSolved") or payload.get("problems_solved") or ""
        ).strip() or None

        raw_review_text = str(payload.get("review_text") or "").strip()
        if not raw_review_text:
            parts = []
            if dislikes:
                parts.append(f"Dislikes: {dislikes}")
            if likes:
                parts.append(f"Likes: {likes}")
            if problems_solved:
                parts.append(f"Problems Solved: {problems_solved}")
            review_text = " | ".join(parts) if parts else ""
        else:
            review_text = raw_review_text

        # Outreach context (TEST-002): payload takes precedence over instance default
        outreach_ctx = payload.get("outreach_context") or self.outreach_context
        if not outreach_ctx:
            prod_name = payload.get("product_name")
            val_props = payload.get("value_propositions")
            if prod_name or val_props:
                outreach_ctx = {
                    "product_name": prod_name,
                    "value_propositions": val_props,
                }

        fallback = {
            "reviewer_name": reviewer_name,
            "reviewer_title": reviewer_title,
            "reviewer_company": reviewer_company,
            "reviewer_company_size": reviewer_company_size,
            "review_text": review_text,
            "review_date": review_date or datetime.now().strftime("%Y-%m-%d"),
            "competitor_product": competitor_product,
        }

        # Stage 1: Pain extraction & scoring LLM call
        analysis_user_message = _build_analysis_user_message(
            reviewer_name=reviewer_name,
            reviewer_title=reviewer_title,
            reviewer_company=reviewer_company,
            reviewer_company_size=reviewer_company_size,
            review_text=review_text,
            review_date=review_date,
            competitor_product=competitor_product,
            likes=likes,
            dislikes=dislikes,
            recommendations=recommendations,
            problems_solved=problems_solved,
        )

        raw_analysis = _call_llm(analysis_user_message, system_prompt=ANALYSIS_SYSTEM_PROMPT)
        parsed_analysis = _extract_json(raw_analysis)
        result = _validate_and_coerce(parsed_analysis, fallback, outreach_context=outreach_ctx)

        # Email drafting and claim prevention is deterministically enforced in _validate_and_coerce
        # (TEST-001, TEST-002, TEST-004). No email LLM call is made.
        return result
